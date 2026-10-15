"""Run resource sampling and synchronized inference measurements.

GPU samples describe device-wide activity, while PyTorch memory describes this
process on its current device. Sampling is optional when nvidia-smi is unavailable.
"""

import json
import math
import os
import resource
import subprocess
import threading
import time
import uuid
from pathlib import Path

import torch

from .provenance import atomic_json


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gpu_memory():
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def distribution(values):
    values = sorted(values)
    if not values:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "max": None,
            "sum": 0,
        }

    def percentile(q):
        x = (len(values) - 1) * q
        lo, hi = math.floor(x), math.ceil(x)
        return values[lo] + (values[hi] - values[lo]) * (x - lo)

    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "max": values[-1],
        "sum": sum(values),
    }


def prediction_fields(
    text,
    generated,
    input_tokens,
    max_tokens,
    eos_ids,
    generation_seconds,
    preprocessing_seconds,
    item_seconds,
):
    eos_ids = eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids]
    count = len(generated)
    return {
        "prediction": text,
        "generated_tokens": count,
        "input_tokens": input_tokens,
        "output_characters": len(text),
        "output_words": len(text.split()),
        "hit_token_limit": bool(
            count >= max_tokens and int(generated[-1]) not in eos_ids
        ),
        "generation_seconds": generation_seconds,
        "preprocessing_seconds": preprocessing_seconds,
        "item_seconds": item_seconds,
        "generation_tokens_per_second": count / generation_seconds
        if generation_seconds > 0
        else None,
        "gpu_memory": gpu_memory(),
    }


def prediction_summary(rows):
    rows = list(rows)
    fields = (
        "generated_tokens",
        "input_tokens",
        "output_characters",
        "output_words",
        "generation_seconds",
        "preprocessing_seconds",
        "item_seconds",
    )
    stats = {
        key: distribution([row[key] for row in rows if row.get(key) is not None])
        for key in fields
    }
    measured = [
        row
        for row in rows
        if row.get("generation_seconds", 0) > 0
        and row.get("generated_tokens") is not None
    ]
    seconds = sum(row["generation_seconds"] for row in measured)
    return {
        "items": len(rows),
        "distributions": stats,
        "generation_tokens_per_second": sum(row["generated_tokens"] for row in measured)
        / seconds
        if seconds
        else None,
        "scope": "Per-item measurements across all saved rows, including resumed rows. Sums across shards are worker time, not parallel wall time. Generated tokens include special/stop tokens; words are whitespace-delimited.",
    }


class RunMonitor:
    """Write unique per-invocation files so resume never overwrites measurements."""

    def __init__(self, directory, phase, interval=2.0):
        self.directory = Path(directory)
        self.phase = phase
        self.interval = interval
        self.stop = threading.Event()
        self.errors = 0
        self.samples = []

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run_id = (
            f"{time.time_ns()}-rank{os.environ.get('RANK', '0')}-{uuid.uuid4().hex[:8]}"
        )
        self.sample_path = self.directory / f"{self.run_id}.gpu.jsonl"
        self.summary_path = self.directory / f"{self.run_id}.json"
        self.device = None
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            self.device = {
                "logical_index": device,
                "name": props.name,
                "uuid": str(getattr(props, "uuid", "")),
                "total_memory_bytes": props.total_memory,
            }
            if self.device["uuid"] and not self.device["uuid"].startswith(("GPU-", "MIG-")):
                self.device["uuid"] = "GPU-" + self.device["uuid"]
            synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.started = time.perf_counter()
        self.cpu_started = time.process_time()
        self.started_unix = time.time()
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()
        return self

    def _sample_loop(self):
        with self.sample_path.open("w") as writer:
            while not self.stop.is_set():
                if self.device and self.device["uuid"]:
                    try:
                        result = subprocess.run(
                            [
                                "nvidia-smi",
                                "--id=" + self.device["uuid"],
                                "--query-gpu=uuid,utilization.gpu,memory.used,power.draw,temperature.gpu",
                                "--format=csv,noheader,nounits",
                            ],
                            capture_output=True,
                            text=True,
                            timeout=2,
                            check=True,
                        )
                        parts = result.stdout.strip().split(",")
                        sample = {
                            "elapsed_seconds": time.perf_counter() - self.started,
                            "uuid": parts[0].strip(),
                        }
                        for key, value in zip(
                            (
                                "utilization_percent",
                                "memory_used_mib",
                                "power_watts",
                                "temperature_c",
                            ),
                            parts[1:],
                        ):
                            try:
                                number = float(value.strip())
                                sample[key] = number if math.isfinite(number) else None
                            except ValueError:
                                sample[key] = None
                        self.samples.append(sample)
                        writer.write(json.dumps(sample) + "\n")
                        writer.flush()
                    except (OSError, subprocess.SubprocessError, ValueError):
                        self.errors += 1
                if self.stop.wait(self.interval):
                    break

    def __exit__(self, exc_type, exc, tb):
        # A failed CUDA operation can prevent synchronization; preserve the original error.
        try:
            synchronize()
        except RuntimeError:
            pass
        seconds = time.perf_counter() - self.started
        self.stop.set()
        self.thread.join(timeout=3)
        summary = {
            "phase": self.phase,
            "status": "failed" if exc_type else "complete",
            "error_type": exc_type.__name__ if exc_type else None,
            "started_unix": self.started_unix,
            "wall_seconds": seconds,
            "process_cpu_seconds": time.process_time() - self.cpu_started,
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            "gpu": self.device,
            "torch_memory": gpu_memory(),
            "sampling_interval_seconds": self.interval,
            "sampling_errors": self.errors,
            "gpu_samples": len(self.samples),
            "sample_file": str(self.sample_path),
            "gpu_statistics": {
                key: distribution(
                    [s[key] for s in self.samples if s.get(key) is not None]
                )
                for key in (
                    "utilization_percent",
                    "memory_used_mib",
                    "power_watts",
                    "temperature_c",
                )
            },
            "scope": "Current process CUDA device; nvidia-smi samples are device-wide and may include other jobs. PyTorch memory is process-local; RSS is process-lifetime peak. Missing sampling is reported, not zero usage.",
        }
        atomic_json(self.summary_path, summary)
        return False
