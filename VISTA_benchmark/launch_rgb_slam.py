#!/usr/bin/env python3
"""Run the existing eight VISTA models when whole GPUs become available.

Workers survive a queue restart. Per-output and per-model file locks prevent
duplicate inference; worker state is saved independently from queue state.
"""

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone


REPO = Path(__file__).resolve().parent
MODELS = (
    "qwen3vl_2b", "qwen3vl_4b", "qwen3vl_8b",
    "internvl3_1b", "internvl3_2b", "internvl3_8b", "internvl3_9b",
    "llama3_video",
)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def process_start(pid):
    try:
        # /proc/<pid>/stat's comm field can contain spaces and parentheses.
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, ValueError, IndexError):
        return None


def alive(record, key="worker"):
    pid = record.get(f"{key}_pid")
    start = record.get(f"{key}_start")
    return bool(pid and start and process_start(pid) == start)


def expected_tasks(dataset_root, csv_path, limit=None):
    """Mirror data.py's accepted clips and sequential task numbering."""
    expected = set()
    accepted = 0

    def present(value):
        return value is not None and value.strip() not in {"", "nan", "NaN", "NA", "N/A", "null", "NULL"}

    with Path(csv_path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            clip = row["clip_id"]
            video = Path(dataset_root) / clip.split("_")[0] / clip / f"{clip}.mp4"
            if not video.is_file():
                continue
            if limit is not None and accepted >= limit:
                break
            accepted += 1
            expected.add((clip, "scene_description"))
            for kind, source, target, count in (
                ("question", "question", "answer", 8),
                ("goal", "goal", "instruction", 17),
            ):
                number = 0
                for index in range(1, count + 1):
                    if present(row.get(f"{source}_{index}")) and present(row.get(f"{target}_{index}")):
                        number += 1
                        expected.add((clip, f"{kind}_{number}"))
    return expected


def validate_predictions(path, expected):
    seen = set()
    empty = duplicates = malformed = wrong_condition = 0
    try:
        with Path(path).open() as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    key = (row["clip_id"], row["task_type"])
                    prediction = row["prediction"]
                    if row.get("input_mode") != "rgb8_slam_left4_right4":
                        wrong_condition += 1
                    if not isinstance(prediction, str) or not prediction.strip():
                        empty += 1
                    if key in seen:
                        duplicates += 1
                    seen.add(key)
                except (ValueError, KeyError, TypeError):
                    malformed += 1
    except FileNotFoundError:
        pass
    result = {
        "expected_rows": len(expected), "unique_rows": len(seen),
        "missing_rows": len(expected - seen), "unexpected_rows": len(seen - expected),
        "empty_rows": empty, "duplicate_rows": duplicates, "malformed_rows": malformed,
        "wrong_condition_rows": wrong_condition,
    }
    result["valid"] = bool(expected) and seen == expected and not (empty or duplicates or malformed or wrong_condition)
    return result


def worker(spec_path):
    spec = read_json(spec_path)
    state_path = Path(spec["state_path"])
    with Path(spec["lock_path"]).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        previous = read_json(state_path, {})
        if previous.get("status") in {"completed", "failed"} or alive(previous, "runner"):
            return 0
        state = {
            "model": spec["model"], "status": "running", "gpu": spec["gpu"],
            "started_at": timestamp(), "worker_pid": os.getpid(),
            "worker_start": process_start(os.getpid()), "log": spec["log_path"],
        }
        write_json(state_path, state)
        try:
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": str(spec["gpu"]), "PYTHONNOUSERSITE": "1",
                "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "PYTHONUNBUFFERED": "1",
            })
            with Path(spec["log_path"]).open("a", buffering=1) as log:
                log.write(f"[{timestamp()}] Starting {spec['model']} on GPU {spec['gpu']}\n")
                child = subprocess.Popen(spec["command"], cwd=REPO, env=env,
                                         stdout=log, stderr=subprocess.STDOUT,
                                         pass_fds=(lock.fileno(),))
                state.update(runner_pid=child.pid, runner_start=process_start(child.pid))
                write_json(state_path, state)
                code = child.wait()
            expected = expected_tasks(spec["dataset_root"], spec["csv_path"], spec["limit"])
            validation = validate_predictions(spec["prediction_path"], expected)
            state.update(status="completed" if code == 0 and validation["valid"] else "failed",
                         exit_code=code, validation=validation, ended_at=timestamp())
        except Exception as exc:
            state.update(status="failed", error=f"{type(exc).__name__}: {exc}", ended_at=timestamp())
        write_json(state_path, state)
        return 0 if state["status"] == "completed" else 1


def gpu_snapshot():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "nvidia-smi failed")
    rows = []
    for line in result.stdout.splitlines():
        index, free, utilization = [field.strip() for field in line.split(",")]
        rows.append({"index": int(index), "free_mib": int(free), "utilization": int(utilization)})
    return rows


def queue(args):
    output = Path(args.output_dir).resolve()
    if args.status:
        state = read_json(output / "queue_status.json")
        print(json.dumps(state, indent=2) if state else f"No queue status at {output}")
        return 0
    output.mkdir(parents=True, exist_ok=True)
    for name in ("logs", "workers", "locks"):
        (output / name).mkdir(exist_ok=True)
    with (output / "queue.lock").open("a") as queue_lock:
        try:
            fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"A queue is already running for {output}", flush=True)
            return 2
        return run_queue(args, output)


def run_queue(args, output):
    config = {
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "csv_path": str(Path(args.csv_path).resolve()),
        "output_dir": str(output), "models": args.models, "limit": args.limit,
    }
    status_path = output / "queue_status.json"
    previous = read_json(status_path, {})
    if previous and previous.get("config") != config:
        raise ValueError("This output directory already has a queue with different settings. Use a new output directory.")
    expected = expected_tasks(config["dataset_root"], config["csv_path"], args.limit)
    if not expected:
        raise ValueError("Dataset selection contains no tasks")
    state = {
        "config": config, "queue_pid": os.getpid(), "queue_start": process_start(os.getpid()),
        "started_at": previous.get("started_at", timestamp()), "status": "running",
        "expected_rows_per_model": len(expected), "models": previous.get("models", {}),
    }
    children = {}
    last_summary = None
    last_report = 0
    while True:
        preflight = read_json(output / "preflight.json")
        if not preflight:
            state.update(status="waiting_for_preflight", updated_at=timestamp())
            state["models"] = {model: state["models"].get(model, {"model": model, "status": "pending"}) for model in args.models}
            write_json(status_path, state)
            if time.monotonic() - last_report >= 60:
                print(f"[{timestamp()}] Waiting for successful input preflight", flush=True)
                last_report = time.monotonic()
            time.sleep(args.poll_seconds)
            continue
        if preflight.get("errors") or preflight.get("num_tasks") != len(expected) or preflight.get("input_mode") != "rgb8_slam_left4_right4":
            state.update(status="preflight_failed", updated_at=timestamp(), preflight=preflight)
            write_json(status_path, state)
            raise ValueError("Input preflight failed or does not match this dataset selection")
        state["status"] = "running"
        for model in args.models:
            record_path = output / "workers" / f"{model}.json"
            record = read_json(record_path, state["models"].get(model, {"model": model, "status": "pending"}))
            if record["status"] in {"running", "starting"} and not alive(record) and not alive(record, "runner"):
                validation = validate_predictions(output / f"predictions_{model}.jsonl", expected)
                record.update(status="completed" if validation["valid"] else "failed",
                              ended_at=timestamp(), validation=validation,
                              recovery_note="Worker no longer exists; completion recovered from predictions.")
                write_json(record_path, record)
            state["models"][model] = record
        for model, child in list(children.items()):
            if child.poll() is not None:
                del children[model]
        active = [record for record in state["models"].values() if record["status"] in {"running", "starting"}]
        pending = [model for model in args.models if state["models"][model]["status"] == "pending"]
        state["gpu_error"] = None
        try:
            gpus = gpu_snapshot() if pending else []
        except Exception as exc:
            gpus = []
            state["gpu_error"] = str(exc)
        state["gpus"] = gpus
        reserved = {record["gpu"] for record in active}
        available = [gpu for gpu in gpus if gpu["index"] not in reserved
                     and gpu["free_mib"] >= args.min_free_mib and gpu["utilization"] < 10]
        slots = max(0, args.max_parallel - len(active))
        for model, gpu in zip(pending[:slots], available):
            family = "venv_qwen" if model.startswith("qwen") else "venv_internvl3" if model.startswith("internvl") else "venv_videollama"
            if model == "internvl3_9b":
                family = "venv_internvl3_9b"
            executable = Path(os.environ.get("VISTA_VENVS_ROOT", str(REPO / "venvs"))) / family / "bin" / "python"
            command = [str(executable), "-u", str(REPO / "run_rgb_slam.py"), "--model", model,
                       "--dataset_root", config["dataset_root"], "--csv_path", config["csv_path"],
                       "--output_dir", str(output)]
            if args.limit is not None:
                command.extend(["--limit", str(args.limit)])
            spec = dict(config, model=model, gpu=gpu["index"], command=command,
                        state_path=str(output / "workers" / f"{model}.json"),
                        lock_path=str(output / "locks" / f"{model}.lock"),
                        log_path=str(output / "logs" / f"{model}.log"),
                        prediction_path=str(output / f"predictions_{model}.jsonl"))
            spec_path = output / "workers" / f"{model}.spec.json"
            write_json(spec_path, spec)
            with (output / "logs" / f"{model}.worker.log").open("a") as worker_log:
                child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker-spec", str(spec_path)],
                                         cwd=REPO, stdout=worker_log, stderr=subprocess.STDOUT, start_new_session=True)
            children[model] = child
            state["models"][model] = dict(model=model, status="starting", gpu=gpu["index"],
                                          worker_pid=child.pid, worker_start=process_start(child.pid), started_at=timestamp())
            state["updated_at"] = timestamp()
            write_json(status_path, state)
        terminal = all(record["status"] in {"completed", "failed"} for record in state["models"].values())
        if terminal:
            state["status"] = "completed" if all(record["status"] == "completed" for record in state["models"].values()) else "finished_with_failures"
            state["ended_at"] = timestamp()
        state["updated_at"] = timestamp()
        write_json(status_path, state)
        summary = ", ".join(f"{model}={state['models'][model]['status']}" for model in args.models)
        if summary != last_summary or time.monotonic() - last_report >= 60:
            print(f"[{timestamp()}] {summary}", flush=True)
            if state["gpu_error"]:
                print(f"GPU probe: {state['gpu_error']}", flush=True)
            last_summary, last_report = summary, time.monotonic()
        if terminal:
            return 0 if state["status"] == "completed" else 1
        time.sleep(args.poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", default=os.environ.get("VISTA_TEST_ROOT", str(Path.home() / "VISTA_data/test")))
    parser.add_argument("--csv_path", default=None)
    parser.add_argument("--output_dir", default=str(REPO / "outputs" / "rgb8_slam8"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max_parallel", type=int, default=2)
    parser.add_argument("--min_free_mib", type=int, default=44000)
    parser.add_argument("--poll_seconds", type=float, default=30)
    parser.add_argument("--status", action="store_true", help="Read saved queue status without starting work")
    parser.add_argument("--worker-spec", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.csv_path is None:
        args.csv_path = str(Path(args.dataset_root) / "test.csv")
    if args.worker_spec:
        return worker(args.worker_spec)
    if args.max_parallel < 1 or args.poll_seconds <= 0 or args.min_free_mib < 1:
        parser.error("max_parallel, min_free_mib and poll_seconds must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if len(set(args.models)) != len(args.models):
        parser.error("models must not contain duplicates")
    return queue(args)


if __name__ == "__main__":
    sys.exit(main())
