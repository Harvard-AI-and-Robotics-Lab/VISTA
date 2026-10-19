"""Generate complete test predictions with configuration-checked resumption."""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import set_seed

from .config import Experiment, SensorConfig
from .data import VistaCollator, VistaDataset
from .encoder_assets import resolve_encoder_configuration
from .provenance import atomic_json, digest, read_jsonl, sha256
from .train import load_backbone
from .metrics import RunMonitor, synchronize, prediction_fields, prediction_summary
from .manifest import PROMPTS_PATH


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    return value


def shard_indices(count, index, shards):
    if shards < 1 or not 0 <= index < shards or shards > count:
        raise ValueError(
            "Use 1 <= num-shards <= items and 0 <= shard-index < num-shards"
        )
    return list(range(index, count, shards))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True, help="Saved adapter directory")
    p.add_argument("--split", choices=["test"], default="test")
    p.add_argument("--output", required=True)
    p.add_argument("--ablate", nargs="*", default=[])
    p.add_argument("--shuffle-sensors", action="store_true")
    p.add_argument("--shuffle-preserve-availability", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-new-tokens", type=int, choices=[4096], default=4096)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    args = p.parse_args()
    config = Experiment.load(args.config)
    resolve_encoder_configuration(config, args.checkpoint)
    set_seed(config.seed)
    torch.backends.cudnn.benchmark = False
    dataset = VistaDataset(
        config.prepared,
        args.split,
        config,
        ablate=args.ablate,
        shuffle_sensors=args.shuffle_sensors,
        shuffle_preserve_availability=args.shuffle_preserve_availability,
    )
    indices = shard_indices(len(dataset), args.shard_index, args.num_shards)
    checkpoint = Path(args.checkpoint).resolve()
    checkpoint_hashes = {}
    for path in sorted(checkpoint.iterdir()):
        if path.suffix in (".json", ".safetensors"):
            checkpoint_hashes[path.name] = sha256(path)
    if not any(k.endswith(".safetensors") for k in checkpoint_hashes):
        raise ValueError("Checkpoint has no safetensors weights")
    if not (checkpoint.parent / "experiment.json").exists():
        raise ValueError("Checkpoint must belong to a recorded VISTA run")
    trained = json.loads((checkpoint.parent / "experiment.json").read_text())
    # Normalize optional sensor fields before comparing checkpoint metadata.
    trained["sensor"] = SensorConfig(**trained["sensor"]).to_dict()
    for key in (
        "model_id",
        "revision",
        "sensor",
        "train_mode",
        "num_frames",
        "min_pixels",
        "max_pixels",
    ):
        if trained[key] != config.to_dict()[key]:
            raise ValueError(f"Inference config differs from training: {key}")
    run_manifest = json.loads(
        (checkpoint.parent / "prepared_manifest.json").read_text()
    )
    if run_manifest["prepared_id"] != dataset.manifest["prepared_id"]:
        raise ValueError(
            "Inference preparation/normalization differs from training"
        )
    protocol = {
        "model_id": config.model_id,
        "revision": config.revision,
        "seed": config.seed,
        "checkpoint_hashes": checkpoint_hashes,
        "prepared_id": dataset.manifest["prepared_id"],
        "split": args.split,
        "ablate": sorted(args.ablate),
        "shuffle_sensors": args.shuffle_sensors,
        "shuffle_preserve_availability": args.shuffle_preserve_availability,
        "sensor_donors": dataset.sensor_donor if args.shuffle_sensors else None,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "repetition_penalty": 1.0,
        "prompts_sha256": sha256(PROMPTS_PATH),
        "config": config.to_dict(),
        "implementation_sha256": {
            str(p.relative_to(Path(__file__).parent)): sha256(p)
            for p in Path(__file__).parent.rglob("*.py")
        },
    }
    if args.num_shards > 1:
        protocol["shard"] = {"index": args.shard_index, "count": args.num_shards}
    protocol["identity"] = digest(protocol)
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    sidecar = out.with_suffix(".protocol.json")
    completion = out.with_suffix(".complete.json")
    done = {}
    if out.exists() or sidecar.exists():
        if not args.resume or not out.exists() or not sidecar.exists():
            raise FileExistsError(
                "Prediction output exists; use a new path or --resume"
            )
        if json.loads(sidecar.read_text())["identity"] != protocol["identity"]:
            raise ValueError("Resume identity mismatch")
        # An interrupted JSON line fails explicitly; never silently discard its bytes.
        for row in read_jsonl(out):
            if row["item_id"] in done:
                raise ValueError("Duplicate prediction item")
            done[row["item_id"]] = row
    else:
        atomic_json(sidecar, protocol)
    expected = {dataset.items[i]["item_id"] for i in indices}
    if set(done) - expected:
        raise ValueError("Predictions contain items outside the selected split")
    with RunMonitor(out.with_suffix(".resources"), "inference_including_model_load"):
        loading_started = time.perf_counter()
        model, processor = load_backbone(config, checkpoint=checkpoint, training=False)
        processor.tokenizer.padding_side = "left"
        model = model.to("cuda").eval()
        synchronize()
        model_load_seconds = time.perf_counter() - loading_started
        collate = VistaCollator(processor, config, training=False)
        elapsed = time.monotonic()
        with out.open("a") as writer, torch.inference_mode():
            for i in indices:
                item = dataset.items[i]
                if item["item_id"] in done:
                    continue
                item_started = time.perf_counter()
                batch = move(collate([dataset[i]]), "cuda")
                synchronize()
                preprocessing_seconds = time.perf_counter() - item_started
                generation_started = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ids = model.generate(
                        **batch,
                        do_sample=False,
                        num_beams=1,
                        repetition_penalty=1.0,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                        pad_token_id=processor.tokenizer.pad_token_id,
                    )
                synchronize()
                generation_seconds = time.perf_counter() - generation_started
                generated = ids[0, batch["input_ids"].shape[1] :]
                text = processor.tokenizer.decode(
                    generated, skip_special_tokens=True
                ).strip()
                if not text:
                    raise RuntimeError(f"Empty prediction for {item['item_id']}")
                row = {k: item[k] for k in ("item_id", "clip_id", "task_type")}
                row.update(prediction_fields(
                    text, generated, batch["input_ids"].shape[1], args.max_new_tokens,
                    model.generation_config.eos_token_id, generation_seconds,
                    preprocessing_seconds, time.perf_counter() - item_started,
                ))
                if args.shuffle_sensors:
                    row["sensor_donor_clip_id"] = dataset.sensor_donor[item["clip_id"]]
                if config.sensor.get("modalities"):
                    core = (
                        model.get_base_model()
                        if hasattr(model, "get_base_model")
                        else model
                    )
                    weights = core.sensor_encoder.last_gate_weights
                    if weights is not None:
                        row["sensor_gate_weights"] = dict(
                            zip(
                                config.sensor["modalities"],
                                weights[0].float().cpu().tolist(),
                            )
                        )
                    mixture = core.sensor_encoder.last_mixture_weights
                    if mixture is not None:
                        row["sensor_mixture_weights"] = dict(
                            zip(
                                config.sensor["modalities"] + ["fallback"],
                                mixture[0].float().cpu().tolist(),
                            )
                        )
                    availability = core.sensor_encoder.last_modality_available
                    if availability is not None:
                        row["sensor_available"] = dict(
                            zip(config.sensor["modalities"], availability[0].cpu().tolist())
                        )
                writer.write(json.dumps(row) + "\n")
                writer.flush()
                done[item["item_id"]] = row
        if set(done) != expected:
            raise RuntimeError(f"Incomplete predictions: {len(done)}/{len(expected)}")
        atomic_json(
            completion,
            {
                "identity": protocol["identity"],
                "expected": len(expected),
                "written": len(done),
                "prediction_sha256": sha256(out),
                "elapsed_seconds_this_invocation": time.monotonic() - elapsed,
                "truncated_items": sum(r["hit_token_limit"] for r in done.values()),
                "model_load_seconds_this_invocation": model_load_seconds,
                "performance": prediction_summary(done.values()),
            },
        )


if __name__ == "__main__":
    main()
