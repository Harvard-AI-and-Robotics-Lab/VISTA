"""Convert pretrained sensor checkpoints and initialize verified encoder assets.

Checkpoint conversion uses tensor-only loading and writes validated weights for
training and inference."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .config import SensorConfig
from .encoders import build_beats
from .primus import PrimusEncoder
from .provenance import atomic_json, sha256

ASSET_SCHEMA = "vista-pretrained-encoders-v1"


def checked_checkpoint(path, expected):
    if len(expected) != 64 or sha256(path) != expected:
        raise ValueError(f"Checkpoint SHA256 mismatch: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def read_assets(path, modalities):
    if not path:
        raise ValueError(
            "Native encoders need encoder_assets; see PRETRAINED_ENCODERS.md"
        )
    root = Path(path)
    if (root / "INCOMPLETE.json").exists():
        raise ValueError("Encoder asset conversion is incomplete")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != ASSET_SCHEMA:
        raise ValueError("Encoder asset schema mismatch")
    for name in modalities:
        entry = manifest["encoders"][name]
        filename = entry["file"]
        if (
            Path(filename).name != filename
            or sha256(root / filename) != entry["sha256"]
        ):
            raise ValueError(f"Invalid or changed {name} weights")
    return manifest


def native_modalities(config):
    return [
        name
        for name, enabled in (
            ("audio", config.sensor["audio_encoder"] == "beats"),
            ("imu", config.sensor["imu_encoder"] == "primus"),
        )
        if enabled and name in config.sensor["modalities"]
    ]


def resolve_encoder_configuration(config, checkpoint=None):
    names = native_modalities(config)
    if not names:
        return {}
    if checkpoint:
        saved = json.loads((Path(checkpoint) / "config.json").read_text())[
            "vista_sensor"
        ]
        if "audio" in names:
            if (
                config.sensor["beats_config"]
                and config.sensor["beats_config"] != saved["beats_config"]
            ):
                raise ValueError(
                    "BEATs configuration differs from the saved checkpoint"
                )
            config.sensor["beats_config"] = saved["beats_config"]
        return {}
    assets = read_assets(config.encoder_assets, names)
    if "audio" in names:
        resolved = assets["encoders"]["audio"]["config"]
        if config.sensor["beats_config"] and config.sensor["beats_config"] != resolved:
            raise ValueError("BEATs configuration differs from the asset manifest")
        config.sensor["beats_config"] = resolved
    return {
        "manifest_sha256": sha256(Path(config.encoder_assets) / "manifest.json"),
        **assets,
    }


def initialize_frontends(model, config):
    names = native_modalities(config)
    if not names:
        return
    assets = read_assets(config.encoder_assets, names)
    cfg = SensorConfig(**config.sensor)
    for name in names:
        frontend = model.sensor_encoder.frontends[name]
        # Build outside Transformers' no-init scope. This also gives the random
        # control the upstream initialization (including GRU/weight_norm parameters).
        network = build_beats(cfg.beats_config) if name == "audio" else PrimusEncoder()
        if config.encoder_initialization == "pretrained":
            state = load_file(
                str(Path(config.encoder_assets) / assets["encoders"][name]["file"])
            )
            network.load_state_dict(state, strict=True)
        device = next(model.sensor_encoder.adapters.parameters()).device
        frontend.network = network.to(device=device, dtype=torch.float32)
        frontend.set_trainable(cfg.finetune_encoders)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--beats", required=True, help="Official non-classifier BEATs .pt"
    )
    parser.add_argument("--beats-sha256", required=True)
    parser.add_argument(
        "--primus", required=True, help="Official PRIMUS multihead .ckpt"
    )
    parser.add_argument("--primus-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "INCOMPLETE.json", vars(args))
    beats = checked_checkpoint(args.beats, args.beats_sha256)
    if not isinstance(beats["cfg"], dict):
        raise ValueError("BEATs checkpoint cfg must be a plain dictionary")
    # Merge official defaults so the resulting model configuration is self-contained.
    from .third_party.beats.BEATs import BEATsConfig

    cfg = vars(BEATsConfig(beats["cfg"]))
    json.dumps(cfg, allow_nan=False)
    source_cfg = dict(cfg)
    # Released checkpoints can carry pretraining LayerDrop. Disable layer skipping
    # for this recipe's fixed DDP parameter participation; frozen eval is unchanged.
    cfg["encoder_layerdrop"] = 0.0
    audio = build_beats(cfg)
    # Upstream ties relative position embeddings. Refuse conflicting copies.
    biases = [
        v
        for k, v in beats["model"].items()
        if k.endswith("relative_attention_bias.weight")
    ]
    if biases and any(not torch.equal(biases[0], value) for value in biases[1:]):
        raise ValueError("Conflicting tied BEATs position embeddings")
    audio.load_state_dict(beats["model"], strict=True)
    primus = checked_checkpoint(args.primus, args.primus_sha256)
    state = {
        k.removeprefix("imu_encoder."): v
        for k, v in primus["state_dict"].items()
        if k.startswith("imu_encoder.")
    }
    for key in ("ssl_head.weight", "ssl_head.bias"):
        if key not in state:
            raise ValueError(
                "Expected PRIMUS multihead checkpoint with imu_encoder prefix"
            )
        expected_shape = (512, 512) if key.endswith("weight") else (512,)
        if tuple(state.pop(key).shape) != expected_shape:
            raise ValueError("PRIMUS released 512-dimensional architecture required")
    imu = PrimusEncoder()
    imu.load_state_dict(state, strict=True)
    entries = {}
    for name, network, source_hash in (
        ("audio", audio, args.beats_sha256),
        ("imu", imu, args.primus_sha256),
    ):
        filename = f"{name}.safetensors"
        # Clone aliases for safetensors; architectural tying is restored on construction.
        save_file(
            {
                k: v.detach().cpu().contiguous().clone()
                for k, v in network.state_dict().items()
            },
            str(root / filename),
        )
        entries[name] = {
            "file": filename,
            "sha256": sha256(root / filename),
            "source_sha256": source_hash,
        }
    entries["audio"]["config"] = cfg
    entries["audio"]["source_config"] = source_cfg
    entries["audio"]["training_overrides"] = {
        "encoder_layerdrop": {
            "source": source_cfg["encoder_layerdrop"],
            "value": 0.0,
            "reason": "Fixed parameter participation; no change to frozen eval inference",
        }
    }
    entries["imu"]["config"] = {
        "channels": 6,
        "samples": 250,
        "sample_rate": 50,
        "embedding": 512,
        "head": "mmcl",
    }
    atomic_json(
        root / "manifest.json",
        {
            "schema": ASSET_SCHEMA,
            "encoders": entries,
            "references_sha256": sha256(
                Path(__file__).resolve().parents[1] / "ENCODER_REFERENCES.json"
            ),
        },
    )
    (root / "INCOMPLETE.json").unlink()


if __name__ == "__main__":
    main()
