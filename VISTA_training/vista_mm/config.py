"""Validated, serializable experiment configuration."""

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

DIMS = {
    "audio": 64,
    "imu": 16,
    "magbaro": 8,
    "gaze": 10,
    "hands": 168,
    "trajectory": 19,
    "metric3d": 48,
}
MPS = ["gaze", "hands", "trajectory", "metric3d"]


@dataclass
class SensorConfig:
    modalities: list[str] = field(default_factory=lambda: list(DIMS))
    width: int = 256
    tokens_per_modality: int = 8
    fused_tokens: int = 8
    readout_tokens_per_modality: int = 0
    heads: int = 8
    adapter_layers: int = 2
    fusion_layers: int = 2
    question_layers: int = 2
    dropout: float = 0.1
    modality_dropout: float = 0.15
    fusion: str = "bottleneck"  # bottleneck exchange or one-way cross_attention
    gate_context: str = "joint"  # joint available-sensor context or independent
    use_reliability_gates: bool = True
    use_question_gates: bool = True
    use_quality: bool = True
    gate_temperature: float = 1.0
    sensor_token_id: int | None = None
    audio_encoder: str = "features"
    imu_encoder: str = "features"
    depth_encoder: str = "statistics"
    beats_config: dict = field(default_factory=dict)
    finetune_encoders: bool = False
    encoder_microbatch: int = 4
    encoder_checkpointing: bool = True

    def __post_init__(self):
        if (
            type(self.readout_tokens_per_modality) is not int
            or not 0 <= self.readout_tokens_per_modality <= self.tokens_per_modality
        ):
            raise ValueError(
                "Readout tokens must be between zero and tokens_per_modality"
            )
        if (
            len(set(self.modalities)) != len(self.modalities)
            or set(self.modalities) - DIMS.keys()
        ):
            raise ValueError(f"Invalid modalities: {self.modalities}")
        dimensions = (
            self.width,
            self.heads,
            self.fused_tokens,
            self.tokens_per_modality,
            self.adapter_layers,
            self.fusion_layers,
            self.question_layers,
        )
        if any(type(value) is not int or value < 1 for value in dimensions):
            raise ValueError(
                "Attention dimensions and layer counts must be positive integers"
            )
        if self.width % self.heads:
            raise ValueError("Invalid attention dimensions")
        if self.fusion not in ("bottleneck", "cross_attention"):
            raise ValueError("fusion must be bottleneck or cross_attention")
        if self.gate_context not in ("joint", "independent"):
            raise ValueError("gate_context must be joint or independent")
        for value, allowed in (
            (self.audio_encoder, ("features", "beats")),
            (self.imu_encoder, ("features", "primus")),
            (self.depth_encoder, ("statistics", "spatial")),
        ):
            if value not in allowed:
                raise ValueError(f"Invalid encoder {value}; expected {allowed}")
        if type(self.encoder_microbatch) is not int or self.encoder_microbatch < 1:
            raise ValueError("encoder_microbatch must be a positive integer")
        if not 0 <= self.modality_dropout < 1 or not 0 <= self.dropout < 1:
            raise ValueError("Dropout must be in [0,1)")
        if not math.isfinite(self.gate_temperature) or self.gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive")

    def to_dict(self):
        return asdict(self)

    @property
    def input_tokens(self):
        return (
            self.fused_tokens + len(self.modalities) * self.readout_tokens_per_modality
        )


@dataclass
class Experiment:
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    revision: str = ""  # immutable model commit, required for a new run
    prepared: str = "data/prepared"
    output_dir: str = "runs/default"
    sensor: dict = field(default_factory=dict)
    seed: int = 17
    train_mode: str = "full"  # full LLM, lora, or alignment-only
    num_frames: int = 8
    min_pixels: int = 128 * 32 * 32
    max_pixels: int = 384 * 32 * 32
    max_length: int = 6144
    max_question_length: int = 256
    epochs: float = 5.0
    max_steps: int = -1
    smoke_seconds: float = 0.0
    batch_size: int = 1
    accumulation: int = 16
    backbone_lr: float = 1e-5
    sensor_lr: float = 1e-4
    encoder_lr: float = 1e-6
    encoder_assets: str | None = None
    encoder_initialization: str = "pretrained"
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    workers: int = 4
    eval_steps: int = 100
    save_steps: int = 100
    checkpoint_first_step: bool = False
    logging_steps: int = 10
    deepspeed: str | None = None
    attention: str = "sdpa"
    gradient_checkpointing: bool = True
    sensor_warmup_steps: int = 0
    internvl_detail_frames: list[int] = field(default_factory=list)
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    initial_checkpoint: str | None = None
    initial_adapter: str | None = None
    min_modality_coverage: float = 0.01
    allow_missing_modalities: list[str] = field(default_factory=list)
    gradient_modulation: str = "none"  # conditional sensor-adapter OGM adaptation
    modulation_alpha: float = 0.1

    def __post_init__(self):
        self.sensor = SensorConfig(**self.sensor).to_dict()
        if self.initial_adapter and (
            self.train_mode != "lora" or self.initial_checkpoint
        ):
            raise ValueError(
                "initial_adapter requires LoRA and cannot combine with initial_checkpoint"
            )
        if (
            type(self.sensor_warmup_steps) is not int
            or self.sensor_warmup_steps < 0
            or (
                self.sensor_warmup_steps
                and (
                    self.train_mode != "lora"
                    or not self.sensor["modalities"]
                    or self.deepspeed
                )
            )
        ):
            raise ValueError(
                "Sensor warmup requires sensor LoRA training without DeepSpeed"
            )
        if self.max_steps > 0 and self.sensor_warmup_steps >= self.max_steps:
            raise ValueError("Sensor warmup must end before the training budget")
        if (
            not isinstance(self.internvl_detail_frames, list)
            or len(self.internvl_detail_frames) > 2
            or len(set(self.internvl_detail_frames)) != len(self.internvl_detail_frames)
            or any(
                type(i) is not int or not 0 <= i < 8
                for i in self.internvl_detail_frames
            )
        ):
            raise ValueError(
                "Select at most two distinct frame indices from 0 through 7"
            )
        if self.gradient_modulation not in ("none", "ogm", "ogm_ge"):
            raise ValueError("gradient_modulation must be none, ogm, or ogm_ge")
        if not math.isfinite(self.modulation_alpha) or self.modulation_alpha <= 0:
            raise ValueError("modulation_alpha must be positive")
        if self.gradient_modulation != "none" and len(self.sensor["modalities"]) < 2:
            raise ValueError(
                "Gradient modulation requires at least two sensor branches"
            )
        if not isinstance(self.allow_missing_modalities, list) or any(
            name not in self.sensor["modalities"]
            for name in self.allow_missing_modalities
        ):
            raise ValueError(
                "allow_missing_modalities must name selected sensor modalities"
            )
        if (
            type(self.max_steps) is not int
            or self.max_steps == 0
            or self.max_steps < -1
        ):
            raise ValueError("max_steps must be -1 or a positive integer")
        if not math.isfinite(self.smoke_seconds) or self.smoke_seconds < 0:
            raise ValueError("smoke_seconds must be nonnegative")
        if self.encoder_initialization not in ("pretrained", "random"):
            raise ValueError("encoder_initialization must be pretrained or random")
        if not math.isfinite(self.encoder_lr) or self.encoder_lr <= 0:
            raise ValueError("encoder_lr must be positive")
        if self.train_mode not in ("full", "lora", "alignment"):
            raise ValueError("train_mode must be full, lora, or alignment")
        if self.train_mode == "lora" and self.sensor["finetune_encoders"]:
            raise ValueError(
                "LoRA keeps native encoders frozen; use full for encoder fine-tuning"
            )
        if self.num_frames != 8:
            raise ValueError("Feature schema v1 is aligned to exactly eight RGB frames")
        if self.save_steps != self.eval_steps:
            raise ValueError(
                "Save/eval intervals must match for best-checkpoint selection"
            )
        if self.batch_size < 1 or self.accumulation < 1:
            raise ValueError("Invalid batch size")

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        text = path.read_text()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = yaml.safe_load(text)
        # PyYAML's YAML 1.1 resolver treats JSON-style `1e-06` as a string.
        # Normalize declared floating-point fields also for ordinary YAML configs.
        for field_info in fields(cls):
            if field_info.type is float and field_info.name in data:
                data[field_info.name] = float(data[field_info.name])
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys: {unknown}")
        if data.get("deepspeed"):
            data["deepspeed"] = str((path.parent / data["deepspeed"]).resolve())
        return cls(**data)

    def to_dict(self):
        return asdict(self)
