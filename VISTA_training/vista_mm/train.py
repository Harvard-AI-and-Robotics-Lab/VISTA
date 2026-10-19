"""Full VISTA training for InternVL3, Qwen3-VL and Gemma 3."""

import argparse
from pathlib import Path
import torch
from transformers import (
    AutoConfig,
    AutoProcessor,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from .config import Experiment, SensorConfig
from .data import (
    VistaCollator,
    VistaDataset,
    verify_prepared,
    SynchronizedQwenProcessor,
)
from .encoder_assets import initialize_frontends, resolve_encoder_configuration
from .encoders import NativeFrontend
from .metrics import RunMonitor
from .internvl_model import VistaInternVL
from .model import SENSOR_TOKEN, VistaQwen3VL
from .gemma_model import VistaGemma3
from .loss import response_statistics
from .provenance import atomic_json, environment_record, sha256


class SensorWarmupCallback(TrainerCallback):
    """Align sensor adapters before allowing updates to the language LoRA weights."""

    def __init__(self, steps):
        self.steps = steps

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if args.world_size != 1:
            raise ValueError("Sensor warmup currently supports single-GPU jobs only")
        enabled = state.global_step >= self.steps
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.requires_grad_(enabled)
        return control


def validate_adapter_initialization(config):
    """A continuation may change readout length, but must retain compatible weights."""
    path = Path(config.initial_adapter)
    saved = AutoConfig.from_pretrained(path)
    experiment = saved.vista_experiment
    for name in ("model_id", "revision", "lora_rank", "lora_alpha", "lora_dropout"):
        if experiment[name] != getattr(config, name):
            raise ValueError(f"Initial adapter differs in {name}")
    wanted = SensorConfig(**config.sensor).to_dict()
    previous = SensorConfig(**getattr(saved, "vista_sensor", config.sensor)).to_dict()
    for name in wanted:
        if (
            name not in ("sensor_token_id", "readout_tokens_per_modality")
            and wanted[name] != previous[name]
        ):
            raise ValueError(f"Initial adapter sensor architecture differs in {name}")


def load_backbone(config, checkpoint=None, training=True):
    resolve_encoder_configuration(config, checkpoint)
    if checkpoint and (Path(checkpoint) / "adapter_config.json").exists():
        from .lora import VistaPeftModel, prepare_native_copy

        if config.train_mode != "lora":
            raise ValueError("An adapter checkpoint requires train_mode=lora")
        model, _ = load_backbone(
            config, checkpoint=config.initial_checkpoint, training=training
        )
        saved = AutoConfig.from_pretrained(checkpoint)
        saved_sensor = getattr(saved, "vista_sensor", None)
        if saved_sensor is not None:
            saved_sensor = SensorConfig(**saved_sensor).to_dict()
        if saved_sensor != getattr(model.config, "vista_sensor", None):
            raise ValueError("Adapter sensor architecture differs from its base model")
        processor_cls = (
            SynchronizedQwenProcessor
            if model.config.model_type == "qwen3_vl"
            else AutoProcessor
        )
        processor = processor_cls.from_pretrained(checkpoint)
        processor.tokenizer.padding_side = "right" if training else "left"
        prepare_native_copy(model)
        model = VistaPeftModel.from_pretrained(model, checkpoint, is_trainable=training)
        return (model, processor)
    sensor_cfg = SensorConfig(**config.sensor)
    use_sensors = bool(sensor_cfg.modalities)
    source = checkpoint or config.model_id
    revision = None if checkpoint else config.revision
    model_config = AutoConfig.from_pretrained(source, revision=revision)
    if model_config.model_type not in ("internvl", "qwen3_vl", "gemma3"):
        raise ValueError(f"Unsupported backbone: {model_config.model_type}")
    processor_cls = (
        SynchronizedQwenProcessor
        if model_config.model_type == "qwen3_vl"
        else AutoProcessor
    )
    processor = processor_cls.from_pretrained(source, revision=revision)
    processor.tokenizer.padding_side = "right" if training else "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    if use_sensors:
        processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": [SENSOR_TOKEN]}
        )
        sensor_cfg.sensor_token_id = processor.tokenizer.convert_tokens_to_ids(
            SENSOR_TOKEN
        )
        if (
            checkpoint
            and SensorConfig(**getattr(model_config, "vista_sensor", {})).to_dict()
            != sensor_cfg.to_dict()
        ):
            raise ValueError(
                "Initial checkpoint sensor architecture differs from this experiment"
            )
        model_config.vista_sensor = sensor_cfg.to_dict()
    model_config.vista_experiment = config.to_dict()
    model_config.vista_original_vocab_size = model_config.text_config.vocab_size
    if not use_sensors:
        raise ValueError("This suite runs full VISTA only")
    cls = {"internvl": VistaInternVL, "qwen3_vl": VistaQwen3VL, "gemma3": VistaGemma3}[
        model_config.model_type
    ]
    model = cls.from_pretrained(
        source,
        revision=revision,
        config=model_config,
        torch_dtype=torch.bfloat16,
        attn_implementation=config.attention,
    )
    if use_sensors:
        if checkpoint is None:
            initialize_frontends(model, config)
        model.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False)
        old = model.generation_config.suppress_tokens or []
        model.generation_config.suppress_tokens = sorted(
            set(old + [sensor_cfg.sensor_token_id])
        )
    if training and checkpoint is None and config.initial_adapter:
        from .lora import VistaPeftModel

        validate_adapter_initialization(config)
        model = VistaPeftModel.from_pretrained(
            model, config.initial_adapter, is_trainable=True
        )
    return (model, processor)


def configure_trainable(model, config):
    already_lora = hasattr(model, "peft_config")
    if already_lora:
        if config.train_mode != "lora":
            raise ValueError("Loaded adapters require LoRA training")
    elif config.train_mode == "full":
        model.requires_grad_(True)
        vision = (
            model.visual
            if model.config.model_type == "qwen3_vl"
            else model.vision_tower
        )
        vision.requires_grad_(False)
    else:
        model.requires_grad_(False)
        if config.sensor.get("modalities"):
            model.sensor_encoder.requires_grad_(True)
    if config.train_mode == "lora" and (not already_lora):
        from peft import LoraConfig
        from .lora import VistaPeftModel, prepare_native_copy

        target = ".*language_model\\.layers\\.\\d+\\.(self_attn\\.(q_proj|k_proj|v_proj|o_proj)|mlp\\.(gate_proj|up_proj|down_proj))$"
        peft_cfg = LoraConfig(
            base_model_name_or_path=config.initial_checkpoint or config.model_id,
            revision=None if config.initial_checkpoint else config.revision,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target,
            modules_to_save=["sensor_encoder"]
            if config.sensor.get("modalities")
            else None,
            bias="none",
            task_type="CAUSAL_LM",
        )
        prepare_native_copy(model)
        model = VistaPeftModel(model, peft_cfg)
    if config.gradient_checkpointing:
        model.enable_input_require_grads()
    for module in model.modules():
        if isinstance(module, NativeFrontend):
            module.set_trainable(config.sensor["finetune_encoders"])
    if not any((p.requires_grad for p in model.parameters())):
        raise ValueError(
            "No trainable parameters; RGB-only alignment mode is not a training experiment"
        )
    return model


class VistaTrainer(Trainer):
    """Average response loss per example so long references do not dominate training."""

    def __init__(self, *args, sensor_lr, encoder_lr=1e-06, **kwargs):
        super().__init__(*args, **kwargs)
        self.sensor_lr = sensor_lr
        self.encoder_lr = encoder_lr
        self.model_accepts_loss_kwargs = False

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        decay_names = set(self.get_decay_parameter_names(self.model))
        groups = []

        def category(name):
            if "sensor_encoder" in name and ".frontends." in name:
                return "encoder"
            return "sensor" if "sensor_encoder" in name else "backbone"

        rates = {
            "encoder": self.encoder_lr,
            "sensor": self.sensor_lr,
            "backbone": self.args.learning_rate,
        }
        for group, rate in rates.items():
            for decay in (False, True):
                params = [
                    p
                    for n, p in self.model.named_parameters()
                    if p.requires_grad
                    and category(n) == group
                    and ((n in decay_names) == decay)
                ]
                if params:
                    groups.append(
                        {
                            "params": params,
                            "lr": rate,
                            "weight_decay": self.args.weight_decay if decay else 0.0,
                        }
                    )
        optimizer_cls = torch.optim.AdamW
        if self.is_deepspeed_enabled:
            ds_config = self.accelerator.state.deepspeed_plugin.deepspeed_config
            if (
                ds_config.get("zero_optimization", {})
                .get("offload_optimizer", {})
                .get("device")
                == "cpu"
            ):
                from deepspeed.ops.adam import DeepSpeedCPUAdam

                optimizer_cls = DeepSpeedCPUAdam
        self.optimizer = optimizer_cls(groups, betas=(0.9, 0.95), eps=1e-08)
        return self.optimizer

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        outputs = model(**inputs, use_cache=False)
        loss = response_statistics(outputs.logits, labels)[0].mean()
        outputs.loss = loss
        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = Experiment.load(args.config)
    if (
        cfg.gradient_modulation != "none"
        or cfg.smoke_seconds
        or cfg.train_mode != "lora"
    ):
        raise ValueError("Training requires the VISTA LoRA configuration")
    set_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    manifest, _ = verify_prepared(
        cfg.prepared, cfg.sensor["modalities"], cfg.min_modality_coverage
    )
    assets = resolve_encoder_configuration(cfg, cfg.initial_adapter)
    root = Path(cfg.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError(f"Training directory is nonempty: {root}")
    atomic_json(root / "experiment.json", cfg.to_dict())
    atomic_json(root / "prepared_manifest.json", manifest)
    source = {
        str(p.relative_to(Path(__file__).parent)): sha256(p)
        for p in Path(__file__).parent.rglob("*.py")
    }
    atomic_json(
        root / "run.json",
        {
            "source_sha256": source,
            "encoder_assets": assets,
            "environment": environment_record(),
            "config_sha256": sha256(args.config),
            "fixed_final_step": cfg.max_steps,
        },
    )
    training_args = TrainingArguments(
        output_dir=str(root),
        seed=cfg.seed,
        data_seed=cfg.seed,
        max_steps=cfg.max_steps,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.accumulation,
        learning_rate=cfg.backbone_lr,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.warmup_ratio,
        max_grad_norm=cfg.max_grad_norm,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="no",
        save_strategy="no",
        logging_steps=1,
        logging_nan_inf_filter=False,
        prediction_loss_only=True,
        remove_unused_columns=False,
        dataloader_num_workers=cfg.workers,
        dataloader_pin_memory=True,
        report_to=["tensorboard"],
    )
    model, processor = load_backbone(cfg)
    model = configure_trainable(model, cfg)
    atomic_json(
        root / "parameters.json",
        {
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total": sum(p.numel() for p in model.parameters()),
            "gpu": torch.cuda.get_device_name(),
            "effective_batch": cfg.batch_size * cfg.accumulation,
        },
    )
    trainer = VistaTrainer(
        model=model,
        args=training_args,
        train_dataset=VistaDataset(cfg.prepared, "train", cfg, training=True),
        data_collator=VistaCollator(processor, cfg, training=True),
        processing_class=processor,
        sensor_lr=cfg.sensor_lr,
        encoder_lr=cfg.encoder_lr,
        callbacks=[SensorWarmupCallback(cfg.sensor_warmup_steps)]
        if cfg.sensor_warmup_steps
        else [],
    )
    with RunMonitor(root / "resources", "trainer_train"):
        training_result = trainer.train()
    trainer.save_metrics("train", training_result.metrics)
    trainer.save_model(str(root / "adapter"))
    processor.save_pretrained(root / "adapter")
    trainer.state.save_to_json(str(root / "trainer_state.json"))
    atomic_json(
        root / "training.complete.json",
        {
            "global_step": trainer.state.global_step,
            "checkpoint": str(root / "adapter"),
            "files": {
                p.name: sha256(p) for p in (root / "adapter").iterdir() if p.is_file()
            },
            "criterion": "Fixed final step, no judge-based selection",
        },
    )


if __name__ == "__main__":
    main()
