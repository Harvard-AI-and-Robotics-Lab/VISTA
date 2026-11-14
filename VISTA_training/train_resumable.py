"""Single-GPU RGB training with complete checkpoints and cooperative pauses.

Create PAUSE_REQUESTED in the model's training directory to save at the next
optimizer boundary and exit with status 75. Restart with --resume to continue.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import run as launcher


def read_json(path):
    return json.loads(Path(path).read_text())


def checkpoint_path(root):
    candidates = []
    for path in root.glob('checkpoint-*'):
        marker = path / 'checkpoint.complete.json'
        if marker.is_file():
            record = read_json(marker)
            if path.name == f"checkpoint-{record['global_step']}":
                candidates.append((record['global_step'], path))
    if not candidates:
        raise FileNotFoundError(f'No complete resumable checkpoint in {root}')
    _, path = max(candidates)
    for name, expected in read_json(path / 'checkpoint.complete.json')['files'].items():
        if launcher.sha(path / name) != expected:
            raise ValueError(f'Checkpoint checksum mismatch: {path / name}')
    for name in ('adapter_model.safetensors', 'optimizer.pt', 'scheduler.pt',
                 'trainer_state.json', 'rng_state.pth'):
        if not (path / name).is_file():
            raise ValueError(f'Incomplete resumable checkpoint: {path / name}')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=tuple(launcher.MODELS), required=True)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--training-root', default=str(launcher.ROOT / 'runs/rgb'))
    parser.add_argument('--stage', choices=('stage1', 'stage2', 'both'), default='both')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint-steps', type=int, default=5)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--smoke-test', action='store_true', help='Three steps per stage, effective batch 1')
    args = parser.parse_args()
    if args.checkpoint_steps < 1 or args.workers < 0:
        parser.error('checkpoint-steps must be positive and workers must be nonnegative')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        parser.error('This recovery entry point supports one GPU')
    runtime = launcher.runtime_for(launcher.MODELS[args.model]['runtime'])
    sys.path.insert(0, str(runtime))
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    import torch
    from transformers import TrainerCallback, TrainingArguments, set_seed
    from vista_mm.config import Experiment
    from vista_mm.data import make_dataset, VistaCollator, verify_prepared
    from vista_mm.metrics import RunMonitor
    from vista_mm.provenance import atomic_json, environment_record
    from vista_mm.train import load_backbone, configure_trainable, VistaTrainer

    model_root = Path(args.training_root).resolve() / args.model
    pause_file = model_root / 'PAUSE_REQUESTED'
    stages = ('stage1', 'stage2') if args.stage == 'both' else (args.stage,)
    source = {str(p.relative_to(runtime)): launcher.sha(p)
              for p in (runtime / 'vista_mm').glob('*.py')}
    source['train_resumable.py'] = launcher.sha(Path(__file__))

    class RecoveryCallback(TrainerCallback):
        def on_step_end(self, training_args, state, control, **kwargs):
            # Save the first step too, then keep only two recent checkpoints.
            if state.global_step == 1:
                control.should_save = True
            if pause_file.exists():
                after = int(pause_file.read_text().strip() or '0')
                if state.global_step >= after:
                    control.should_save = True
                    control.should_training_stop = True
            return control

        def on_save(self, training_args, state, control, **kwargs):
            path = Path(training_args.output_dir) / f'checkpoint-{state.global_step}'
            files = {p.name: launcher.sha(p) for p in path.iterdir()
                     if p.is_file() and p.name != 'checkpoint.complete.json'}
            atomic_json(path / 'checkpoint.complete.json',
                        dict(global_step=state.global_step, files=files))
            atomic_json(model_root / 'progress.json', dict(stage=stage,
                        global_step=state.global_step, target_steps=cfg.max_steps,
                        checkpoint=str(path), status='checkpoint_saved'))
            return control

    for stage in stages:
        config = launcher.model_configuration(args.model, stage, 'rgb')
        root = model_root / stage
        config.update(prepared=str(Path(args.prepared).resolve()), output_dir=str(root),
                      batch_size=1, accumulation=32, workers=args.workers)
        if stage == 'stage2':
            config['initial_adapter'] = str(model_root / 'stage1/adapter')
        if args.smoke_test:
            config.update(max_steps=3, accumulation=1, workers=0)
        cfg = Experiment(**config)
        manifest, _ = verify_prepared(cfg.prepared, [], 0)
        resume = None
        if root.exists() and any(root.iterdir()):
            if not args.resume:
                raise FileExistsError(f'Use --resume for existing run: {root}')
            if (read_json(root / 'experiment.json') != cfg.to_dict()
                    or read_json(root / 'prepared_manifest.json')['prepared_id'] != manifest['prepared_id']
                    or read_json(root / 'run.json')['source_sha256'] != source):
                raise ValueError('Resume configuration, prepared data, or source differs from saved run')
            complete = root / 'training.complete.json'
            if complete.exists():
                record = read_json(complete)
                assert record['global_step'] == cfg.max_steps
                for name, checksum in record['files'].items():
                    assert launcher.sha(root / 'adapter' / name) == checksum
                continue
            resume = checkpoint_path(root)
        else:
            root.mkdir(parents=True, exist_ok=True)
            atomic_json(root / 'experiment.json', cfg.to_dict())
            atomic_json(root / 'prepared_manifest.json', manifest)
            atomic_json(root / 'run.json', dict(source_sha256=source,
                        environment=environment_record(), input_mode='rgb',
                        checkpoint_steps=args.checkpoint_steps, workers=args.workers,
                        fixed_final_step=cfg.max_steps, planned_lora_update_steps=cfg.max_steps))
        set_seed(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        model, processor = load_backbone(cfg)
        model = configure_trainable(model, cfg)
        training_args = TrainingArguments(
            output_dir=str(root), seed=cfg.seed, data_seed=cfg.seed,
            max_steps=cfg.max_steps, num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.accumulation,
            learning_rate=cfg.backbone_lr, weight_decay=cfg.weight_decay,
            lr_scheduler_type='cosine', warmup_ratio=cfg.warmup_ratio,
            max_grad_norm=cfg.max_grad_norm, bf16=True, tf32=True,
            gradient_checkpointing=cfg.gradient_checkpointing,
            gradient_checkpointing_kwargs={'use_reentrant':False},
            eval_strategy='no', save_strategy='steps', save_steps=args.checkpoint_steps,
            save_total_limit=2, save_only_model=False, logging_steps=1,
            logging_nan_inf_filter=False, prediction_loss_only=True,
            remove_unused_columns=False, dataloader_num_workers=cfg.workers,
            dataloader_pin_memory=True, report_to=['tensorboard'])
        trainer = VistaTrainer(
            model=model, args=training_args,
            train_dataset=make_dataset(cfg.prepared, 'train', cfg, training=True),
            data_collator=VistaCollator(processor, cfg, training=True),
            processing_class=processor, sensor_lr=cfg.sensor_lr, encoder_lr=cfg.encoder_lr,
            callbacks=[RecoveryCallback()])
        atomic_json(root / 'parameters.json', dict(
            trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
            total=sum(p.numel() for p in model.parameters()), gpu=torch.cuda.get_device_name(),
            effective_batch=cfg.batch_size * cfg.accumulation))
        if resume:
            atomic_json(root / 'resume.json', dict(checkpoint=str(resume),
                        global_step=read_json(resume / 'trainer_state.json')['global_step']))
        with RunMonitor(root / 'resources', 'trainer_train'):
            result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        trainer.state.save_to_json(str(root / 'trainer_state.json'))
        if trainer.state.global_step < cfg.max_steps:
            checkpoint = checkpoint_path(root)
            atomic_json(root / 'training.paused.json', dict(global_step=trainer.state.global_step,
                        checkpoint=str(checkpoint), reason='PAUSE_REQUESTED'))
            atomic_json(model_root / 'progress.json', dict(stage=stage, status='paused',
                        global_step=trainer.state.global_step, checkpoint=str(checkpoint)))
            return 75
        trainer.save_metrics('train', result.metrics)
        trainer.save_model(str(root / 'adapter'))
        processor.save_pretrained(root / 'adapter')
        atomic_json(root / 'training.complete.json', dict(global_step=trainer.state.global_step,
                    checkpoint=str(root / 'adapter'), criterion='Fixed final step, no judge-based selection',
                    files={p.name:launcher.sha(p) for p in (root / 'adapter').iterdir() if p.is_file()}))
        paused = root / 'training.paused.json'
        if paused.exists():
            paused.unlink()
        atomic_json(model_root / 'progress.json', dict(stage=stage, status='complete',
                    global_step=trainer.state.global_step, checkpoint=str(root / 'adapter')))
        del trainer, model, processor
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
