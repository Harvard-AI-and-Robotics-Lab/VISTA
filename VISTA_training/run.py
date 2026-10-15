"""Train, predict, or score the five archived VISTA models."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
DATABASE = json.loads((ROOT / 'models.json').read_text())
MODELS = DATABASE['models']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2) + '\n'
    if path.exists() and path.read_text() == text:
        return
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    temporary.replace(path)


def verify_runtime(path, family):
    provenance = json.loads((ROOT / 'provenance.json').read_text())
    expected = provenance['runtime_hashes'][family]
    for relative, checksum in expected.items():
        if sha(path / relative) != checksum:
            raise ValueError(f'Runtime differs from declared {family} hashes: {relative}')
    present = {p.relative_to(path).as_posix() for p in (path / 'vista_mm').glob('*.py')}
    if present != set(expected):
        raise ValueError('Unexpected runtime Python files')


def runtime_for(family):
    verify_runtime(ROOT, 'native')
    if family == 'native':
        return ROOT
    provenance = json.loads((ROOT / 'provenance.json').read_text())
    checksum = sha(ROOT / 'qwen.patch')
    if checksum != provenance['qwen_patch_sha256']:
        raise ValueError('Qwen runtime patch has changed')
    target = ROOT / 'outputs/runtime' / ('qwen-' + checksum[:12])
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix='.qwen-', dir=target.parent))
        try:
            shutil.copytree(ROOT / 'vista_mm', temporary / 'vista_mm',
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            subprocess.run(['patch', '--batch', '-p1', '--input', str(ROOT / 'qwen.patch')],
                           cwd=temporary, capture_output=True, text=True, check=True)
            verify_runtime(temporary, 'qwen')
            try:
                temporary.rename(target)
            except FileExistsError:
                shutil.rmtree(temporary)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
    verify_runtime(target, 'qwen')
    return target


def verify():
    provenance = json.loads((ROOT / 'provenance.json').read_text())
    for name, expected in [('judge.py', provenance['judge_sha256']),
                           ('rubric.json', provenance['rubric_sha256']),
                           ('models.json', provenance['models_sha256'])]:
        if sha(ROOT / name) != expected:
            raise ValueError(f'Preserved file changed: {name}')
    for name, model in MODELS.items():
        for stage, config in model['configs'].items():
            encoded = json.dumps(config, sort_keys=True, separators=(',', ':')).encode()
            if hashlib.sha256(encoded).hexdigest() != provenance['config_digests'][name][stage]:
                raise ValueError(f'Campaign config changed: {name}/{stage}')
            if config['sensor']['modalities'] != ['imu', 'magbaro', 'gaze', 'hands', 'trajectory']:
                raise ValueError('Only the selected five-stream no-audio method is included')
    runtime_for('native')
    runtime_for('qwen')
    print(json.dumps({'status': 'verified', 'models': list(MODELS), 'configs': 15,
                      'runtimes': ['native', 'qwen'], 'paid_api_calls': 0}))


def configuration(args, stage, gpus):
    config = json.loads(json.dumps(MODELS[args.model]['configs'][stage]))
    for field in ('prepared', 'encoder_assets'):
        if getattr(args, field, None):
            config[field] = str(Path(getattr(args, field)).resolve())
    if args.action == 'train':
        training = Path(args.training_root).resolve()
        config['output_dir'] = str(training / args.model / stage)
        if stage == 'stage2':
            config['initial_adapter'] = str(training / args.model / 'stage1/adapter')
        if gpus != MODELS[args.model]['training_gpus']:
            batch = config['batch_size'] * gpus
            if 32 % batch:
                raise ValueError('Requested GPUs cannot preserve effective batch size 32')
            config['accumulation'] = 32 // batch
        if getattr(args, 'smoke_test', False):
            # Exercise both stages and leave one step after sensor-only warmup.
            # These overrides belong only to disposable smoke runs.
            config.update(max_steps=2, batch_size=1, accumulation=1, workers=0,
                          sensor_warmup_steps=1 if stage == 'stage2' else 0)
    checksum = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    path = ROOT / 'outputs/configs' / f'{args.model}-{stage}-{checksum}.json'
    if not args.dry_run:
        write_json(path, config)
    return path


def execute(command, runtime, dry_run):
    print(json.dumps({'runtime': str(runtime), 'command': command}), flush=True)
    if dry_run:
        return
    # Python otherwise prepends cwd/script-dir ahead of PYTHONPATH, masking the
    # selected Qwen runtime with the editable shared package. Also inherited by
    # torchrun's worker Python processes.
    env = dict(os.environ, PYTHONPATH=str(runtime), PYTHONSAFEPATH='1')
    env.setdefault('OMP_NUM_THREADS', '4')
    env.setdefault('TOKENIZERS_PARALLELISM', 'false')
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def download(args):
    from huggingface_hub import snapshot_download
    model = MODELS[args.model]
    if 'hub' not in model:
        raise ValueError('This model is archived on GitHub; only the InternVL adapters are on the private HF repo')
    hub = model['hub']
    cached = Path(snapshot_download(hub['repo'], revision=hub['revision'],
                                   allow_patterns=[hub['folder'] + '/*']))
    parent = ROOT / 'checkpoints' / args.model / 'stage2'
    adapter = parent / 'adapter'
    adapter.mkdir(parents=True, exist_ok=True)
    for path in (cached / hub['folder']).iterdir():
        if path.is_file():
            shutil.copy2(path, adapter / path.name)
    for name, expected in model['checkpoint_hashes'].items():
        if sha(adapter / name) != expected:
            raise ValueError(f'Downloaded checkpoint hash mismatch: {name}')
    write_json(parent / 'experiment.json', model['checkpoint_experiment'])
    write_json(parent / 'prepared_manifest.json', DATABASE['prepared_manifest'])
    print(json.dumps({'checkpoint': str(adapter), 'status': 'verified'}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    commands.add_parser('verify', help='Verify all five archived configs and runtime variants offline')
    for action in ('train', 'predict', 'judge', 'download-checkpoint'):
        p = commands.add_parser(action)
        p.add_argument('--model', choices=tuple(MODELS), required=True)
        if action == 'download-checkpoint':
            continue
        p.add_argument('--prepared', help='Complete prepared native_v2 dataset')
        p.add_argument('--encoder-assets', help='Verified pretrained encoder asset directory')
        p.add_argument('--dry-run', action='store_true', help='Print commands without training, inference, or API calls')
        if action == 'train':
            p.add_argument('--stage', choices=('stage1', 'stage2', 'both'), default='both')
            p.add_argument('--training-root', default=str(ROOT / 'runs'))
            p.add_argument('--gpus', type=int, choices=(1, 2), help='Defaults to archived GPU count; preserves effective batch 32')
            p.add_argument('--smoke-test', action='store_true',
                           help='Two optimizer steps per stage, microbatch 1, and one stage-2 warmup step')
        elif action == 'predict':
            p.add_argument('--checkpoint', help='Adapter directory; its parent needs recorded training/data metadata')
            p.add_argument('--output', help='Prediction JSONL destination')
            p.add_argument('--resume', action='store_true')
        else:
            p.add_argument('--predictions', help='Completed prediction JSONL')
            p.add_argument('--output-dir', help='Judge output directory')
            p.add_argument('--key-files', nargs='+', required=True, help='External OpenRouter key files')
            p.add_argument('--concurrency', type=int, default=12)
            p.add_argument('--max-cost-usd', type=float, default=5)
    args = parser.parse_args()
    if args.action == 'verify':
        verify()
        return
    if args.action == 'download-checkpoint':
        download(args)
        return
    model = MODELS[args.model]
    runtime = runtime_for(model['runtime'])
    gpus = getattr(args, 'gpus', None) or (
        1 if getattr(args, 'smoke_test', False) else model['training_gpus'])
    if model['runtime'] != 'qwen' and gpus != 1:
        parser.error('The archived InternVL/Gemma trainer supports one GPU; Qwen supports two')
    if args.action == 'train':
        if args.smoke_test and Path(args.training_root).resolve() == ROOT / 'runs':
            args.training_root = str(ROOT / 'outputs/smoke-runs')
        stages = ('stage1', 'stage2') if args.stage == 'both' else (args.stage,)
        for stage in stages:
            config = configuration(args, stage, gpus)
            command = [sys.executable, '-u']
            if gpus > 1:
                command += ['-m', 'torch.distributed.run', '--standalone',
                            f'--nproc-per-node={gpus}']
            command += ['-m', 'vista_mm.train', '--config', str(config)]
            execute(command, runtime, args.dry_run)
        return
    config = configuration(args, 'eval', 1)
    output = ROOT / 'outputs' / args.model
    if args.action == 'predict':
        candidates = [ROOT / 'runs' / args.model / 'stage2/adapter',
                      ROOT / 'checkpoints' / args.model / 'stage2/adapter',
                      Path('/workspace/VISTA_campaigns') / model['campaign'] / args.model / 'stage2/adapter']
        checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else next(
            (p for p in candidates if p.exists()), candidates[-1])
        command = [sys.executable, '-u', '-m', 'vista_mm.predict', '--config', str(config),
                   '--checkpoint', str(checkpoint), '--split', 'val', '--output',
                   str(Path(args.output).resolve() if args.output else output / 'predictions.jsonl'),
                   '--max-new-tokens', str(model['max_new_tokens'])]
        if args.resume:
            command.append('--resume')
    else:
        profile = ROOT / 'outputs/configs/judge-profile.json'
        no_overrides = ROOT / 'outputs/configs/no-runtime-overrides.json'
        if not args.dry_run:
            write_json(profile, DATABASE['judge_profile'])
            write_json(no_overrides, {})
        command = [sys.executable, '-u', str(ROOT / 'judge.py'), '--config', str(config),
                   '--predictions', str(Path(args.predictions).resolve() if args.predictions else output / 'predictions.jsonl'),
                   '--split', 'val', '--rubric', str(ROOT / 'rubric.json'), '--judge-profile', str(profile),
                   '--output-dir', str(Path(args.output_dir).resolve() if args.output_dir else output / 'judge3'),
                   '--key-files', *args.key_files, '--concurrency', str(args.concurrency),
                   '--max-cost-usd', str(args.max_cost_usd), '--model-label', f'VISTA {args.model}',
                   '--runtime-settings', str(no_overrides)]
    execute(command, runtime, args.dry_run)


if __name__ == '__main__':
    main()
