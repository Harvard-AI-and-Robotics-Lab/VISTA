"""Run the existing frozen VISTA models with 8 RGB + 4 paired SLAM views.

The RGB sampler, model loaders, decoding settings and task prompts are imported
unchanged from the baseline. Only the auxiliary views and their explanation are
added. Outputs are deliberately separate from the RGB-only experiment.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback

from data import load_dataset
from models import extract_frames, free_model, load_model, run_model
from prompts import build_guidance_prompt, build_qa_prompt, build_scene_prompt
from slam_frames import extract_slam_frames


HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = Path(os.environ.get('VISTA_TEST_ROOT', str(Path.home() / 'VISTA_data/test')))
MODELS = (
    'qwen3vl_2b', 'qwen3vl_4b', 'qwen3vl_8b',
    'internvl3_1b', 'internvl3_2b', 'internvl3_8b', 'internvl3_9b',
    'llama3_video',
)
CONDITION = 'rgb8_slam_left4_right4'
INPUT_DESCRIPTION = (
    'The input contains 16 images of the same clip from three cameras. '
    'Images 1-8 are the original RGB frames in chronological order. '
    'Images 9-16 are four pairs of SLAM camera views, sampled uniformly '
    'at shared times in the left and right videos: '
    '9=left(t1), 10=right(t1), 11=left(t2), 12=right(t2), '
    '13=left(t3), 14=right(t3), 15=left(t4), 16=right(t4). '
    'Each pair uses the same timestamp in the exported left and right videos. '
    'The SLAM sequence covers the same clip as the RGB sequence; '
    'it is additional camera evidence, not footage after the RGB sequence.\n\n'
)


def tasks_for(sample):
    yield 'scene_description', build_scene_prompt()
    for i, qa in enumerate(sample['qa_pairs'], 1):
        yield f'question_{i}', build_qa_prompt(qa['question'])
    for i, action in enumerate(sample['actions'], 1):
        yield f'goal_{i}', build_guidance_prompt(action['goal'])


def prepare_frames(sample):
    rgb = extract_frames(sample['video_path'], num_frames=8)
    if len(rgb) != 8:
        raise ValueError(f'Expected 8 RGB frames, decoded {len(rgb)}')
    slam, metadata = extract_slam_frames(sample['video_path'], num_pairs=4)
    if len(slam) != 8:
        raise ValueError(f'Expected 8 SLAM frames, decoded {len(slam)}')
    return rgb + slam, {
        'clip_id': sample['clip_id'], 'input_mode': CONDITION,
        'rgb_path': sample['video_path'], 'num_rgb_frames': 8,
        'num_slam_left_frames': 4, 'num_slam_right_frames': 4,
        'order': 'RGB1..RGB8,L1,R1,L2,R2,L3,R3,L4,R4',
        'slam': metadata,
    }


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight(samples, output_dir):
    errors = []
    mismatched_counts = []
    with (output_dir / 'frame_manifest.jsonl').open('w') as out:
        for i, sample in enumerate(samples, 1):
            try:
                frames, metadata = prepare_frames(sample)
                out.write(json.dumps(metadata) + '\n')
                if metadata['slam']['frame_count_mismatch']:
                    mismatched_counts.append(sample['clip_id'])
                del frames
            except Exception as error:
                errors.append({'clip_id': sample['clip_id'], 'error': str(error)})
            if i % 50 == 0:
                print(f'[PREFLIGHT] {i}/{len(samples)} clips, {len(errors)} errors', flush=True)
    summary = {
        'input_mode': CONDITION, 'num_clips': len(samples),
        'num_tasks': sum(1 for sample in samples for _ in tasks_for(sample)),
        'valid_clips': len(samples) - len(errors), 'errors': errors,
        'slam_frame_count_mismatch_clips': mismatched_counts,
        'synchronization': 'matching_exported_video_timestamps',
        'physical_capture_sync_verified': False,
    }
    write_json(output_dir / 'preflight.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 1 if errors else 0


def read_completed(path, expected):
    completed = set()
    if not path.exists():
        return completed
    with path.open() as source:
        for line_no, line in enumerate(source, 1):
            row = json.loads(line)
            if row.get('input_mode') != CONDITION:
                raise ValueError(f'{path}:{line_no}: incompatible ablation condition')
            key = row['clip_id'], row['task_type']
            if key not in expected:
                raise ValueError(f'{path}:{line_no}: task outside this run: {key}')
            if key in completed:
                raise ValueError(f'{path}:{line_no}: duplicate prediction: {key}')
            if isinstance(row.get('prediction'), str) and row['prediction'].strip():
                completed.add(key)
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--dataset_root', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--csv_path', type=Path)
    parser.add_argument('--output_dir', type=Path, default=HERE / 'outputs/rgb8_slam8')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--preflight_only', action='store_true')
    parser.add_argument('--max_consecutive_errors', type=int, default=3)
    args = parser.parse_args()
    if args.csv_path is None:
        args.csv_path = args.dataset_root / "test.csv"
    if not args.preflight_only and not args.model:
        parser.error('--model is required for inference')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_dataset(str(args.dataset_root), str(args.csv_path))
    if args.limit is not None:
        samples = samples[:args.limit]
    if not samples:
        raise ValueError('No clips loaded')
    if args.preflight_only:
        return preflight(samples, args.output_dir)

    # One writer per model/output directory, including direct invocations.
    import fcntl
    lock = (args.output_dir / f'.{args.model}.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    expected = {(s['clip_id'], t) for s in samples for t, _ in tasks_for(s)}
    configuration = {
        'model': args.model, 'input_mode': CONDITION,
        'dataset_root': str(args.dataset_root.resolve()),
        'csv_path': str(args.csv_path.resolve()), 'csv_sha256': fingerprint(args.csv_path),
        'limit': args.limit, 'num_clips': len(samples), 'num_tasks': len(expected),
        'input_description': INPUT_DESCRIPTION,
        'source_sha256': {name: fingerprint(HERE / name) for name in (
            'run_rgb_slam.py', 'slam_frames.py', 'models.py', 'prompts.py', 'data.py')},
    }
    config_path = args.output_dir / f'config_{args.model}.json'
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError('Run configuration changed; use a new output directory')
    write_json(config_path, configuration)
    predictions_path = args.output_dir / f'predictions_{args.model}.jsonl'
    completed = read_completed(predictions_path, expected)
    print(f'[INFO] {args.model}: {len(samples)} clips, {len(expected)} tasks, '
          f'{len(completed)} already complete; 8 RGB + 4 left + 4 right', flush=True)
    if completed == expected:
        return 0
    failures = 0
    consecutive_errors = 0
    load_model(args.model, hf_token=os.environ.get('HF_TOKEN'))
    try:
        with predictions_path.open('a', buffering=1) as out, \
                (args.output_dir / f'errors_{args.model}.jsonl').open('a', buffering=1) as errors, \
                (args.output_dir / f'frames_{args.model}.jsonl').open('a', buffering=1) as manifests:
            for clip_index, sample in enumerate(samples, 1):
                pending = [(task, prompt) for task, prompt in tasks_for(sample)
                           if (sample['clip_id'], task) not in completed]
                if not pending:
                    continue
                try:
                    frames, metadata = prepare_frames(sample)
                    manifests.write(json.dumps(metadata) + '\n')
                except Exception as error:
                    errors.write(json.dumps({'clip_id': sample['clip_id'],
                        'stage': 'frames', 'error': str(error),
                        'traceback': traceback.format_exc()}) + '\n')
                    failures += len(pending)
                    consecutive_errors += 1
                    if consecutive_errors >= args.max_consecutive_errors:
                        raise RuntimeError('Repeated frame preparation failures') from error
                    continue
                for task, prompt in pending:
                    try:
                        prediction = run_model(frames, INPUT_DESCRIPTION + prompt, args.model)
                        if not isinstance(prediction, str) or not prediction.strip():
                            raise ValueError('Model returned an empty prediction')
                        out.write(json.dumps({'clip_id': sample['clip_id'],
                            'task_type': task, 'prediction': prediction,
                            'input_mode': CONDITION, 'num_frames': 16}) + '\n')
                        completed.add((sample['clip_id'], task))
                        consecutive_errors = 0
                    except Exception as error:
                        failures += 1
                        consecutive_errors += 1
                        errors.write(json.dumps({'clip_id': sample['clip_id'],
                            'task_type': task, 'stage': 'inference', 'error': str(error),
                            'traceback': traceback.format_exc()}) + '\n')
                        print(f'[ERROR] {sample["clip_id"]}/{task}: {error}', flush=True)
                        if consecutive_errors >= args.max_consecutive_errors:
                            raise RuntimeError('Repeated inference failures; inspect model log') from error
                print(f'[PROGRESS] {args.model}: clip {clip_index}/{len(samples)}, '
                      f'{len(completed)}/{len(expected)} valid predictions', flush=True)
    finally:
        free_model()
        write_json(args.output_dir / f'summary_{args.model}.json', {
            'model': args.model, 'input_mode': CONDITION,
            'expected_tasks': len(expected), 'completed_tasks': len(completed),
            'missing_tasks': len(expected - completed), 'failures_this_run': failures,
            'complete': completed == expected,
        })
    return 0 if completed == expected else 1


if __name__ == '__main__':
    raise SystemExit(main())
