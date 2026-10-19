# run VISTA

import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data import load_dataset
from models import (
    CLOSED_SOURCE_MODELS,
    GeminiDailyQuotaExhaustedError,
    OPEN_SOURCE_MODELS,
    decoding_signature,
    extract_frames,
    free_model,
    guard_decoding,
    load_model,
    run_model,
)
from prompts import build_guidance_prompt, build_qa_prompt, build_scene_prompt


def _cred(name):
    """Environment credential, with empty string normalised to None.

    An empty-but-set value is not the same as an absent one: huggingface_hub turns
    HF_TOKEN="" into the header `Authorization: Bearer ` and raises
    httpx.LocalProtocolError: Illegal header value b'Bearer '.
    """
    value = os.environ.get(name)
    return value if value else None


def parse_args():
    parser = argparse.ArgumentParser(description="Run VISTA benchmark")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(CLOSED_SOURCE_MODELS | OPEN_SOURCE_MODELS),
    )
    parser.add_argument("--gemini_api_key", default=_cred("GEMINI_API_KEY"))
    parser.add_argument("--openai_api_key", default=_cred("OPENAI_API_KEY"))
    parser.add_argument("--hf_token", default=_cred("HF_TOKEN"))
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of clips loaded before task expansion")
    parser.add_argument("--max_predictions", type=int, default=None,
                        help="Run only the first N expanded tasks")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument(
        "--output_name",
        default=None,
        help="Suffix for predictions_<name>.jsonl; defaults to the model key",
    )
    return parser.parse_args()


def build_tasks(samples):
    tasks = []
    for sample in samples:
        tasks.append({
            "sample": sample,
            "task_type": "scene_description",
            "prompt": build_scene_prompt(),
        })
        for qa in sample["qa_pairs"]:
            index = qa["index"]
            tasks.append({
                "sample": sample,
                "task_type": f"question_{index}",
                "prompt": build_qa_prompt(qa["question"]),
            })
        for action in sample["actions"]:
            index = action["index"]
            tasks.append({
                "sample": sample,
                "task_type": f"goal_{index}",
                "prompt": build_guidance_prompt(action["goal"]),
            })
    return tasks


def load_completed(output_file):
    completed = set()
    if not os.path.exists(output_file):
        return completed
    with open(output_file, "r", encoding="utf-8") as file:
        for line in file:
            try:
                item = json.loads(line)
                input_mode = item.get("input_mode", "rgb_text")
                if input_mode != "rgb_text":
                    raise ValueError(
                        f"Cannot resume {input_mode!r} predictions in an RGB/text run. "
                        "Use a new --output_dir or --output_name."
                    )
                completed.add((item["clip_id"], item["task_type"]))
            except (json.JSONDecodeError, KeyError):
                pass
    return completed


def validate_args(args):
    if args.num_frames < 2:
        raise ValueError("--num_frames must be at least 2")
    if args.max_predictions is not None and args.max_predictions < 0:
        raise ValueError("--max_predictions must be non-negative")


def main():
    args = parse_args()
    validate_args(args)

    os.makedirs(args.output_dir, exist_ok=True)
    output_name = args.output_name or args.model
    output_file = os.path.join(args.output_dir, f"predictions_{output_name}.jsonl")
    error_file = os.path.join(args.output_dir, f"errors_{output_name}.jsonl")

    completed = load_completed(output_file)
    if completed:
        print(f"[INFO] Resuming - {len(completed)} rows already done")
    # Before touching a GPU: never append to a file generated under different decoding.
    guard_decoding(args.output_dir, output_name, len(completed))
    print(f"[INFO] Decoding   : {decoding_signature()}")

    if args.model in OPEN_SOURCE_MODELS:
        load_model(args.model, hf_token=args.hf_token)

    samples = load_dataset(args.dataset_root, args.csv_path)
    if args.limit is not None:
        samples = samples[:args.limit]
        print(f"[INFO] Clip limit applied: {args.limit}")

    tasks = build_tasks(samples)
    if args.max_predictions is not None:
        tasks = tasks[:args.max_predictions]

    print(f"[INFO] Model      : {args.model}")
    print("[INFO] Input mode : rgb_text")
    print(f"[INFO] Frames     : {args.num_frames}")
    print(f"[INFO] Tasks      : {len(tasks)}")

    api_keys = {
        "gemini_api_key": args.gemini_api_key,
        "openai_api_key": args.openai_api_key,
        "hf_token": args.hf_token,
    }

    current_clip_id = None
    current_frames = None
    fatal_error = None
    with (
        open(output_file, "a", encoding="utf-8") as output_stream,
        open(error_file, "a", encoding="utf-8") as error_stream,
    ):
        for task in tqdm(tasks, desc="Running inference"):
            sample = task["sample"]
            clip_id = sample["clip_id"]
            task_type = task["task_type"]
            if (clip_id, task_type) in completed:
                continue

            try:
                if clip_id != current_clip_id:
                    current_frames = extract_frames(
                        sample["video_path"], num_frames=args.num_frames
                    )
                    if not current_frames:
                        raise ValueError(f"No frames extracted for {clip_id}")
                    current_clip_id = clip_id

                prompt = task["prompt"]

                prediction = run_model(
                    current_frames,
                    prompt,
                    args.model,
                    **api_keys,
                )
                output_stream.write(json.dumps({
                    "clip_id": clip_id,
                    "task_type": task_type,
                    "prediction": prediction,
                    "input_mode": "rgb_text",
                    "num_frames": args.num_frames,
                    "visual_overlay": "none",
                }, ensure_ascii=False) + "\n")
                output_stream.flush()
                completed.add((clip_id, task_type))
            except Exception as error:
                import traceback

                error_stream.write(json.dumps({
                    "clip_id": clip_id,
                    "task_type": task_type,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }, ensure_ascii=False) + "\n")
                error_stream.flush()
                print(f"[ERROR] {clip_id} {task_type}: {error}")
                if isinstance(error, GeminiDailyQuotaExhaustedError):
                    fatal_error = error
                    break

    free_model()
    if fatal_error:
        raise fatal_error
    print(f"\n[INFO] Done. Predictions saved to {output_file}")
    print(f"[INFO] Errors saved to {error_file}")


if __name__ == "__main__":
    main()
