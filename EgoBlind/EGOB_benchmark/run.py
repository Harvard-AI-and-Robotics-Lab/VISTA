import os
import contextlib
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import argparse
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

import multiprocessing
from config import SEED, NUM_FRAMES
from data import load_dataset, extract_frames
from models import LLAMA3_VIDEO_RESIZE, load_model, run_model, free_model
from prompts import build_blind_prompt

np.random.seed(SEED)
load_dotenv(override=True)

############################################
# ARGS
############################################

parser = argparse.ArgumentParser(description="Run EgoBlind benchmark")
parser.add_argument("--dataset_root",      required=True)
parser.add_argument("--csv_path",          required=True)
parser.add_argument("--output_dir",        default="outputs")
parser.add_argument("--model",             required=True, choices=[
    "gemini_3_1_flash_lite", "gpt_realtime_mini",
    "internvl3_1b", "internvl3_2b", "internvl3_8b", "internvl3_14b",
    "llava_video", "llama3_video", "gemma3_4b",
    "qwen3vl_2b", "qwen3vl_4b", "qwen3vl_8b", "qwen3vl_32b"
])
parser.add_argument("--gemini_api_key",    default=None)
parser.add_argument("--openai_api_key",    default=None)
parser.add_argument("--hf_token",          default=None)
parser.add_argument("--limit",             type=int, default=None)
parser.add_argument(
    "--num_frames",
    dest="num_frames",
    type=int,
    default=None,
    help="Override the model's configured frame count. Omit to use config.py.",
)
parser.add_argument(
    "--llama3_video_resize",
    type=int,
    default=LLAMA3_VIDEO_RESIZE,
    help="Resize VideoLLaMA3 frames to this square size. Use 0 to disable resizing.",
)
args = parser.parse_args()

args.gemini_api_key = args.gemini_api_key or os.getenv("GEMINI_API_KEY")
args.openai_api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
args.hf_token = args.hf_token or os.getenv("HF_TOKEN")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    import torch
    torch.manual_seed(SEED)

    os.makedirs(args.output_dir, exist_ok=True)
    OUTPUT_FILE = os.path.join(args.output_dir, f"predictions_{args.model}.jsonl")
    ERROR_FILE  = os.path.join(args.output_dir, f"errors_{args.model}.jsonl")

    # Load already processed question_ids for resumption
    completed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    completed_ids.add(item["question_id"])
                except:
                    pass
        print(f"[INFO] Resuming — {len(completed_ids)} predictions already done")

    # Load model
    if args.model in {"qwen3vl_2b", "qwen3vl_4b", "qwen3vl_8b", "qwen3vl_32b", "internvl3_1b", "internvl3_2b", "internvl3_8b", "internvl3_14b", "llava_video", "llama3_video", "gemma3_4b"}:
        load_model(args.model, hf_token=args.hf_token)

    # Load dataset
    samples = load_dataset(args.dataset_root, args.csv_path)
    if args.limit is not None:
        samples = samples[:args.limit]
        print(f"[INFO] Limit applied: {args.limit} samples")

    print(f"[INFO] Model  : {args.model}")
    print(f"[INFO] Samples: {len(samples)}")
    if args.num_frames is not None:
        print(f"[INFO] Frame override: {args.num_frames}")
    else:
        print(f"[INFO] Frames from config: {NUM_FRAMES[args.model]}")

    total   = len(samples)
    done    = len(completed_ids)
    errors  = 0
    skipped = 0

    with open(OUTPUT_FILE, "a") as out_f, open(ERROR_FILE, "a") as err_f:
        pbar = tqdm(samples, desc="Running inference", total=total,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
        pbar.set_postfix(done=done, errors=errors, skipped=skipped)
        for sample in pbar:
            question_id = sample["question_id"]
            if question_id in completed_ids:
                skipped += 1
                pbar.set_postfix(done=done, errors=errors, skipped=skipped)
                continue
            try:
                num_frames = args.num_frames if args.num_frames is not None else NUM_FRAMES[args.model]
                frames = extract_frames(
                    sample["video_path"],
                    sample["timestamp"],
                    num_frames=num_frames
                )
                if not frames:
                    raise ValueError(f"No frames extracted for {question_id}")

                prompt = build_blind_prompt(sample["question"])
                with open(os.devnull, 'w') as devnull, contextlib.redirect_stderr(devnull):
                    pred = run_model(
                        frames=frames,
                        question=prompt,
                        model_key=args.model,
                        gemini_api_key=args.gemini_api_key,
                        openai_api_key=args.openai_api_key,
                        hf_token=args.hf_token,
                        llama3_video_resize=args.llama3_video_resize
    )
                result = {"question_id": question_id, "pred": pred}
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()
                done += 1
                pbar.set_postfix(done=done, errors=errors, skipped=skipped)

            except Exception as e:
                import traceback
                import gc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                error = {"question_id": question_id, "error": str(e), "traceback": traceback.format_exc()}
                err_f.write(json.dumps(error) + "\n")
                err_f.flush()
                errors += 1
                pbar.set_postfix(done=done, errors=errors, skipped=skipped)

    free_model()
    print(f"\n[INFO] Done. Predictions saved to {OUTPUT_FILE}")
    print(f"[INFO] Errors saved to {ERROR_FILE}")
