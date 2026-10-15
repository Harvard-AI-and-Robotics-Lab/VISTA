"""
dataset_egoblind.py — EgoBlind-specific adapter for the standalone v5 judge engine.

It exposes data loading and frame extraction:
    load_items(csv_path, preds_path) -> list[dict]   canonical items
    get_frames(item, num_frames)     -> list[PIL.Image]

Plus DATASET_ROOT, set by the engine from --dataset-root.

A canonical item contains item_id (unique, used for resume), qtype (used for
aggregation), question, references, and prediction. This adapter also supplies
video_name and timestamp for frame lookup and sampling.
"""

import json

import cv2
import pandas as pd
from PIL import Image


# Set by the engine (vlm_judge.py) from --dataset-root before load.
DATASET_ROOT = None

REF_COLS = ["answer0", "answer1", "answer2", "answer3"]

# Video-number ranges -> split folder
SPLIT_RANGES = {
    "split_0": (923, 988),     "split_1": (989, 1054),
    "split_2": (1055, 1119),   "split_3": (1120, 1185),
    "split_4": (1186, 1250),   "split_5": (1251, 1316),
    "split_6": (1317, 1382),   "split_7": (1383, 1447),
    "split_8": (1448, 1513),   "split_9": (1514, 1578),
}


# ---------------------------------------------------------------------------
# load_items: join test.csv with the candidate's predictions on question_id
# ---------------------------------------------------------------------------

def _load_predictions(preds_path):
    preds = {}
    with open(preds_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            preds[obj["question_id"]] = obj["pred"]
    return preds


def _clean_references(row):
    refs = []
    for col in REF_COLS:
        if col in row and not pd.isna(row[col]):
            val = str(row[col]).strip()
            if val:
                refs.append(val)
    return refs


def load_items(csv_path, preds_path):
    df = pd.read_csv(csv_path)
    preds = _load_predictions(preds_path)

    items, missing = [], 0
    for _, row in df.iterrows():
        qid = row["question_id"]
        if qid not in preds:
            missing += 1
            continue
        items.append({
            "item_id":    qid,                       # required by the engine
            "qtype":      str(row.get("type", "")),  # required by the engine
            "video_name": str(row["video_name"]),
            "question":   str(row["question"]),
            "references": _clean_references(row),
            "prediction": preds[qid],
            "timestamp":  float(row["start-time/s"]),
        })
    if missing:
        print(f"[egoblind] {missing} rows skipped (no matching prediction)")
    print(f"[egoblind] {len(items)} items ready")
    return items


# ---------------------------------------------------------------------------
# get_frames: resolve the video, sample N frames uniformly up to the timestamp
# ---------------------------------------------------------------------------

def _find_video_path(video_name):
    num = int(video_name)
    for split, (start, end) in SPLIT_RANGES.items():
        if start <= num <= end:
            import os
            path = os.path.join(DATASET_ROOT, split, f"{num:05d}.mp4")
            return path if os.path.exists(path) else None
    return None


def _uniform_indices(end_frame, n):
    if end_frame <= 0:
        return [0]
    if n <= 1:
        return [end_frame // 2]
    raw = [round(i * end_frame / (n - 1)) for i in range(n)]
    seen, out = set(), []
    for idx in raw:
        idx = min(idx, end_frame)
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def get_frames(item, num_frames):
    """Return a list of PIL RGB frames truncated at the item's timestamp.
    Empty list if the video is missing/unreadable (engine logs it as failed)."""
    video_path = _find_video_path(item["video_name"])
    if video_path is None:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0 or fps <= 0:
        cap.release()
        return []

    end_frame = min(int(item["timestamp"] * fps), total - 1)
    if end_frame < 0:
        end_frame = 0

    frames = []
    for idx in _uniform_indices(end_frame, num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, bgr = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames
