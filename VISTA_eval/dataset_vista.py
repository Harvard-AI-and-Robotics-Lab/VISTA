"""
dataset_vista.py — VISTA-specific adapter for the standalone v5 judge engine.

Exposes:
    load_items(csv_path, preds_path) -> list[dict]
    get_frames(item, num_frames)     -> list[PIL.Image]

VISTA has three task types per clip:
  - scene_description : free-form scene description (up to 9 reference descriptions)
  - question_i        : answer to question_i (single reference: answer_i)
  - goal_i            : step-by-step guidance for goal_i (single reference: instruction_i)

Prediction files are predictions_<model>.jsonl from VISTA_benchmark/run.py:
    {"clip_id": "TC01_001", "task_type": "scene_description", "prediction": "..."}
    {"clip_id": "TC01_001", "task_type": "question_1",        "prediction": "..."}
    {"clip_id": "TC01_001", "task_type": "goal_1",            "prediction": "..."}

Plus DATASET_ROOT, set by the engine from --dataset-root.
"""

import json
import os

import cv2
import pandas as pd
from PIL import Image


DATASET_ROOT = None

SCENE_DESC_COLS = [f"scene_description_{i}" for i in range(1, 10)]
MAX_QA    = 9
MAX_GOALS = 15  # CSV goes up to goal_15/instruction_15


# ---------------------------------------------------------------------------
# load_items: explode each CSV row into one item per (clip_id, task_type)
# ---------------------------------------------------------------------------

def _iter_jsonl_lines(path):
    """Yield decoded lines. Some gpt_realtime_mini VISTA prediction files carry a
    stray cp1252 curly-quote byte (0x92 = ') in otherwise UTF-8 text -- fall back
    to cp1252 for just the offending line instead of crashing the whole load."""
    with open(path, "rb") as f:
        for raw in f:
            try:
                yield raw.decode("utf-8")
            except UnicodeDecodeError:
                yield raw.decode("cp1252")


def _load_predictions(preds_path):
    """Key: (clip_id, task_type) -> prediction string."""
    preds = {}
    for line in _iter_jsonl_lines(preds_path):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        preds[(obj["clip_id"], obj["task_type"])] = obj["prediction"]
    return preds


def _clean_vals(row, cols):
    vals = []
    for col in cols:
        if col in row and not pd.isna(row[col]):
            v = str(row[col]).strip()
            if v:
                vals.append(v)
    return vals


def load_items(csv_path, preds_path):
    df = pd.read_csv(csv_path)
    preds = _load_predictions(preds_path)

    items, missing = [], 0
    for _, row in df.iterrows():
        clip_id = str(row["clip_id"])

        # ── scene_description ────────────────────────────────────────────────
        key = (clip_id, "scene_description")
        refs = _clean_vals(row, SCENE_DESC_COLS)
        if refs:
            if key in preds:
                items.append({
                    "item_id":   f"{clip_id}|scene_description",
                    "qtype":     "scene_description",
                    "clip_id":   clip_id,
                    "task_type": "scene_description",
                    "question":  "Describe the scene visible in the video.",
                    "references": refs,
                    "prediction": preds[key],
                })
            else:
                missing += 1

        # ── question_i / answer_i ─────────────────────────────────────────────
        for i in range(1, MAX_QA + 1):
            q_col, a_col = f"question_{i}", f"answer_{i}"
            if q_col not in row or pd.isna(row[q_col]):
                continue
            q = str(row[q_col]).strip()
            if not q:
                continue
            a = str(row[a_col]).strip() if a_col in row and not pd.isna(row[a_col]) else ""
            if not a:
                continue
            key = (clip_id, f"question_{i}")
            if key in preds:
                items.append({
                    "item_id":   f"{clip_id}|question_{i}",
                    "qtype":     "qa",
                    "clip_id":   clip_id,
                    "task_type": f"question_{i}",
                    "question":  q,
                    "references": [a],
                    "prediction": preds[key],
                })
            else:
                missing += 1

        # ── goal_i / instruction_i ────────────────────────────────────────────
        for i in range(1, MAX_GOALS + 1):
            g_col, inst_col = f"goal_{i}", f"instruction_{i}"
            if g_col not in row or pd.isna(row[g_col]):
                continue
            g = str(row[g_col]).strip()
            if not g:
                continue
            inst = str(row[inst_col]).strip() if inst_col in row and not pd.isna(row[inst_col]) else ""
            if not inst:
                continue
            key = (clip_id, f"goal_{i}")
            if key in preds:
                items.append({
                    "item_id":   f"{clip_id}|goal_{i}",
                    "qtype":     "guidance",
                    "clip_id":   clip_id,
                    "task_type": f"goal_{i}",
                    "question":  g,
                    "references": [inst],
                    "prediction": preds[key],
                })
            else:
                missing += 1

    if missing:
        print(f"[vista] {missing} items skipped (no matching prediction)")
    print(f"[vista] {len(items)} items ready")
    return items


# ---------------------------------------------------------------------------
# get_frames: resolve the video, sample N frames uniformly over the full clip
# ---------------------------------------------------------------------------

def _find_video_path(clip_id):
    tc_folder = clip_id.split("_")[0]
    # Try blurred first (canonical), then plain mp4
    for fname in (f"{clip_id}_blurred.mp4", f"{clip_id}.mp4"):
        path = os.path.join(DATASET_ROOT, tc_folder, clip_id, fname)
        if os.path.exists(path):
            return path
    return None


def _uniform_indices(total_frames, n):
    if total_frames <= 0:
        return [0]
    if n <= 1:
        return [total_frames // 2]
    raw = [round(i * total_frames / (n - 1)) for i in range(n)]
    seen, out = set(), []
    for idx in raw:
        idx = min(idx, total_frames - 1)
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def get_frames(item, num_frames):
    """Return a list of PIL RGB frames sampled uniformly over the full clip.
    Empty list if the video is missing/unreadable (engine logs it as failed)."""
    video_path = _find_video_path(item["clip_id"])
    if video_path is None:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    frames = []
    for idx in _uniform_indices(total - 1, num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, bgr = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames
