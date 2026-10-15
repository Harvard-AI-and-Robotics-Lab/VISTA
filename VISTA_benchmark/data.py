#data.py VISTA

import os
import pandas as pd


def load_dataset(dataset_root, csv_path, split_name=None):
    """
    Load VISTA dataset from a CSV file.
    Returns a list of dicts with clip_id, video_path, and annotations.
    """
    df = pd.read_csv(csv_path)
    samples = []
    missing = 0

    for _, row in df.iterrows():
        clip_id = str(row["clip_id"])

        # Find TC folder from clip_id (e.g. TC01_002 -> TC01)
        tc_folder = clip_id.split("_")[0]
        clip_dir = os.path.join(dataset_root, tc_folder, clip_id)

        video_path = os.path.join(clip_dir, f"{clip_id}.mp4")
        if not os.path.exists(video_path):
            print(f"[WARNING] Video not found: {video_path}")
            missing += 1
            continue

        # ── Scene descriptions ────────────────────────────────────────────────
        scene_descriptions = [
            str(row[f"scene_description_{i}"])
            for i in range(1, 9)
            if f"scene_description_{i}" in row and pd.notna(row[f"scene_description_{i}"])
            and str(row[f"scene_description_{i}"]).strip()
        ]

        # ── Q/A pairs ─────────────────────────────────────────────────────────
        qa_pairs = []
        for i in range(1, 9):
            q_col = f"question_{i}"
            a_col = f"answer_{i}"
            if q_col in row and a_col in row:
                q = row[q_col]
                a = row[a_col]
                if pd.notna(q) and pd.notna(a) and str(q).strip():
                    qa_pairs.append({
                        "index": i,
                        "question": str(q).strip(),
                        "answer": str(a).strip()
                    })

        # ── Goal / instruction pairs ──────────────────────────────────────────
        actions = []
        for i in range(1, 18):
            g_col = f"goal_{i}"
            inst_col = f"instruction_{i}"
            if g_col in row and inst_col in row:
                g = row[g_col]
                inst = row[inst_col]
                if pd.notna(g) and pd.notna(inst) and str(g).strip():
                    actions.append({
                        "index": i,
                        "goal": str(g).strip(),
                        "instruction": str(inst).strip()
                    })

        samples.append({
            "clip_id": clip_id,
            "video_path": video_path,
            "scene_descriptions": scene_descriptions,
            "qa_pairs": qa_pairs,
            "actions": actions,
        })

    print(f"Loaded {len(samples)} clips from {csv_path} "
          f"({len(df)} CSV rows, {missing} without a video file)")

    # Fail loudly instead of producing an empty, exit-0 "successful" run. A wrong
    # --dataset_root used to yield "Loaded 0 clips" and then write nothing at all.
    if not samples:
        raise RuntimeError(
            f"No clips loaded. Checked dataset_root={dataset_root!r} against "
            f"{len(df)} rows in {csv_path!r}. Expected videos at "
            f"<dataset_root>/<TCxx>/<clip_id>/<clip_id>.mp4"
        )
    if missing:
        print(f"[WARNING] {missing}/{len(df)} clips in the CSV have no video and were skipped.")

    return samples
