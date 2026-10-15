import os
import cv2
import pandas as pd

# Split ranges mapping
SPLIT_RANGES = {
    "split_0": (923,  988),
    "split_1": (989,  1054),
    "split_2": (1055, 1119),
    "split_3": (1120, 1185),
    "split_4": (1186, 1250),
    "split_5": (1251, 1316),
    "split_6": (1317, 1382),
    "split_7": (1383, 1447),
    "split_8": (1448, 1513),
    "split_9": (1514, 1578),
}

def find_split(video_num):
    """Find the split folder corresponding to the video number."""
    for split, (start, end) in SPLIT_RANGES.items():
        if start <= video_num <= end:
            return split
    return None

def find_video_path(dataset_root, video_name):
    """Find the full path of the video file."""
    video_num = int(video_name)
    split = find_split(video_num)
    if split is None:
        return None
    video_file = f"{int(video_name):05d}.mp4"
    video_path = os.path.join(dataset_root, split, video_file)
    return video_path if os.path.exists(video_path) else None

def extract_frames(video_path, timestamp, num_frames=8):
    """
    Extract frames uniformly up to the given timestamp.
    If num_frames is None, extract at 1fps up to the timestamp.
    Returns a list of frames (numpy arrays in BGR format).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = int(timestamp * fps)
    end_frame = min(end_frame, total_frames - 1)

    if end_frame <= 0:
        end_frame = total_frames - 1

    if num_frames is None:
        # 1fps mode: one frame per second up to timestamp
        indices = [int(i * fps) for i in range(int(timestamp) + 1)]
        indices = [min(i, end_frame) for i in indices]
    else:
        # Uniform sampling mode
        indices = [int(i * end_frame / (num_frames - 1)) for i in range(num_frames)]

    indices = sorted(set(indices))

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames

def load_dataset(dataset_root, csv_path):
    """
    Load the EgoBlind dataset from the CSV file.
    Returns a list of dicts with question_id, question, timestamp, video_path.
    """
    df = pd.read_csv(csv_path)
    samples = []

    for _, row in df.iterrows():
        question_id = row["question_id"]
        video_name  = str(row["video_name"])
        question    = row["question"]
        timestamp   = float(row["start-time/s"])

        video_path = find_video_path(dataset_root, video_name)
        if video_path is None:
            print(f"[WARNING] Video not found for {question_id} (video {video_name})")
            continue

        samples.append({
            "question_id": question_id,
            "question":    question,
            "timestamp":   timestamp,
            "video_path":  video_path,
        })

    print(f"Loaded {len(samples)} samples from {csv_path}")
    return samples