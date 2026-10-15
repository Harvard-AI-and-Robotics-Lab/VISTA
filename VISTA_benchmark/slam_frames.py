"""Sample paired SLAM images on the common timeline of VISTA's MP4 exports.

The current dataset has no per-frame SLAM capture timestamp sidecars. Its
``ANNONIMISATION/export_slam_qa.py`` writes the VRS records of each camera in
order to a constant-rate MP4 using the same FPS, with PTS starting at zero.
Consequently, this module verifies matching *exported video* timestamps, not
device-clock capture synchronization. In particular, unequal record counts
can indicate missing records whose location is unknown. That limitation and
the two frame counts are included in every returned manifest.
"""

import math
from pathlib import Path

import cv2


def extract_slam_frames(rgb_video_path, num_pairs=4):
    """Return BGR frames ``[L1, R1, ..., Ln, Rn]`` and JSON-safe metadata.

    SLAM files are resolved beside the RGB video, under ``vrs/``. Both views
    use exactly the same frame indices at the same FPS. The desired times
    include both ends of the common available timeline and are evenly spaced;
    each is rounded to its nearest available frame. A longer view is truncated
    to the shorter view's last timestamp, rather than sampled independently.

    Raises ValueError or RuntimeError on missing/invalid videos, too few frames,
    unequal FPS, failed reads or inconsistent decoded presentation timestamps.
    No partial result is returned, and no image resizing or overlay is applied.
    """
    if not isinstance(num_pairs, int) or isinstance(num_pairs, bool) or num_pairs < 2:
        raise ValueError("num_pairs must be an integer >= 2")

    vrs_dir = Path(rgb_video_path).parent / "vrs"
    paths = {side: vrs_dir / f"slam_{side}.mp4" for side in ("left", "right")}
    captures = {}
    streams = {}
    try:
        for side, path in paths.items():
            if not path.is_file():
                raise ValueError(f"Missing SLAM {side} video: {path}")
            cap = cv2.VideoCapture(str(path))
            captures[side] = cap
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open SLAM {side} video: {path}")
            count_value = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(count_value) or count_value < num_pairs:
                raise ValueError(f"SLAM {side} needs at least {num_pairs} frames: {path}")
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError(f"Invalid SLAM {side} FPS {fps}: {path}")
            streams[side] = {
                "path": str(path),
                "frame_count": int(count_value),
                "fps": fps,
            }

        fps = streams["left"]["fps"]
        if not math.isclose(fps, streams["right"]["fps"], rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                "SLAM left/right FPS differ; a shared frame-index timeline is invalid: "
                f"{fps} versus {streams['right']['fps']} ({vrs_dir})"
            )
        common_count = min(streams[side]["frame_count"] for side in streams)
        last_index = common_count - 1
        target_indices = [i * last_index / (num_pairs - 1) for i in range(num_pairs)]
        indices = [int(math.floor(value + 0.5)) for value in target_indices]
        if len(set(indices)) != num_pairs:
            raise ValueError(f"Not enough distinct common SLAM frames in {vrs_dir}")

        frames = []
        pairs = []
        # Much smaller than one frame, while allowing container rounding.
        pts_tolerance_s = max(1e-6, 1e-4 / fps)
        for pair_index, (target_index, frame_index) in enumerate(zip(target_indices, indices)):
            expected_time_s = frame_index / fps
            pair = {
                "pair_index": pair_index,
                "target_time_s": target_index / fps,
                "exported_time_s": expected_time_s,
                "frame_index_left": frame_index,
                "frame_index_right": frame_index,
            }
            for side in ("left", "right"):
                cap = captures[side]
                if not cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
                    raise RuntimeError(f"Cannot seek SLAM {side} frame {frame_index}: {paths[side]}")
                ok, frame = cap.read()
                if not ok or frame is None or frame.size == 0:
                    raise RuntimeError(f"Cannot decode SLAM {side} frame {frame_index}: {paths[side]}")
                next_index = cap.get(cv2.CAP_PROP_POS_FRAMES)
                if not math.isfinite(next_index) or abs(next_index - (frame_index + 1)) > 0.01:
                    raise RuntimeError(f"Wrong decoded SLAM {side} index for frame {frame_index}: {next_index}")
                pts_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if not math.isfinite(pts_s) or abs(pts_s - expected_time_s) > pts_tolerance_s:
                    raise RuntimeError(
                        f"Unexpected SLAM {side} presentation time for frame {frame_index}: "
                        f"{pts_s}s instead of {expected_time_s}s ({paths[side]})"
                    )
                pair[f"decoded_pts_{side}_s"] = float(pts_s)
                frames.append(frame)
            if abs(pair["decoded_pts_left_s"] - pair["decoded_pts_right_s"]) > pts_tolerance_s:
                raise RuntimeError(f"SLAM left/right presentation timestamps disagree: {pair}")
            pairs.append(pair)

        counts_match = streams["left"]["frame_count"] == streams["right"]["frame_count"]
        metadata = {
            "version": 1,
            "sampling": "uniform_common_exported_timeline_nearest_frame_including_endpoints",
            "time_domain": "exported_video_seconds",
            "physical_capture_sync_verified": False,
            "capture_timestamp_note": (
                "The SLAM exporter writes camera records independently at a common FPS "
                "without capture timestamp sidecars. Matching MP4 timestamps do not "
                "verify matching device capture timestamps."
            ),
            "frame_count_mismatch": not counts_match,
            "common_frame_count": common_count,
            "common_last_time_s": last_index / fps,
            "num_pairs": num_pairs,
            "num_frames": len(frames),
            "frame_order": [f"slam_{side}_{i + 1}" for i in range(num_pairs) for side in ("left", "right")],
            "streams": streams,
            "pairs": pairs,
        }
        return frames, metadata
    finally:
        for cap in captures.values():
            cap.release()
