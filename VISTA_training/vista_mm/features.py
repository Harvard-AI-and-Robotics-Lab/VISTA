"""Versioned features from actual VISTA CSV schemas; all times use device seconds.

Values, per-channel validity and per-time quality travel together. Never interpolate
across an unobserved gap or infer sensor timestamps from video duration alone.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from .config import DIMS

SCHEMA = "vista-sensors-v1"
STEPS = 64


class SensorDataError(ValueError):
    """A known sensor input inconsistency that may be explicitly masked."""


def read_table(path, timestamp, scale):
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if timestamp not in df:
        raise ValueError(f"Missing {timestamp} in {path}")
    t = pd.to_numeric(df[timestamp], errors="coerce").to_numpy(np.float64) * scale
    valid = np.isfinite(t)
    df, t = df.loc[valid].reset_index(drop=True), t[valid]
    order = np.argsort(t, kind="stable")
    df, t = df.iloc[order].reset_index(drop=True), t[order]
    unique = np.r_[True, np.diff(t) > 0] if len(t) else np.zeros(0, bool)
    return t[unique], df.loc[unique].reset_index(drop=True)


def columns(df, names):
    missing = set(names) - set(df)
    if missing:
        raise ValueError(f"Feature schema mismatch, missing columns: {sorted(missing)}")
    return df[names].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)


def blank(dim, length=STEPS):
    return (
        np.zeros((length, dim), np.float32),
        np.zeros((length, dim), bool),
        np.zeros(length, np.float32),
    )


def align(t, x, grid, max_gap, mask=None, quality=None):
    """Bin-average dense samples; nearest observation for sparse bins, within tolerance.

    Binning provides low-pass averaging before downsampling. Per-channel masks
    distinguish untracked hands from valid zero coordinates.
    """
    if len(t) != len(x):
        raise ValueError("Timestamp/value length mismatch")
    out, valid, qout = blank(x.shape[1], len(grid))
    if not len(t):
        return out, valid, qout
    mask = np.isfinite(x) if mask is None else (mask & np.isfinite(x))
    quality = np.ones(len(t)) if quality is None else np.nan_to_num(quality).clip(0, 1)
    edges = np.r_[grid[0], (grid[1:] + grid[:-1]) / 2, grid[-1]]
    for i, center in enumerate(grid):
        ids = np.flatnonzero(
            (t >= edges[i]) & (t <= edges[i + 1]) & (np.abs(t - center) <= max_gap)
        )
        if not len(ids):
            nearest = int(np.argmin(np.abs(t - center)))
            if abs(t[nearest] - center) > max_gap or center < t[0] or center > t[-1]:
                continue
            ids = np.array([nearest])
        w = mask[ids] * quality[ids, None]
        denom = w.sum(0)
        valid[i] = denom > 0
        out[i] = (np.where(mask[ids], x[ids], 0) * w).sum(0) / np.maximum(denom, 1e-12)
        qout[i] = float(quality[ids].mean()) if valid[i].any() else 0
    return out, valid, qout


def rgb_clock(directory, video=None, allow_trailing_timestamp=False):
    p = Path(directory) / "vrs/rgb_timestamps.csv"
    df = pd.read_csv(p)
    indices = df["frame_index"].to_numpy(np.int64)
    t = df["capture_timestamp_ns"].to_numpy(np.float64) * 1e-9
    if len(t) < 8 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError(f"Invalid RGB clock: {p}")
    if not np.array_equal(indices, np.arange(len(indices))):
        raise ValueError(f"RGB frame indices are not contiguous: {p}")
    if video is not None:
        import cv2

        cap = cv2.VideoCapture(str(video))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if count != len(t):
            if allow_trailing_timestamp and len(t) == count + 1 and count >= 8:
                # CSV frame_index identifies exported frames; the final index has
                # no decoded frame. Retain the observed prefix without new times.
                t = t[:count]
            else:
                raise ValueError(
                    f"Video/clock frame mismatch for {video}: {count} != {len(t)}"
                )
    sample_ids = np.rint(np.linspace(0, len(t) - 1, 8)).astype(np.int64)
    return t, sample_ids


def read_rgb(video, indices, expected_count):
    import cv2

    indices = [int(i) for i in indices]
    if not indices or min(indices) < 0 or max(indices) >= expected_count:
        raise ValueError(f"Invalid requested frame indices for {video}")
    cap = cv2.VideoCapture(str(video))
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count != expected_count:
            raise ValueError(
                f"Video/clock frame mismatch for {video}: {count} != {expected_count}"
            )
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if len(frames) == len(indices):
            return np.stack(frames)
    finally:
        cap.release()

    # Some H.264 streams decode correctly but cannot seek to their last frame.
    # Decode from the start and retrieve the exact original indices; never
    # substitute neighboring frames or change the prepared timestamps.
    cap = cv2.VideoCapture(str(video))
    wanted = set(indices)
    selected = {}
    try:
        for idx in range(max(indices) + 1):
            if not cap.grab():
                raise ValueError(f"Cannot sequentially decode {video} frame {idx}")
            if idx in wanted:
                ok, frame = cap.retrieve()
                if not ok:
                    raise ValueError(f"Cannot retrieve {video} frame {idx}")
                selected[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return np.stack([selected[idx] for idx in indices])
    finally:
        cap.release()


def imu(directory, grid):
    values, masks, qualities = [], [], []
    for side in ("left", "right"):
        table = read_table(
            directory / f"vrs/imu_{side}.csv", "capture_timestamp_ns", 1e-9
        )
        if table is None:
            x, mask, quality = blank(8)
        else:
            t, df = table
            x = columns(
                df, [f"{kind}_{axis}" for kind in ("accel", "gyro") for axis in "xyz"]
            )
            x = np.c_[
                x, np.linalg.norm(x[:, :3], axis=1), np.linalg.norm(x[:, 3:], axis=1)
            ]
            x, mask, quality = align(t, x, grid, 0.05)
        values.append(x)
        masks.append(mask)
        qualities.append(quality)
    return np.concatenate(values, 1), np.concatenate(masks, 1), np.mean(qualities, 0)


def magbaro(directory, grid):
    mag = read_table(directory / "vrs/magnetometer.csv", "capture_timestamp_ns", 1e-9)
    bar = read_table(directory / "vrs/barometer.csv", "capture_timestamp_ns", 1e-9)
    a, am, aq = blank(4)
    b, bm, bq = blank(4)
    if mag is not None:
        t, df = mag
        x = columns(df, ["mag_x", "mag_y", "mag_z"])
        x = np.c_[x, np.linalg.norm(x, axis=1)]
        ok = df["valid"].astype(str).str.lower().isin(["true", "1"]).to_numpy()
        a, am, aq = align(t, x, grid, 0.15, np.broadcast_to(ok[:, None], x.shape))
    if bar is not None:
        t, df = bar
        x = columns(df, ["pressure", "temperature"])
        if len(t):
            dp = np.gradient(x[:, 0], t) if len(t) > 1 else np.zeros(len(t))
            x = np.c_[x, x[:, 0] - x[0, 0], dp]
            b, bm, bq = align(t, x, grid, 0.15)
    return np.c_[a, b], np.c_[am, bm], (aq + bq) / 2


def gaze(directory, grid):
    table = read_table(
        directory / "mps/eye_gaze/general_eye_gaze.csv", "tracking_timestamp_us", 1e-6
    )
    if table is None:
        return blank(10)
    t, df = table
    angles = columns(
        df, ["left_yaw_rads_cpf", "right_yaw_rads_cpf", "pitch_rads_cpf", "depth_m"]
    )
    center = (
        columns(df, [f"t{a}_left_eye_cpf" for a in "xyz"])
        + columns(df, [f"t{a}_right_eye_cpf" for a in "xyz"])
    ) / 2
    yaw, pitch = angles[:, :2].mean(1), angles[:, 2]
    direction = np.c_[np.tan(yaw), np.tan(pitch), np.ones(len(t))]
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
    x = np.c_[angles, center, direction]
    good = np.isfinite(angles[:, :3]).all(1) & np.isfinite(center).all(1)
    bounds = columns(
        df,
        [
            "left_yaw_low_rads_cpf",
            "left_yaw_high_rads_cpf",
            "right_yaw_low_rads_cpf",
            "right_yaw_high_rads_cpf",
            "pitch_low_rads_cpf",
            "pitch_high_rads_cpf",
        ],
    )
    uncertainty = np.abs(bounds[:, 1::2] - bounds[:, ::2]).mean(1)
    quality = np.exp(-uncertainty) * good
    mask = np.broadcast_to(good[:, None], x.shape).copy()
    mask[:, 3] &= angles[:, 3] > 0  # Missing depth does not invalidate gaze direction.
    return align(t, x, grid, 0.15, mask, quality)


def rotations(q):
    valid = np.isfinite(q).all(1) & (np.linalg.norm(q, axis=1) > 0.5)
    safe = np.where(valid[:, None], q, [0.0, 0.0, 0.0, 1.0])
    r = Rotation.from_quat(safe)
    return r, valid


def hands(directory, grid):
    table = read_table(
        directory / "mps/hand_tracking/hand_tracking_results.csv",
        "tracking_timestamp_us",
        1e-6,
    )
    if table is None:
        return blank(168)
    t, df = table
    if not len(t):
        return blank(168)
    xx, mm, qq = [], [], []
    for side in ("left", "right"):
        landmarks = columns(
            df, [f"t{a}_{side}_landmark_{j}_device" for j in range(21) for a in "xyz"]
        )
        wrist = columns(df, [f"t{a}_{side}_device_wrist" for a in "xyz"])
        rot, rv = rotations(columns(df, [f"q{a}_{side}_device_wrist" for a in "xyzw"]))
        rot6 = rot.as_matrix()[:, :, :2].reshape(-1, 6)
        normals = columns(
            df,
            [f"n{a}_{side}_{part}_device" for part in ("palm", "wrist") for a in "xyz"],
        )
        confidence = columns(df, [f"{side}_tracking_confidence"])[:, 0]
        good = (
            (confidence > 0)
            & rv
            & np.isfinite(landmarks).all(1)
            & np.isfinite(wrist).all(1)
        )
        velocity, omega = np.zeros_like(wrist), np.zeros_like(wrist)
        derivative_valid = np.zeros(len(t), bool)
        for i in range(1, len(t)):
            dt = t[i] - t[i - 1]
            if good[i] and good[i - 1] and 0 < dt <= 0.15:
                velocity[i] = (wrist[i] - wrist[i - 1]) / dt
                omega[i] = (rot[i - 1].inv() * rot[i]).as_rotvec() / dt
                derivative_valid[i] = True
        # 63 landmark + 3 wrist position + 6 rotation + 6 normals + 6 derivatives = 84/hand.
        x = np.c_[landmarks, wrist, rot6, normals, velocity, omega]
        mask = np.broadcast_to(good[:, None], x.shape).copy()
        mask[:, -6:] &= derivative_valid[:, None]
        xx.append(x)
        mm.append(mask)
        qq.append(np.where(good, confidence.clip(0, 1), 0))
    return align(
        t,
        np.c_[xx[0], xx[1]],
        grid,
        0.15,
        np.c_[mm[0], mm[1]],
        np.maximum(qq[0], qq[1]),
    )


def trajectory(directory, grid):
    table = read_table(
        directory / "mps/slam/closed_loop_trajectory.csv", "tracking_timestamp_us", 1e-6
    )
    if table is None:
        return blank(19)
    t, df = table
    if not len(t):
        return blank(19)
    # Do not use UTC, ECEF, graph/session IDs or absolute world origin as model features.
    position = columns(df, [f"t{a}_world_device" for a in "xyz"])
    rot, good = rotations(columns(df, [f"q{a}_world_device" for a in "xyzw"]))
    quality = columns(df, ["quality_score"])[:, 0]
    good &= np.isfinite(position).all(1) & (quality > 0)
    visible = good & (t >= grid[0]) & (t <= grid[-1])
    if not visible.any():
        return blank(19)
    first = np.flatnonzero(visible)[0]
    r0 = rot[first].inv()
    relative_position = r0.apply(position - position[first])
    relative_rot6 = (r0 * rot).as_matrix()[:, :, :2].reshape(-1, 6)
    velocity = columns(df, [f"device_linear_velocity_{a}_device" for a in "xyz"])
    omega = columns(df, [f"angular_velocity_{a}_device" for a in "xyz"])
    gravity = r0.apply(columns(df, [f"gravity_{a}_world" for a in "xyz"]))
    x = np.c_[relative_position, relative_rot6, velocity, omega, gravity, quality]
    return align(t, x, grid, 0.05, np.broadcast_to(good[:, None], x.shape), quality)


def audio(directory, grid, timestamp_reference):
    """64 log-mel channels from ambient VRS audio, never spoken annotation WAVs.

    Record timestamps may denote a block's start or end; the choice is an explicit,
    provenance-recorded preparation argument. Discontinuous audio blocks stay masked.
    """
    import soundfile as sf
    import torch
    import torchaudio

    path = directory / "vrs/audio.wav"
    if not all(
        (directory / name).is_file()
        for name in ("vrs/audio.wav", "vrs/audio.wav.json", "vrs/audio_timestamps.csv")
    ):
        return blank(64)
    sidecar = json.loads((directory / "vrs/audio.wav.json").read_text())
    stamps = pd.read_csv(directory / "vrs/audio_timestamps.csv")[
        "timestamp_s"
    ].to_numpy(np.float64)
    wave, sr = sf.read(path, dtype="float32", always_2d=True)
    if not len(stamps) or sr != sidecar["audio_sample_rate"]:
        raise SensorDataError(f"Invalid audio metadata: {path}")
    block = int(round(sidecar["first_audio_record_duration"] * sr))
    if block <= 0 or len(wave) != block * len(stamps):
        raise SensorDataError(f"Cannot map audio samples to device clock: {path}")
    if (
        wave.shape[1] != 7
        or not np.isfinite(stamps).all()
        or np.any(np.diff(stamps) <= 0)
    ):
        raise SensorDataError(f"Invalid audio channels or record clock: {path}")
    if timestamp_reference not in ("start", "end"):
        raise ValueError(
            "Set --audio-timestamp-reference start or end from the exporter convention"
        )
    starts = stamps - (block / sr if timestamp_reference == "end" else 0)
    # Power-average channels after STFT: waveform averaging would cause phase cancellation.
    signal = torch.from_numpy(wave.T.copy())
    if sr != 16000:
        signal = torchaudio.functional.resample(signal, sr, 16000)
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        n_mels=64,
        f_min=20,
        f_max=7600,
        center=False,
    )
    power = transform(signal).mean(0)
    x = torch.log(power.clamp_min(1e-10)).T.numpy()
    sample_center = (np.arange(len(x)) * 160 + 200) / 16000 * sr
    bi = np.minimum((sample_center / block).astype(int), len(starts) - 1)
    t = starts[bi] + (sample_center - bi * block) / sr
    half_window = 0.0125
    gaps = np.flatnonzero(np.abs(starts[1:] - starts[:-1] - block / sr) > 2 / sr)
    good = np.ones(len(t), bool)
    for g in gaps:
        boundary_sample = (g + 1) * block
        good &= np.abs(sample_center - boundary_sample) > half_window * sr
    return align(t, x, grid, 0.08, np.broadcast_to(good[:, None], x.shape))


def extract(
    clip,
    depth_root=None,
    audio_timestamp_reference=None,
    enabled=None,
    allow_trailing_timestamp=False,
    mask_invalid_modalities=(),
):
    directory = Path(clip["directory"])
    times, frame_indices = rgb_clock(directory, clip["video"], allow_trailing_timestamp)
    grid = np.linspace(times[0], times[-1], STEPS)
    data = {
        "frame_indices": frame_indices,
        "frame_times": (times[frame_indices] - times[0]).astype(np.float32),
        "device_times": grid,
        "rgb_frame_count": np.array(len(times)),
    }
    enabled = list(DIMS) if enabled is None else enabled
    methods = {
        "imu": imu,
        "magbaro": magbaro,
        "gaze": gaze,
        "hands": hands,
        "trajectory": trajectory,
    }
    for name, dim in DIMS.items():
        if name not in enabled:
            x, mask, quality = blank(dim, 1 if name == "metric3d" else STEPS)
        elif name in methods:
            x, mask, quality = methods[name](directory, grid)
        elif name == "audio":
            try:
                x, mask, quality = audio(directory, grid, audio_timestamp_reference)
            except SensorDataError as exc:
                if name not in mask_invalid_modalities:
                    raise
                x, mask, quality = blank(dim)
                data["audio_issue"] = np.array(str(exc))
        else:
            p = (
                Path(depth_root) / clip["split"] / f"{clip['clip_id']}.npz"
                if depth_root
                else None
            )
            if p is None or not p.exists():
                x, mask, quality = blank(dim, 1)
            else:
                with np.load(p, allow_pickle=False) as d:
                    if str(d["video_sha256"]) != clip["video_sha256"]:
                        raise ValueError(f"Stale Metric3D features: {p}")
                    if not np.array_equal(d["frame_indices"], frame_indices):
                        raise ValueError(f"Metric3D/RGB sampling mismatch: {p}")
                    x, mask, quality = d["values"], d["mask"], d["quality"]
        if x.shape != mask.shape or x.shape[-1] != dim or not np.isfinite(x).all():
            raise ValueError(f"Invalid {name} features for {clip['clip_id']}")
        data[name] = np.where(mask, x, 0).astype(np.float32)
        data[name + "_mask"] = mask.astype(bool)
        data[name + "_quality"] = np.asarray(quality, np.float32)
    return data
