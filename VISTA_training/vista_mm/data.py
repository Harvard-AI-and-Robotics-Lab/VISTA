"""Prepared VISTA datasets and multimodal input collation."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import Qwen3VLProcessor

from .config import SensorConfig
from .features import SCHEMA, read_rgb
from .manifest import build_prompt
from .model import SENSOR_TOKEN
from .native_features import NATIVE_SCHEMA, RECIPE
from .provenance import read_jsonl, sha256

def detail_frame_content(frames, selected):
    """Eight overview frames, with four quadrant crops of two fixed keyframes.

    All detail pixels come from the original sampled frames. No extra timestamps,
    reference answers, or question-dependent region selection enter this path.
    """
    content, images = [], []
    for index, frame in enumerate(frames):
        content.extend(
            [{"type": "text", "text": f"Frame{index + 1} overview:"}, {"type": "image"}]
        )
        images.append(frame)
        if index in selected:
            h, w = frame.shape[:2]
            for label, crop in (
                ("upper left", frame[: h // 2, : w // 2]),
                ("upper right", frame[: h // 2, w // 2 :]),
                ("lower left", frame[h // 2 :, : w // 2]),
                ("lower right", frame[h // 2 :, w // 2 :]),
            ):
                content.extend(
                    [
                        {"type": "text", "text": f"Frame{index + 1} {label} detail:"},
                        {"type": "image"},
                    ]
                )
                images.append(np.ascontiguousarray(crop))
    return content, images


class SynchronizedQwenProcessor(Qwen3VLProcessor):
    """Align video-token timestamps with the sampled device timestamps."""

    frame_timestamps = None

    def _calculate_timestamps(self, indices, fps, merge_size):
        if self.frame_timestamps is None:
            return super()._calculate_timestamps(indices, fps, merge_size)
        times = list(map(float, self.frame_timestamps))
        if len(times) != len(indices):
            raise ValueError("Timestamp count differs from sampled frame count")
        if len(times) % merge_size:
            times += [times[-1]] * (merge_size - len(times) % merge_size)
        return [
            float(np.mean(times[i : i + merge_size]))
            for i in range(0, len(times), merge_size)
        ]


def verify_prepared(root, selected_modalities=(), min_coverage=0):
    root = Path(root)
    if (root / "INCOMPLETE.json").exists():
        raise ValueError("Preparation did not finish")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["schema"] not in (SCHEMA, NATIVE_SCHEMA):
        raise ValueError("Feature schema version mismatch")
    for filename, expected in manifest["checksums"].items():
        if sha256(root / filename) != expected:
            raise ValueError(f"Prepared manifest checksum mismatch: {filename}")
    normalization = json.loads((root / "normalization.json").read_text())
    if normalization["schema"] != manifest["schema"]:
        raise ValueError("Normalization and prepared feature schemas differ")
    train_ids = sorted(c["clip_id"] for c in read_jsonl(root / "train.clips.jsonl"))
    if (
        normalization["fit_split"] != "train"
        or normalization["train_clip_ids"] != train_ids
    ):
        raise ValueError(
            "Normalization is not fitted exclusively on the current training clips"
        )
    for name in selected_modalities:
        if manifest["coverage"]["train"][name] < min_coverage:
            raise ValueError(
                f"{name}: insufficient training coverage; do not silently run a missing-modality experiment"
            )
    return manifest, normalization


def sensor_availability(features, modalities, sensor_config, normalization):
    """Clip-level availability using the same masks as the selected frontends."""
    present = []
    for name in modalities:
        native = (
            name == "audio" and sensor_config.get("audio_encoder") == "beats"
        ) or (name == "imu" and sensor_config.get("imu_encoder") == "primus")
        if native:
            mask = features[name + "_native_mask"]
        else:
            key = (
                "metric3d_spatial"
                if name == "metric3d"
                and sensor_config.get("depth_encoder") == "spatial"
                else name
            )
            mask = features[key + "_mask"] & (
                np.asarray(normalization[key]["count"])[None, :] > 0
            )
        present.append(bool(mask.any()))
    return tuple(present)


def matched_sensor_donors(signatures, seed):
    """Derange clips within availability strata, retaining unmatched singletons."""
    groups = {}
    for cid in sorted(signatures):
        groups.setdefault(signatures[cid], []).append(cid)
    rng = np.random.default_rng(seed)
    donors = {}
    for ids in groups.values():
        rng.shuffle(ids)
        donors.update(zip(ids, ids[1:] + ids[:1]))
    if not any(cid != donor for cid, donor in donors.items()):
        raise ValueError("No clips have matching availability for sensor shuffling")
    return donors


class VistaDataset(Dataset):
    def __init__(
        self,
        root,
        split,
        config,
        training=False,
        ablate=(),
        shuffle_sensors=False,
        shuffle_preserve_availability=False,
    ):
        if split not in ("train", "test") or (training and split != "train"):
            raise ValueError("Training must use train; test stays held out")
        self.root, self.config, self.split = Path(root), config, split
        self.manifest, norm = verify_prepared(
            root,
            [
                name
                for name in config.sensor.get("modalities", [])
                if name not in config.allow_missing_modalities
            ],
            config.min_modality_coverage,
        )
        self.norm = norm["modalities"]
        native = []
        for name, encoder in (("audio", "beats"), ("imu", "primus")):
            if (
                name in config.sensor["modalities"]
                and config.sensor.get(name + "_encoder") == encoder
            ):
                native.append(name + "_native")
        if (
            "metric3d" in config.sensor["modalities"]
            and config.sensor.get("depth_encoder") == "spatial"
        ):
            native.append("metric3d_spatial")
        if native:
            if (
                self.manifest["schema"] != NATIVE_SCHEMA
                or self.manifest.get("native_recipe") != RECIPE
            ):
                raise ValueError(
                    "Native encoder inputs require the exact v2 preparation recipe"
                )
            for name in native:
                modality = name.removesuffix("_native").removesuffix("_spatial")
                if (
                    modality not in config.allow_missing_modalities
                    and self.manifest["coverage"]["train"].get(name, 0)
                    < config.min_modality_coverage
                ):
                    raise ValueError(f"Insufficient native training coverage: {name}")
        self.items = read_jsonl(self.root / f"{split}.items.jsonl")
        self.clips = {
            c["clip_id"]: c for c in read_jsonl(self.root / f"{split}.clips.jsonl")
        }
        self.ablate, self.training = set(ablate), training
        if self.ablate - set(config.sensor.get("modalities", [])):
            raise ValueError("Ablation includes a modality absent from the model")
        self.modalities = config.sensor.get("modalities", [])
        self.sensor_donor = {cid: cid for cid in self.clips}
        if shuffle_preserve_availability and not shuffle_sensors:
            raise ValueError("Availability matching requires sensor shuffling")
        if shuffle_sensors:
            # A fixed, within-split derangement; no train/test mixing and no item-level donor leakage.
            ids = sorted(self.clips)
            if len(ids) < 2:
                raise ValueError("Sensor shuffle requires at least two clips")
            if shuffle_preserve_availability:
                signatures = {}
                for cid in ids:
                    clip = self.clips[cid]
                    path = self.root / clip["features"]
                    if sha256(path) != clip["feature_sha256"]:
                        raise ValueError(f"Corrupt feature cache: {cid}")
                    with np.load(path, allow_pickle=False) as features:
                        signatures[cid] = sensor_availability(
                            features, self.modalities, config.sensor, self.norm
                        )
                self.sensor_donor = matched_sensor_donors(signatures, config.seed)
            else:
                rng = np.random.default_rng(config.seed)
                rng.shuffle(ids)
                self.sensor_donor = dict(zip(ids, ids[1:] + ids[:1]))
        self._verified = set()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        clip = self.clips[item["clip_id"]]
        donor = self.clips[self.sensor_donor[item["clip_id"]]]
        for c in (clip, donor):
            if c["clip_id"] not in self._verified:
                if sha256(self.root / c["features"]) != c["feature_sha256"]:
                    raise ValueError(f"Corrupt feature cache: {c['clip_id']}")
                if sha256(c["video"]) != c["video_sha256"]:
                    raise ValueError(
                        f"RGB video changed after preparation: {c['clip_id']}"
                    )
                self._verified.add(c["clip_id"])
        with np.load(self.root / clip["features"], allow_pickle=False) as d:
            frames = read_rgb(
                clip["video"], d["frame_indices"], int(d["rgb_frame_count"])
            )
            frame_times = d["frame_times"].copy()
            frame_indices = d["frame_indices"].copy()
            frame_count = int(d["rgb_frame_count"])
        sensors = {}
        with np.load(self.root / donor["features"], allow_pickle=False) as d:
            relative = d["device_times"] - d["device_times"][0]
            duration = float(relative[-1])
            for name in self.modalities:
                spatial = (
                    name == "metric3d"
                    and self.config.sensor.get("depth_encoder") == "spatial"
                )
                key = "metric3d_spatial" if spatial else name
                x = d[key].copy()
                mask = d[key + "_mask"].copy()
                quality = d[key + "_quality"].copy()
                stats = self.norm[key]
                mask &= np.asarray(stats["count"])[None, :] > 0
                x = (x - np.asarray(stats["mean"])) / np.asarray(stats["std"])
                x = np.where(mask, np.clip(x, -10, 10), 0).astype(np.float32)
                if name in self.ablate:
                    x[:] = 0
                    mask[:] = False
                    quality[:] = 0
                times = (
                    d[key + "_times"].copy()
                    if spatial
                    else np.array([duration / 2], np.float32)
                    if name == "metric3d"
                    else relative.astype(np.float32)
                )
                sensors[name] = {
                    "values": x,
                    "mask": mask,
                    "quality": quality,
                    "times": times,
                    "duration": np.array(duration, np.float32),
                }
                if spatial:
                    sensors[name]["positions"] = d[key + "_positions"].copy()
                if (
                    name == "audio"
                    and self.config.sensor.get("audio_encoder") == "beats"
                ) or (
                    name == "imu" and self.config.sensor.get("imu_encoder") == "primus"
                ):
                    for suffix in ("values", "mask", "times"):
                        sensors[name]["native_" + suffix] = d[
                            name + "_native_" + suffix
                        ].copy()
                    if name in self.ablate:
                        sensors[name]["native_values"][:] = 0
                        sensors[name]["native_mask"][:] = False
        return dict(
            item,
            frames=frames,
            frame_times=frame_times,
            frame_indices=frame_indices,
            frame_count=frame_count,
            sensors=sensors,
        )


class VistaCollator:
    def __init__(self, processor, config, training=True):
        self.processor, self.config, self.training = processor, config, training
        self.tokenizer = processor.tokenizer

    def encode_prompt(self, example):
        sensors = bool(self.config.sensor.get("modalities", []))
        slots = (
            (
                "\n" + SENSOR_TOKEN * SensorConfig(**self.config.sensor).input_tokens
            )
            if sensors
            else ""
        )
        prompt = build_prompt(example)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": prompt + slots},
                ],
            },
        ]
        if self.processor.__class__.__name__ == "Gemma3Processor":
            content = []
            for index, timestamp in enumerate(example["frame_times"]):
                content.extend(
                    [
                        {
                            "type": "text",
                            "text": f"Frame {index + 1} at {float(timestamp):.2f}s:",
                        },
                        {"type": "image"},
                    ]
                )
            content.append(
                {"type": "text", "text": prompt + slots}
            )
            text = self.processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.processor(
                text=[text],
                images=list(example["frames"]),
                return_tensors="pt",
                images_kwargs={"do_pan_and_scan": False},
            )
            if len(inputs["input_ids"][0]) >= self.config.max_length:
                raise ValueError(
                    f"{example['item_id']}: Gemma prompt exceeds max_length"
                )
            return inputs
        detail_images = None
        if self.config.internvl_detail_frames:
            if self.processor.__class__.__name__ != "InternVLProcessor":
                raise ValueError(
                    "Detail-frame inputs currently require native InternVL"
                )
            content, detail_images = detail_frame_content(
                example["frames"], self.config.internvl_detail_frames
            )
            messages[0]["content"] = content + [
                {"type": "text", "text": prompt + slots}
            ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if self.processor.__class__.__name__ == "InternVLProcessor":
            # Official native video processor: preserve the same eight sampled frames,
            # using its model-specific resolution and Frame1...Frame8 placeholders.
            if detail_images is not None:
                inputs = self.processor(
                    text=[text],
                    images=detail_images,
                    return_tensors="pt",
                    images_kwargs={"crop_to_patches": False},
                )
            else:
                inputs = self.processor(
                    text=[text],
                    videos=[example["frames"]],
                    return_tensors="pt",
                    videos_kwargs={
                        "do_sample_frames": False,
                        "size": dict(self.processor.image_processor.size),
                    },
                )
            inputs.pop("token_type_ids", None)
            inputs.pop("mm_token_type_ids", None)
            if len(inputs["input_ids"][0]) >= self.config.max_length:
                raise ValueError(
                    f"{example['item_id']}: InternVL prompt exceeds max_length"
                )
            return inputs
        # Preserve original sampled frame indices and their physical timestamps. Qwen's
        # metadata fps/index conversion must not re-interpret eight frames as eight adjacent frames.
        duration = float(example["frame_times"][-1])
        fps = (example["frame_count"] - 1) / max(duration, 1e-6)
        metadata = {
            "fps": fps,
            "total_num_frames": example["frame_count"],
            "frames_indices": example["frame_indices"].tolist(),
            "duration": duration,
        }
        self.processor.frame_timestamps = example["frame_times"]
        try:
            inputs = self.processor(
                text=[text],
                videos=[example["frames"]],
                return_tensors="pt",
                videos_kwargs={
                    "do_sample_frames": False,
                    "video_metadata": [metadata],
                    "size": {
                        "shortest_edge": self.config.min_pixels,
                        "longest_edge": self.config.max_pixels,
                    },
                },
            )
        finally:
            self.processor.frame_timestamps = None
        inputs.pop("token_type_ids", None)
        inputs.pop("mm_token_type_ids", None)
        # Make sure every processor-generated video placeholder reaches the model unchanged.
        if len(inputs["input_ids"][0]) >= self.config.max_length:
            raise ValueError(
                f"{example['item_id']}: visual/prompt tokens exceed max_length; lower pixel budget explicitly"
            )
        return inputs

    def __call__(self, examples):
        encoded = [self.encode_prompt(e) for e in examples]
        seqs, labels = [], []
        for e, inputs in zip(examples, encoded):
            prefix = inputs["input_ids"][0]
            if self.training:
                # Deterministic uniform reference choice under the seeded DataLoader RNG.
                ref = (
                    e["references"][torch.randint(len(e["references"]), ()).item()]
                    if e["split"] == "train"
                    else e["references"][0]
                )
                target = self.tokenizer(ref, add_special_tokens=False)["input_ids"] + [
                    (
                        self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
                        if self.processor.__class__.__name__ == "Gemma3Processor"
                        else self.tokenizer.eos_token_id
                    )
                ]
                if len(prefix) + len(target) > self.config.max_length:
                    raise ValueError(
                        f"{e['item_id']}: answer would be truncated; increase max_length explicitly"
                    )
                target = torch.tensor(target, dtype=torch.long)
                seqs.append(torch.cat([prefix, target]))
                labels.append(torch.cat([torch.full_like(prefix, -100), target]))
            else:
                seqs.append(prefix)
        max_len = max(len(s) for s in seqs)
        padding_side = "right" if self.training else "left"
        batch = {
            "input_ids": torch.full(
                (len(seqs), max_len), self.tokenizer.pad_token_id, dtype=torch.long
            ),
            "attention_mask": torch.zeros((len(seqs), max_len), dtype=torch.long),
        }
        if self.training:
            batch["labels"] = torch.full((len(seqs), max_len), -100, dtype=torch.long)
        for i, seq in enumerate(seqs):
            start = 0 if padding_side == "right" else max_len - len(seq)
            batch["input_ids"][i, start : start + len(seq)] = seq
            batch["attention_mask"][i, start : start + len(seq)] = 1
            if self.training:
                batch["labels"][i, start : start + len(seq)] = labels[i]
        if "token_type_ids" in encoded[0]:
            batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
            for i, (seq, inputs) in enumerate(zip(seqs, encoded)):
                start = 0 if padding_side == "right" else max_len - len(seq)
                types = inputs["token_type_ids"][0]
                batch["token_type_ids"][i, start : start + len(types)] = types
        for key in (
            "pixel_values_videos",
            "video_grid_thw",
            "pixel_values",
            "image_grid_thw",
        ):
            if key in encoded[0]:
                batch[key] = torch.cat([e[key] for e in encoded], 0)
        if self.config.sensor.get("modalities", []):
            question = self.tokenizer(
                [e["question"] for e in examples],
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            if question["input_ids"].shape[1] > self.config.max_question_length:
                raise ValueError(
                    "Question exceeds max_question_length; do not silently truncate gate conditioning"
                )
            batch["question_input_ids"] = question["input_ids"]
            batch["question_attention_mask"] = question["attention_mask"]
            batch["sensors"] = {
                m: {
                    k: torch.from_numpy(
                        np.stack([e["sensors"][m][k] for e in examples])
                    )
                    for k in examples[0]["sensors"][m]
                }
                for m in examples[0]["sensors"]
            }
        return batch
