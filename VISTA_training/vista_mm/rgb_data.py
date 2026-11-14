"""RGB-only preparation and datasets, without sensor features or encoder assets."""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from .provenance import atomic_json, digest, read_jsonl, sha256

RGB_SCHEMA = "vista-rgb-v1"
SPLITS = ("train", "val", "test")


def annotation_items(csv_path, split):
    """One example per task, retaining all numbered columns and scene references."""
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if "clip_id" not in columns or len(columns) != len(set(columns)):
            raise ValueError(f"Missing clip_id or duplicate CSV columns: {csv_path}")
        indices = {
            prefix: sorted(int(m.group(1)) for c in columns
                           if (m := re.fullmatch(prefix + r"_([1-9]\d*)", c)))
            for prefix in ("scene_description", "question", "goal")
        }
        items, clip_ids = [], []
        for row in reader:
            cid = (row.get("clip_id") or "").strip()
            if not re.fullmatch(r"TC\d+_\d+", cid) or cid in clip_ids:
                raise ValueError(f"Invalid or duplicate clip_id: {cid!r}")
            clip_ids.append(cid)
            def value(column):
                return (row.get(column) or "").strip()
            def append(task, question, references):
                items.append(dict(item_id=f"{cid}|{task}", clip_id=cid, split=split,
                                  task_type=task, question=question, references=references))
            refs = [value(f"scene_description_{i}") for i in indices["scene_description"]]
            refs = [ref for ref in refs if ref]
            if refs:
                append("scene_description", "Describe the scene visible in the video.", refs)
            for prefix, answer in (("question", "answer"), ("goal", "instruction")):
                # Also catch orphaned answers/instructions instead of silently losing them.
                numbers = set(indices[prefix]) | {
                    int(m.group(1)) for c in columns
                    if (m := re.fullmatch(answer + r"_([1-9]\d*)", c))
                }
                for i in sorted(numbers):
                    question, target = value(f"{prefix}_{i}"), value(f"{answer}_{i}")
                    if bool(question) != bool(target):
                        raise ValueError(f"Unpaired annotation: {cid}/{prefix}_{i}")
                    if question:
                        append(f"{prefix}_{i}", question, [target])
            if not items or items[-1]["clip_id"] != cid:
                raise ValueError(f"Clip has no complete annotations: {cid}")
    if not clip_ids:
        raise ValueError(f"Empty annotation CSV: {csv_path}")
    return clip_ids, items


def video_metadata(video, allow_trailing_timestamp=False):
    import cv2
    from .features import rgb_clock

    video = Path(video)
    if not video.is_file():
        raise FileNotFoundError(video)
    cap = cv2.VideoCapture(str(video))
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    if count < 8:
        raise ValueError(f"Video must contain at least eight frames: {video}")
    clock = video.parent / "vrs/rgb_timestamps.csv"
    if clock.is_file():
        times, indices = rgb_clock(video.parent, video, allow_trailing_timestamp)
        source = "device_timestamps"
    else:
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"No usable RGB timestamps or video FPS: {video}")
        times = np.arange(count, dtype=np.float64) / fps
        indices = np.rint(np.linspace(0, count - 1, 8)).astype(np.int64)
        source = "video_fps"
    return dict(video=str(video.resolve()), video_sha256=sha256(video),
                frame_count=count, frame_indices=indices.tolist(),
                frame_times=(times[indices] - times[0]).astype(np.float32).tolist(),
                timestamp_source=source,
                timestamp_sha256=sha256(clock) if clock.is_file() else None)


def _match_mm(items, clips, root, split):
    """Match tasks, references, order and visual inputs before claiming a paired run."""
    mm_items = read_jsonl(Path(root) / f"{split}.items.jsonl")
    ours = {item["item_id"]: item for item in items}
    if len(mm_items) != len(ours) or {i["item_id"] for i in mm_items} != set(ours):
        raise ValueError(f"RGB and multimodal task sets differ for {split}")
    for item in mm_items:
        wanted = ours[item["item_id"]]
        for key in ("clip_id", "task_type", "references", "question"):
            actual = item[key]
            if key == "question" and item["task_type"].startswith("goal_"):
                actual = actual.removeprefix("Give navigation guidance for this goal: ")
            if actual != wanted[key]:
                raise ValueError(f"RGB/MM annotation differs: {item['item_id']}/{key}")
    mm_clips = {c["clip_id"]: c for c in read_jsonl(Path(root) / f"{split}.clips.jsonl")}
    if set(mm_clips) != {c["clip_id"] for c in clips}:
        raise ValueError(f"RGB and multimodal clip sets differ for {split}")
    for clip in clips:
        mm = mm_clips[clip["clip_id"]]
        if mm["video_sha256"] != clip["video_sha256"]:
            raise ValueError(f"RGB/MM video differs: {clip['clip_id']}")
        feature = Path(root) / mm["features"]
        if sha256(feature) != mm["feature_sha256"]:
            raise ValueError(f"Corrupt multimodal feature file: {feature}")
        with np.load(feature, allow_pickle=False) as data:
            if (not np.array_equal(data["frame_indices"], clip["frame_indices"])
                    or not np.allclose(data["frame_times"], clip["frame_times"], atol=1e-6, rtol=0)):
                raise ValueError(f"RGB/MM frame selection or timing differs: {clip['clip_id']}")
    return [ours[item["item_id"]] for item in mm_items]


def prepare_rgb(dataset_root, output, compare_mm_prepared=None, allow_trailing_timestamp=False,
                splits=SPLITS):
    dataset_root, output = Path(dataset_root).resolve(), Path(output).resolve()
    splits = tuple(splits)
    if not splits or len(set(splits)) != len(splits) or set(splits) - set(SPLITS):
        raise ValueError("Select distinct RGB splits from train, val, test")
    if output.exists():
        raise FileExistsError(f"Use a new preparation directory: {output}")
    # Check annotation identity and split separation before writing anything.
    annotations = {s: annotation_items(dataset_root / f"{s}.csv", s) for s in splits}
    seen = set()
    for split, (ids, _) in annotations.items():
        if seen.intersection(ids):
            raise ValueError(f"Clip overlap between splits: {split}")
        seen.update(ids)
    output.mkdir(parents=True)
    atomic_json(output / "INCOMPLETE.json", {"schema": RGB_SCHEMA})
    manifest = dict(schema=RGB_SCHEMA, dataset_root=str(dataset_root), num_frames=8,
                    splits=list(splits),
                    annotation_sha256={}, checksums={}, counts={}, timestamp_sources={},
                    compared_mm_prepared=str(Path(compare_mm_prepared).resolve())
                    if compare_mm_prepared else None)
    for split, (ids, items) in annotations.items():
        clips = []
        for cid in ids:
            video = dataset_root / split / cid.split("_")[0] / cid / f"{cid}.mp4"
            clip = dict(clip_id=cid, **video_metadata(video, allow_trailing_timestamp))
            clips.append(clip)
            source = clip["timestamp_source"]
            manifest["timestamp_sources"][source] = manifest["timestamp_sources"].get(source, 0) + 1
        if compare_mm_prepared and split in ("train", "test"):
            items = _match_mm(items, clips, compare_mm_prepared, split)
        for kind, records in (("clips", clips), ("items", items)):
            filename = f"{split}.{kind}.jsonl"
            with (output / filename).open("w", encoding="utf-8", newline="\n") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")
            manifest["checksums"][filename] = sha256(output / filename)
        manifest["annotation_sha256"][split] = sha256(dataset_root / f"{split}.csv")
        manifest["counts"][split] = dict(clips=len(clips), items=len(items))
    manifest["prepared_id"] = digest(manifest)
    atomic_json(output / "manifest.json", manifest)
    (output / "INCOMPLETE.json").unlink()
    return manifest


def verify_rgb(root, selected_modalities=()):
    root = Path(root)
    if selected_modalities:
        raise ValueError("An RGB preparation cannot provide sensor modalities")
    if (root / "INCOMPLETE.json").exists():
        raise ValueError("RGB preparation did not finish")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    identity = {k: v for k, v in manifest.items() if k != "prepared_id"}
    if manifest.get("schema") != RGB_SCHEMA or manifest.get("prepared_id") != digest(identity):
        raise ValueError("Invalid RGB preparation schema or identity")
    # Manifests written before explicit split selection always contained all three.
    splits = manifest.get("splits", list(SPLITS))
    if (not splits or len(set(splits)) != len(splits) or set(splits) - set(SPLITS)
            or set(manifest["counts"]) != set(splits)
            or set(manifest["annotation_sha256"]) != set(splits)):
        raise ValueError("Invalid RGB preparation splits")
    expected = {f"{s}.{k}.jsonl" for s in splits for k in ("clips", "items")}
    if set(manifest["checksums"]) != expected or manifest["num_frames"] != 8:
        raise ValueError("Incomplete RGB preparation manifest")
    for filename, checksum in manifest["checksums"].items():
        if sha256(root / filename) != checksum:
            raise ValueError(f"RGB preparation checksum mismatch: {filename}")
    return manifest


class RGBDataset:
    def __init__(self, root, split, config, training=False, **sensor_options):
        if split not in SPLITS or (training and split != "train"):
            raise ValueError("Only the train split may be used for training")
        if config.input_mode != "rgb" or config.sensor["modalities"]:
            raise ValueError("RGBDataset requires input_mode=rgb without sensors")
        if any(sensor_options.values()):
            raise ValueError("Sensor ablations and shuffling do not apply to RGB runs")
        self.root, self.config, self.split = Path(root), config, split
        self.manifest = verify_rgb(root)
        if split not in self.manifest["counts"]:
            raise ValueError(f"RGB split was not prepared: {split}")
        self.items = read_jsonl(self.root / f"{split}.items.jsonl")
        self.clips = {c["clip_id"]: c for c in read_jsonl(self.root / f"{split}.clips.jsonl")}
        self._verified = set()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        from .features import read_rgb
        item = self.items[index]
        clip = self.clips[item["clip_id"]]
        if clip["clip_id"] not in self._verified:
            if sha256(clip["video"]) != clip["video_sha256"]:
                raise ValueError(f"RGB video changed after preparation: {clip['clip_id']}")
            self._verified.add(clip["clip_id"])
        indices = np.asarray(clip["frame_indices"], dtype=np.int64)
        return dict(item, frames=read_rgb(clip["video"], indices, clip["frame_count"]),
                    frame_times=np.asarray(clip["frame_times"], dtype=np.float32),
                    frame_indices=indices, frame_count=clip["frame_count"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compare-mm-prepared")
    parser.add_argument("--allow-trailing-timestamp", action="store_true")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=SPLITS)
    args = parser.parse_args()
    manifest = prepare_rgb(**vars(args))
    print(json.dumps({"output": str(Path(args.output).resolve()), "counts": manifest["counts"],
                      "timestamp_sources": manifest["timestamp_sources"]}, indent=2))


if __name__ == "__main__":
    main()
