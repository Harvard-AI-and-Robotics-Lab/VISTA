"""Lossless on-disk InternVL RGB pixels; prompts and reference sampling stay live."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
import inspect
import json
import multiprocessing
from pathlib import Path

from .provenance import atomic_json, digest, read_jsonl, sha256

SCHEMA = "vista-internvl-rgb-pixels-v1"
SHAPE = (8, 3, 448, 448)


def recipe():
    from .data import internvl_rgb_pixel_values
    from .rgb_data import read_zero_shot_frames

    return dict(
        shape=list(SHAPE), dtype="float32",
        transform=digest(inspect.getsource(internvl_rgb_pixel_values)),
        decoder=digest(inspect.getsource(read_zero_shot_frames)),
        packages={name: version(name) for name in
                  ("torch", "torchvision", "numpy", "Pillow", "opencv-python")},
    )


def clip_key(clip):
    return digest({name: clip[name] for name in
                   ("video_sha256", "frame_indices", "frame_count")})


def _write_clip(job):
    import torch
    from safetensors.torch import load_file, save_file
    from .data import internvl_rgb_pixel_values
    from .rgb_data import read_zero_shot_frames

    output, split, clip = job
    if sha256(clip["video"]) != clip["video_sha256"]:
        raise ValueError(f"RGB video changed after preparation: {clip['clip_id']}")
    frames = read_zero_shot_frames(clip["video"], clip["frame_indices"])
    pixels = internvl_rgb_pixel_values(frames)
    if pixels.dtype != torch.float32 or tuple(pixels.shape) != SHAPE or not torch.isfinite(pixels).all():
        raise ValueError(f"Invalid RGB pixels: {clip['clip_id']}")
    relative = f"{split}/{clip['clip_id']}.safetensors"
    target = Path(output) / relative
    temporary = target.with_suffix(".pending")
    save_file({"pixel_values": pixels.contiguous()}, temporary)
    if not torch.equal(pixels, load_file(temporary)["pixel_values"]):
        raise ValueError(f"Pixel cache round-trip changed values: {clip['clip_id']}")
    temporary.replace(target)
    return split, clip["clip_id"], dict(file=relative, key=clip_key(clip),
                                      sha256=sha256(target), bytes=target.stat().st_size)


def _worker_init():
    import cv2
    import torch
    # Cache preparation only; do not change training thread/worker settings.
    torch.set_num_threads(1)
    cv2.setNumThreads(1)


def build_cache(prepared, output, splits=("train",), workers=1):
    from .rgb_data import verify_rgb

    if workers < 1:
        raise ValueError("Cache preparation needs at least one worker")
    prepared, output = Path(prepared), Path(output).resolve()
    source = verify_rgb(prepared)
    splits = list(splits)
    if not splits or len(splits) != len(set(splits)) or set(splits) - set(source["counts"]):
        raise ValueError("Select distinct splits present in the RGB preparation")
    jobs = [(str(output), split, clip) for split in splits
            for clip in read_jsonl(prepared / f"{split}.clips.jsonl")]
    manifest = dict(schema=SCHEMA, prepared_id=source["prepared_id"], recipe=recipe(),
                    splits=splits, clips={split: {} for split in splits})
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "INCOMPLETE.json", dict(schema=SCHEMA, clips=len(jobs)))
    for split in splits:
        (output / split).mkdir()

    def record(results):
        for done, (split, cid, entry) in enumerate(results, 1):
            manifest["clips"][split][cid] = entry
            if done == 1 or done % 10 == 0 or done == len(jobs):
                progress = dict(completed=done, total=len(jobs), clip_id=cid)
                atomic_json(output / "progress.json", progress)
                print(json.dumps(progress), flush=True)

    if workers == 1:
        record(map(_write_clip, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_worker_init) as pool:
            record(pool.map(_write_clip, jobs, chunksize=1))
    manifest["cache_id"] = digest(manifest)
    atomic_json(output / "manifest.json", manifest)
    (output / "INCOMPLETE.json").unlink()
    return manifest


class PixelCache:
    def __init__(self, root, prepared_manifest, split, clips):
        self.root = Path(root)
        if (self.root / "INCOMPLETE.json").exists():
            raise ValueError("RGB pixel cache did not finish")
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        manifest = self.manifest
        identity = {k: v for k, v in manifest.items() if k != "cache_id"}
        if manifest.get("schema") != SCHEMA or manifest.get("cache_id") != digest(identity):
            raise ValueError("Invalid RGB pixel cache identity")
        if (manifest["prepared_id"] != prepared_manifest["prepared_id"]
                or manifest["recipe"] != recipe()):
            raise ValueError("RGB pixel cache preparation or transform differs")
        if split not in manifest["splits"] or split not in manifest["clips"]:
            raise ValueError(f"RGB pixel cache has no {split} split")
        self.entries = manifest["clips"][split]
        if set(self.entries) != set(clips):
            raise ValueError("RGB pixel cache clip set differs")
        for cid, clip in clips.items():
            entry = self.entries[cid]
            if entry["key"] != clip_key(clip) or entry["file"] != f"{split}/{cid}.safetensors":
                raise ValueError(f"RGB pixel cache clip inputs differ: {cid}")
            path = self.root / entry["file"]
            if not path.is_file() or path.stat().st_size != entry["bytes"]:
                raise ValueError(f"Missing or truncated RGB pixel cache: {cid}")
        self._verified = set()

    def load(self, cid):
        import torch
        from safetensors.torch import load_file

        entry = self.entries[cid]
        path = self.root / entry["file"]
        if cid not in self._verified:
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"RGB pixel cache checksum mismatch: {cid}")
        tensors = load_file(path)
        if set(tensors) != {"pixel_values"}:
            raise ValueError(f"Invalid RGB pixel cache tensors: {cid}")
        pixels = tensors["pixel_values"]
        if pixels.dtype != torch.float32 or tuple(pixels.shape) != SHAPE:
            raise ValueError(f"Invalid RGB pixel cache dtype/shape: {cid}")
        if cid not in self._verified:
            if not torch.isfinite(pixels).all():
                raise ValueError(f"Non-finite RGB pixel cache: {cid}")
            self._verified.add(cid)
        return pixels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train"])
    parser.add_argument("--workers", type=int, default=1, help="Cache preparation processes only")
    args = parser.parse_args()
    result = build_cache(**vars(args))
    print(json.dumps(dict(status="complete", cache_id=result["cache_id"],
                          counts={s: len(c) for s, c in result["clips"].items()})))


if __name__ == "__main__":
    main()
