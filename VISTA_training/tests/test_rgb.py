"""Behavioral checks for RGB preparation and multimodal-compatible training."""

import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vista_mm.config import Experiment
from vista_mm.rgb_data import RGBDataset, annotation_items, prepare_rgb, verify_rgb

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("vista_run", ROOT / "run.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.fixture
def raw_dataset(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    for split, cid in (("train", "TC01_001"), ("val", "TC01_002"), ("test", "TC01_003")):
        row = dict(clip_id=cid, scene_description_1="A room.", scene_description_9="An indoor room.",
                   question_9="Where is the door?", answer_9="On the left.",
                   goal_17="Find the exit", instruction_17="Walk toward the door.")
        with (root / f"{split}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        directory = root / split / "TC01" / cid
        directory.mkdir(parents=True)
        writer = cv2.VideoWriter(str(directory / f"{cid}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32))
        assert writer.isOpened()
        for i in range(10):
            writer.write(np.full((32, 32, 3), i * 20, np.uint8))
        writer.release()
        (directory / "vrs").mkdir()
        with (directory / "vrs/rgb_timestamps.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame_index", "capture_timestamp_ns"])
            writer.writerows((i, 1_000_000_000 + i * 100_000_000) for i in range(10))
    return root


def rgb_config(**updates):
    return Experiment(**dict(launcher.model_configuration("internvl3_2b", "stage1", "rgb"), **updates))


def test_preparation_and_dataset_need_no_sensor_cache(raw_dataset, tmp_path):
    out = tmp_path / "prepared"
    manifest = prepare_rgb(raw_dataset, out)
    assert manifest["counts"] == {s: {"clips": 1, "items": 3} for s in ("train", "val", "test")}
    assert manifest["timestamp_sources"] == {"device_timestamps": 3}
    assert not (out / "normalization.json").exists()
    assert not list(out.glob("*.npz"))
    dataset = RGBDataset(out, "train", rgb_config(), training=True)
    sample = dataset[0]
    assert sample["frames"].shape == (8, 32, 32, 3)
    assert sample["frame_indices"].tolist() == [0, 1, 3, 4, 5, 6, 8, 9]
    assert np.allclose(sample["frame_times"], [0, .1, .3, .4, .5, .6, .8, .9])
    assert sample["references"] == ["A room.", "An indoor room."]
    assert "sensors" not in sample
    assert [i["task_type"] for i in dataset.items] == ["scene_description", "question_9", "goal_17"]
    with pytest.raises(ValueError, match="train split"):
        RGBDataset(out, "test", rgb_config(), training=True)
    with pytest.raises(ValueError, match="ablations"):
        RGBDataset(out, "test", rgb_config(), shuffle_sensors=True)
    with pytest.raises(ValueError, match="sensor modalities"):
        verify_rgb(out, ["imu"])
    # Changing a video or manifest after preparation must fail before it is used.
    Path(dataset.clips[sample["clip_id"]]["video"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="video changed"):
        RGBDataset(out, "train", rgb_config())[0]
    with (out / "train.items.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        verify_rgb(out)


def test_video_timing_fallback_is_recorded(raw_dataset, tmp_path):
    for path in raw_dataset.rglob("rgb_timestamps.csv"):
        path.unlink()
    result = prepare_rgb(raw_dataset, tmp_path / "prepared")
    assert result["timestamp_sources"] == {"video_fps": 3}


def test_preparation_can_keep_validation_without_a_test_split(raw_dataset, tmp_path):
    (raw_dataset / "test.csv").unlink()
    out = tmp_path / "prepared"
    manifest = prepare_rgb(raw_dataset, out, splits=("train", "val"))
    assert manifest["splits"] == ["train", "val"]
    assert set(verify_rgb(out)["counts"]) == {"train", "val"}
    assert not (out / "test.items.jsonl").exists()
    assert len(RGBDataset(out, "train", rgb_config(), training=True)) == 3
    assert len(RGBDataset(out, "val", rgb_config())) == 3
    with pytest.raises(ValueError, match="train split"):
        RGBDataset(out, "val", rgb_config(), training=True)
    with pytest.raises(ValueError, match="not prepared"):
        RGBDataset(out, "test", rgb_config())


def test_annotation_and_split_errors_are_not_silently_dropped(raw_dataset, tmp_path):
    path = raw_dataset / "train.csv"
    text = path.read_text()
    path.write_text(text.replace("On the left.", ""))
    with pytest.raises(ValueError, match="Unpaired"):
        annotation_items(path, "train")
    path.write_text(text)
    val = raw_dataset / "val.csv"
    val.write_text(val.read_text().replace("TC01_002", "TC01_001"))
    with pytest.raises(ValueError, match="overlap"):
        prepare_rgb(raw_dataset, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize("model", list(launcher.MODELS))
def test_all_model_configs_keep_the_mm_recipe(model, tmp_path):
    gpu_count = launcher.MODELS[model]["training_gpus"]
    for stage in ("stage1", "stage2", "eval"):
        original = launcher.MODELS[model]["configs"][stage]
        rgb = launcher.model_configuration(model, stage, "rgb")
        assert launcher.model_configuration(model, stage) == original
        for key in ("model_id", "revision", "num_frames", "max_steps", "backbone_lr",
                    "max_length", "min_pixels", "max_pixels", "lora_rank", "lora_alpha", "lora_dropout"):
            assert rgb[key] == original[key]
        assert rgb["sensor"] == {"modalities": []}
        assert rgb["sensor_warmup_steps"] == 0
        assert rgb["batch_size"] * rgb["accumulation"] * gpu_count == 32
        # Qwen's extra visual_mode belongs to its patched runtime.
        native = {k: v for k, v in rgb.items() if k != "visual_mode"}
        Experiment(**native)
    args = SimpleNamespace(model=model, action="train", input_mode="rgb", prepared=None,
                           encoder_assets=None, training_root=str(tmp_path / "rgb"),
                           batch_size=1, smoke_test=False, dry_run=False)
    old_root = launcher.ROOT
    launcher.ROOT = tmp_path
    try:
        path = launcher.configuration(args, "stage2", 1)
        config = json.loads(path.read_text())
        assert config["accumulation"] == 32
        assert config["initial_adapter"] == str(tmp_path / "rgb" / model / "stage1/adapter")
    finally:
        launcher.ROOT = old_root


@pytest.mark.parametrize("change", [dict(sensor={"modalities": ["imu"]}),
                                    dict(sensor_warmup_steps=35), dict(encoder_assets="encoders")])
def test_rgb_config_rejects_sensor_dependencies(change):
    with pytest.raises(ValueError, match="RGB mode"):
        rgb_config(**change)


def test_rgb_judging_keeps_high_numbered_tasks(raw_dataset, tmp_path):
    from vista_mm.judge3 import Judge3Rubric
    _, items = annotation_items(raw_dataset / "test.csv", "test")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("".join(json.dumps(dict(i, prediction="A response.")) + "\n" for i in items))
    rubric = Judge3Rubric(ROOT / "rubric.json")
    loaded = rubric.load_items(raw_dataset / "test.csv", predictions, input_mode="rgb")
    assert {i["item_id"] for i in loaded} == {i["item_id"] for i in items}
    assert next(i for i in loaded if i["task_type"] == "goal_17")["qtype"] == "guidance"


def test_collator_masks_the_prompt_and_emits_no_sensor_inputs(monkeypatch):
    import torch
    from vista_mm.data import VistaCollator
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 2
        def __call__(self, text, **kwargs):
            return {"input_ids": [7, 8]}
    collator = VistaCollator(SimpleNamespace(tokenizer=Tokenizer()), rgb_config())
    monkeypatch.setattr(collator, "encode_prompt", lambda e: {
        "input_ids": torch.tensor([[3, 4, 5]]), "pixel_values": torch.ones(1, 3, 4, 4)})
    batch = collator([dict(item_id="TC01_001|scene_description", split="train", references=["A room."])])
    assert batch["labels"].tolist() == [[-100, -100, -100, 7, 8, 2]]
    assert "pixel_values" in batch
    assert not {"sensors", "question_input_ids", "question_attention_mask"}.intersection(batch)
