"""Cache parity, corruption detection, and unchanged stochastic supervision."""

import json
from pathlib import Path

import pytest
import torch

from test_rgb import raw_dataset, rgb_config
from vista_mm.data import VistaCollator, internvl_rgb_pixel_values
from vista_mm.provenance import digest
from vista_mm.rgb_cache import build_cache
from vista_mm.rgb_data import RGBDataset, prepare_rgb


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, return_tensors=None, **kwargs):
        ids = [ord(c) % 200 + 3 for c in text.replace("<IMG_CONTEXT>", "I")]
        return {"input_ids": torch.tensor([ids]) if return_tensors else ids}


class InternVLProcessor:
    image_seq_length = 256
    tokenizer = Tokenizer()


@pytest.fixture
def cached_data(raw_dataset, tmp_path):
    prepared, cache = tmp_path / "prepared", tmp_path / "pixels"
    prepare_rgb(raw_dataset, prepared, splits=("train", "val"))
    build_cache(prepared, cache)
    return prepared, cache


def test_cached_batches_and_rng_are_identical(cached_data):
    prepared, cache = cached_data
    config = rgb_config()
    plain = RGBDataset(prepared, "train", config, training=True)
    cached = RGBDataset(prepared, "train", config, training=True, pixel_cache=cache)
    collator = VistaCollator(InternVLProcessor(), config, training=True)
    torch.manual_seed(17)
    for index in (0, 1, 2, 0):
        sample = plain[index]
        cached_sample = cached[index]
        assert "frames" not in cached_sample
        assert torch.equal(internvl_rgb_pixel_values(sample["frames"]),
                           cached_sample["rgb_pixel_values"])
        rng = torch.get_rng_state()
        expected = collator([sample])
        after = torch.get_rng_state()
        torch.set_rng_state(rng)
        actual = collator([cached_sample])
        assert torch.equal(after, torch.get_rng_state())
        assert set(actual) == set(expected)
        for key in actual:
            assert torch.equal(actual[key], expected[key]), key


def test_cached_dataset_bypasses_video_decode(cached_data, monkeypatch):
    from vista_mm import rgb_data

    prepared, cache = cached_data
    cached = RGBDataset(prepared, "train", rgb_config(), pixel_cache=cache)
    def unexpected_decode(*args):
        raise AssertionError("Cached examples must not decode video")
    monkeypatch.setattr(rgb_data, "read_zero_shot_frames", unexpected_decode)
    assert cached[0]["rgb_pixel_values"].shape == (8, 3, 448, 448)


def test_cache_rejects_modified_video(cached_data):
    prepared, cache = cached_data
    cached = RGBDataset(prepared, "train", rgb_config(), pixel_cache=cache)
    Path(cached.clips["TC01_001"]["video"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="video changed"):
        cached[0]


@pytest.mark.parametrize("damage", ["checksum", "missing", "incomplete", "recipe", "preparation", "clip"])
def test_cache_rejects_stale_or_incomplete_inputs(cached_data, damage):
    prepared, cache = cached_data
    if damage == "checksum":
        p = cache / "train/TC01_001.safetensors"
        data = bytearray(p.read_bytes())
        data[-1] ^= 1
        p.write_bytes(data)
    elif damage == "missing":
        (cache / "train/TC01_001.safetensors").unlink()
    elif damage == "incomplete":
        (cache / "INCOMPLETE.json").write_text("{}")
    else:
        p = cache / "manifest.json"
        manifest = json.loads(p.read_text())
        if damage == "recipe":
            manifest["recipe"]["dtype"] = "bfloat16"
        elif damage == "preparation":
            manifest["prepared_id"] = "different"
        else:
            manifest["clips"]["train"]["TC01_001"]["key"] = "different"
        manifest["cache_id"] = digest({k: v for k, v in manifest.items() if k != "cache_id"})
        p.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="RGB pixel cache"):
        RGBDataset(prepared, "train", rgb_config(), pixel_cache=cache)[0]


def test_cache_is_explicit_and_model_specific(cached_data):
    prepared, cache = cached_data
    assert "frames" in RGBDataset(prepared, "val", rgb_config())[0]
    with pytest.raises(ValueError, match="no val split"):
        RGBDataset(prepared, "val", rgb_config(), pixel_cache=cache)
    cfg = rgb_config(model_id="Qwen/Qwen3-VL-4B-Instruct")
    with pytest.raises(ValueError, match="supports InternVL"):
        RGBDataset(prepared, "train", cfg, pixel_cache=cache)
