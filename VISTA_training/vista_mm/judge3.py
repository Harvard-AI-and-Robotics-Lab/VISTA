"""VISTA judge prompt, verdict parsing, and frame encoding."""

import importlib.util
import json
from pathlib import Path

from PIL import Image

from .judge_rubric import task_group


def _paper_file(name):
    for parent in Path(__file__).resolve().parents:
        path = parent / "VISTA_eval" / name
        if path.is_file():
            return path
    raise FileNotFoundError("The paper-facing VISTA_eval package is required")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PAPER_ENGINE_PATH = _paper_file("vlm_judge_openrouter.py")
PAPER_DATASET_PATH = _paper_file("dataset_vista.py")
PAPER_ENGINE = _load(PAPER_ENGINE_PATH, "vista_paper_judge_engine")
PAPER_DATASET = _load(PAPER_DATASET_PATH, "vista_paper_judge_dataset")


class Judge3Rubric:
    """Use the Appendix B.4 prompt and the VISTA_eval parser."""

    def __init__(self, path):
        self.contract = PAPER_ENGINE.load_rubric_contract(path)
        canonical = PAPER_ENGINE.load_rubric_contract(
            PAPER_ENGINE_PATH.parent / "prompts/assistancevlmjudge_vista.json"
        )
        for key in ("system", "rubric", "tasks", "component_limits",
                    "component_step", "critical_error_cap"):
            if self.contract[key] != canonical[key]:
                raise ValueError("Judge contract differs from the paper: " + key)

    def prompt(self, item, prediction):
        group = task_group(item["task_type"])
        qtype = "scene_description" if group == "scene" else group
        return PAPER_ENGINE.build_prompt_v5(
            dict(item, qtype=qtype, prediction=prediction), self.contract
        )

    def parse(self, raw):
        return PAPER_ENGINE.parse_judge_output_v5(raw, self.contract)

    def image_content(self, frames):
        return [{"type": "image_url", "image_url": {"url": PAPER_ENGINE._frame_to_data_url(
            frame if isinstance(frame, Image.Image) else Image.fromarray(frame)
        )}} for frame in frames]

    def load_items(self, csv_path, predictions):
        return PAPER_DATASET.load_items(csv_path, predictions)
