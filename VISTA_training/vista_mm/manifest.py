"""Task prompts shared with the RGB benchmark."""

import importlib.util
from pathlib import Path


def _load_prompts():
    for parent in Path(__file__).resolve().parents:
        path = parent / "VISTA_benchmark/prompts.py"
        if path.is_file():
            spec = importlib.util.spec_from_file_location("vista_task_prompts", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return path, module
    raise FileNotFoundError("VISTA_benchmark/prompts.py is required")


PROMPTS_PATH, PROMPTS = _load_prompts()


def build_prompt(item):
    task = item["task_type"]
    if task == "scene_description":
        return PROMPTS.build_scene_prompt()
    if task.startswith("question_") and task.removeprefix("question_").isdigit():
        return PROMPTS.build_qa_prompt(item["question"])
    if task.startswith("goal_") and task.removeprefix("goal_").isdigit():
        # The prepared cache wraps the original goal in this fixed prefix.
        goal = item["question"].removeprefix("Give navigation guidance for this goal: ")
        return PROMPTS.build_guidance_prompt(goal)
    raise ValueError(f"Unknown task: {task}")
