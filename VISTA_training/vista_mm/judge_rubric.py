"""Map VISTA annotation types to response groups."""

import re

def task_group(task):
    if task == "scene_description":
        return "scene"
    if re.fullmatch(r"question_\d+", task):
        return "qa"
    if re.fullmatch(r"goal_\d+", task):
        return "guidance"
    raise ValueError(f"Unknown task: {task}")
