"""Judge 3 validation in integer units of 0.05 points."""

import re

RUBRIC_VERSION = "assistancevlmjudge3-5-v1"
LIMITS = {
    "evidence_grounding": 35,
    "answer_correctness": 35,
    "task_completion": 20,
    "accessible_communication": 10,
}
FLAGS = ("meaningfully_correct", "critical_error", "gt_conflict", "evidence_limited")
TEXT_FIELDS = ("visual_evidence", "rationale")
VERDICT_KEYS = set(LIMITS) | set(FLAGS) | set(TEXT_FIELDS)


def task_group(task):
    if task == "scene_description":
        return "scene"
    if re.fullmatch(r"question_\d+", task):
        return "qa"
    if re.fullmatch(r"goal_\d+", task):
        return "guidance"
    raise ValueError(f"Unknown task: {task}")


def validate_verdict(d, limits=None):
    limits = LIMITS if limits is None else limits
    if not isinstance(d, dict) or set(d) != VERDICT_KEYS:
        raise ValueError("Verdict must have exactly the declared rubric keys")
    for key, maximum in limits.items():
        if type(d[key]) is not int or not 0 <= d[key] <= maximum:
            raise ValueError(f"Invalid rubric value: {key}")
    for key in FLAGS:
        if type(d[key]) is not bool:
            raise ValueError(f"Invalid boolean: {key}")
    for key in TEXT_FIELDS:
        if not isinstance(d[key], str) or not d[key].strip():
            raise ValueError(f"Missing explanation: {key}")
    if d["critical_error"] and d["meaningfully_correct"]:
        raise ValueError("A critical error cannot be meaningfully correct")
    uncapped = sum(d[k] for k in LIMITS)
    return {
        **d,
        "uncapped_score": uncapped,
        "score": min(uncapped, 39) if d["critical_error"] else uncapped,
        "rubric_version": RUBRIC_VERSION,
        "parse_error": False,
    }
