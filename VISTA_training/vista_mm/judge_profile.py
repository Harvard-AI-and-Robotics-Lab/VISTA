"""Load explicit OpenRouter judge settings without importing model runtimes."""

import json
from pathlib import Path


def load_profile(path):
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "model",
        "revision",
        "provider_request",
        "max_tokens",
        "reasoning",
        "response_format",
    }
    if (
        not isinstance(profile, dict)
        or not required <= profile.keys()
        or profile.keys() - required - {"temperature"}
    ):
        raise ValueError("Invalid judge profile fields")
    if not isinstance(profile["model"], str) or "/" not in profile["model"]:
        raise ValueError("Use an explicit OpenRouter model ID")
    if type(profile["max_tokens"]) is not int or profile["max_tokens"] <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if not isinstance(profile["provider_request"], dict) or not isinstance(
        profile["reasoning"], dict
    ):
        raise ValueError("Provider and reasoning settings must be objects")
    if profile["response_format"] != {"type": "json_object"}:
        raise ValueError("Judge requires JSON-object responses")
    return profile


def request_settings(protocol):
    settings = {
        key: protocol[key]
        for key in ("model", "max_tokens", "reasoning", "response_format")
    }
    settings["provider"] = protocol["provider_request"]
    if "temperature" in protocol:
        settings["temperature"] = protocol["temperature"]
    return settings
