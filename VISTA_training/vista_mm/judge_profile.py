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
    }
    if (
        not isinstance(profile, dict)
        or not required <= profile.keys()
        or profile.keys() - required - {"temperature"}
    ):
        raise ValueError("Invalid judge profile fields")
    if profile["model"] != "openai/gpt-6-luna":
        raise ValueError("Judge model must be openai/gpt-6-luna")
    if type(profile["max_tokens"]) is not int or profile["max_tokens"] <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if not isinstance(profile["provider_request"], dict) or not isinstance(
        profile["reasoning"], dict
    ):
        raise ValueError("Provider and reasoning settings must be objects")
    if profile["reasoning"] != {"effort": "high"}:
        raise ValueError("Judge reasoning effort must be high")
    if profile["provider_request"].get("only") != ["openai"] or profile["provider_request"].get("allow_fallbacks") is not False:
        raise ValueError("Judge must route only to OpenAI without fallbacks")
    if profile.get("temperature", 1) != 1:
        raise ValueError("Judge temperature must be 1")
    return profile


def request_settings(protocol):
    if protocol["model"] != "openai/gpt-6-luna":
        raise ValueError("Judge model must be openai/gpt-6-luna")
    if protocol["reasoning"] != {"effort": "high"}:
        raise ValueError("Judge reasoning effort must be high")
    if protocol["provider_request"].get("only") != ["openai"] or protocol["provider_request"].get("allow_fallbacks") is not False:
        raise ValueError("Judge must route only to OpenAI without fallbacks")
    if protocol.get("temperature", 1) != 1:
        raise ValueError("Judge temperature must be 1")
    settings = {
        key: protocol[key]
        for key in ("model", "max_tokens", "reasoning")
    }
    settings["provider"] = protocol["provider_request"]
    settings["temperature"] = 1
    return settings
