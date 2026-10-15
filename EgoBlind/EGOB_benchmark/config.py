############################################
# CONFIG
############################################

# Random seed for reproducibility
SEED = 42

# Default frame count for all open- and closed-source models
NUM_FRAMES = {
    "gemini_3_1_flash_lite": 8,
    "gpt_realtime_mini": 8,
    "internvl3_1b":  8,
    "internvl3_2b":  8,
    "internvl3_8b":  8,
    "internvl3_14b": 8,
    "llava_video":   8,
    "llama3_video":  8,
    "gemma3_4b":     8,
    "qwen3vl_2b":  8,
    "qwen3vl_4b":  8,
    "qwen3vl_8b":  8,
    "qwen3vl_32b": 8,
}
