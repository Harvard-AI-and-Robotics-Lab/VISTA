#!/usr/bin/env bash
#
# VISTA zero-shot benchmark launcher.
#
#   bash run.bash <model_key> [limit]
#   e.g.  bash run.bash qwen3vl_2b
#         bash run.bash internvl3_1b 5      # only the first 5 clips (smoke test)
#
# Set VISTA_DATASET_DIR and credentials in the repository-root .env.
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$REPO_DIR/.." && pwd)"

# Load untracked dataset configuration and credentials.
if [ -f "$REPO_ROOT/.env" ]; then
    set -a; . "$REPO_ROOT/.env"; set +a
fi

############################################
# CONFIG
############################################
: "${VISTA_DATASET_DIR:?Set VISTA_DATASET_DIR to the downloaded VISTA folder in .env}"
DATASET_ROOT="$VISTA_DATASET_DIR/test"
CSV_PATH="$VISTA_DATASET_DIR/test.csv"
OUTPUT_DIR="$REPO_DIR/outputs"
ENVS_DIR="$REPO_DIR/venvs"

if [ ! -d "$DATASET_ROOT" ] || [ ! -f "$CSV_PATH" ]; then
    echo "[ERROR] VISTA_DATASET_DIR must contain test/ and test.csv: $VISTA_DATASET_DIR"
    exit 1
fi

MODEL=${1:-"llama3_video"}
LIMIT=${2:-}

############################################
# MODEL TABLES
############################################
# NOTE: these keys must stay in sync with run.py's --model choices and with the model_map
# in models.py. internvl3_38b is deliberately absent: models.py has no entry for it, so it
# would reserve GPUs and then die with "Unknown model key".
declare -A TP_NEEDED
TP_NEEDED["qwen3vl_2b"]=1
TP_NEEDED["qwen3vl_4b"]=1
TP_NEEDED["qwen3vl_8b"]=1
TP_NEEDED["qwen3vl_32b"]=2
TP_NEEDED["internvl3_1b"]=1
TP_NEEDED["internvl3_2b"]=1
TP_NEEDED["internvl3_8b"]=1
TP_NEEDED["internvl3_9b"]=1
TP_NEEDED["internvl3_14b"]=1
TP_NEEDED["llava_video"]=1
TP_NEEDED["llama3_video"]=1
TP_NEEDED["gemma3_4b"]=1
TP_NEEDED["gemma3_12b"]=1
# 27B is 51.1 GiB in bf16 and does not fit one 48 GB A6000, so it is sharded over two.
TP_NEEDED["gemma3_27b"]=2
TP_NEEDED["gemini_3_1_flash_lite"]=0
TP_NEEDED["gpt_realtime_mini"]=0

declare -A VENV_MAP
VENV_MAP["qwen3vl_2b"]="venv_qwen"
VENV_MAP["qwen3vl_4b"]="venv_qwen"
VENV_MAP["qwen3vl_8b"]="venv_qwen"
VENV_MAP["qwen3vl_32b"]="venv_qwen"
VENV_MAP["internvl3_1b"]="venv_internvl3"
VENV_MAP["internvl3_2b"]="venv_internvl3"
VENV_MAP["internvl3_8b"]="venv_internvl3"
# internvl3_9b is the one InternVL3 size built on InternLM2 rather than Qwen2. Its vendored
# modeling code assumes PreTrainedModel provides .generate(), which transformers 4.50 moved
# into GenerationMixin, so it needs the transformers 4.49 environment. On 4.57 it loads and
# then fails per-clip with "'InternLM2ForCausalLM' object has no attribute 'generate'" --
# which run.py catches, leaving an empty predictions file and exit code 0.
VENV_MAP["internvl3_9b"]="venv_internvl3_9b"
VENV_MAP["internvl3_14b"]="venv_internvl3"
VENV_MAP["llava_video"]="venv_llava"
VENV_MAP["llama3_video"]="venv_videollama"
# Gemma 3 runs on vLLM by default (VISTA_GEMMA_BACKEND=vllm), so it belongs in the vLLM env.
# Set VISTA_GEMMA_BACKEND=hf for the transformers reference implementation instead -- that path
# needs venv_internvl3 (transformers 4.57) and is ~8x slower.
VENV_MAP["gemma3_4b"]="venv_qwen"
VENV_MAP["gemma3_12b"]="venv_qwen"
VENV_MAP["gemma3_27b"]="venv_qwen"
VENV_MAP["gemini_3_1_flash_lite"]="venv_api"
VENV_MAP["gpt_realtime_mini"]="venv_api"

if [ -z "${TP_NEEDED[$MODEL]+x}" ]; then
    echo "[ERROR] Unknown model key: $MODEL"
    echo "[ERROR] Known keys: ${!TP_NEEDED[*]}"
    exit 1
fi

VENV=${VENV_MAP[$MODEL]}
N_GPUS=${TP_NEEDED[$MODEL]}
echo "[INFO] Model: $MODEL → needs $N_GPUS GPU(s)"

############################################
# GPU SELECTION
############################################
# API-only models need no GPU. The old code ran the picker with n=0, ended up with an
# empty CUDA_VISIBLE_DEVICES and exited 1 before reaching run.py.
if [ "$N_GPUS" -gt 0 ]; then
    if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
        CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
            awk -F',' -v n="$N_GPUS" '
            BEGIN { count=0 }
            {
                gsub(/ /,"",$1); gsub(/ /,"",$2)
                if ($2+0 > 44000 && count < n) {
                    printf "%s,", $1
                    count++
                }
            }' | sed 's/,$//')

        if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
            echo "[ERROR] Not enough free GPUs for $MODEL (need $N_GPUS with >44 GB free)."
            echo "[ERROR] Set CUDA_VISIBLE_DEVICES explicitly to override, e.g. CUDA_VISIBLE_DEVICES=6"
            nvidia-smi --query-gpu=index,memory.free --format=csv
            exit 1
        fi
    fi
    export CUDA_VISIBLE_DEVICES
    echo "[INFO] CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"
else
    echo "[INFO] $MODEL is API-based — no GPU required."
fi

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONNOUSERSITE=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Use vLLM's PyTorch sampler by default to avoid requiring a local CUDA compiler
# for FlashInfer's JIT kernels. Set VLLM_USE_FLASHINFER_SAMPLER=1 to opt in.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
# Allow the FlashInfer cache location to be overridden independently.
export FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/flashinfer}"

############################################
# PYTHON ENVIRONMENT
############################################
# Resolve the env to a concrete interpreter and call it directly. Sourcing bin/activate
# silently fell through to the system python when the env was missing, which then died at
# `import torch` with a confusing message.
ENV_PREFIX="$ENVS_DIR/$VENV"
if [ ! -x "$ENV_PREFIX/bin/python" ]; then
    echo "[ERROR] No python at $ENV_PREFIX/bin/python (model $MODEL wants env '$VENV')."
    echo "[ERROR] Create or symlink the environment under $ENVS_DIR; see README.md#environment-setup."
    exit 1
fi
PY="$ENV_PREFIX/bin/python"
# Put the env's bin/ first so console scripts are found too. vLLM's flashinfer backend
# shells out to `ninja` at JIT time and dies with FileNotFoundError without this.
export PATH="$ENV_PREFIX/bin:$PATH"
export VIRTUAL_ENV="$ENV_PREFIX"
echo "[INFO] Model: $MODEL → env: $VENV ($ENV_PREFIX)"

############################################
# RUN
############################################
mkdir -p "$OUTPUT_DIR"
echo "[INFO] dataset_root: $DATASET_ROOT"
echo "[INFO] csv_path    : $CSV_PATH"
echo "[INFO] output_dir  : $OUTPUT_DIR"

# Credentials are NOT passed on argv -- run.py reads HF_TOKEN / OPENAI_API_KEY /
# GEMINI_API_KEY from the environment. On a shared machine anything
# on the command line is visible to every user via `ps`.
#
# An empty-but-SET credential is worse than an unset one: huggingface_hub turns HF_TOKEN=""
# into the header `Authorization: Bearer ` and dies with
# `httpx.LocalProtocolError: Illegal header value b'Bearer '`. Unset the blank ones.
for _v in HF_TOKEN OPENAI_API_KEY GEMINI_API_KEY; do
    if [ -z "${!_v:-}" ]; then unset "$_v"; else export "$_v"; fi
done
unset _v

ARGS=(
    --dataset_root      "$DATASET_ROOT"
    --csv_path          "$CSV_PATH"
    --output_dir        "$OUTPUT_DIR"
    --model             "$MODEL"
)
[ -n "$LIMIT" ] && ARGS+=(--limit "$LIMIT")

exec "$PY" "$REPO_DIR/run.py" "${ARGS[@]}"
