#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${1:?Usage: bash scripts/run_model.sh <model> <dataset_root> <csv_path> [output_dir]}"
DATASET_ROOT="${2:?Usage: bash scripts/run_model.sh <model> <dataset_root> <csv_path> [output_dir]}"
CSV_PATH="${3:?Usage: bash scripts/run_model.sh <model> <dataset_root> <csv_path> [output_dir]}"
OUTPUT_DIR="${4:-${BENCH_DIR}/outputs/${MODEL}}"

cd "${BENCH_DIR}"

python run.py \
  --dataset_root "${DATASET_ROOT}" \
  --csv_path "${CSV_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --model "${MODEL}"