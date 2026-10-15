# EgoBlind Zero-Shot Benchmark

Run zero-shot inference on EgoBlind test questions using sampled RGB frames and
question text. All model keys default to eight requested frames and a 4,096-token
output cap. Predictions are saved as JSONL for text metrics or the hosted VLM judge
in [EGOBLIND_eval](../EGOBLIND_eval/README.md).

## Folder setup

- `run.py`: main script for zero-shot inference
- `eval.py`: computes BLEU, ROUGE, SBERT, and BERTScore summaries from prediction files
- `models.py`: model loading and inference for Qwen3-VL, InternVL3, Gemma 3, LLaVA-Video, VideoLLaMA3, Gemini, and GPT Realtime Mini
- `data.py`: loads the CSV and extracts frames from videos
- `prompts.py`: builds the prompt sent to the model
- `config.py`: default frame counts and seed
- `requirements.txt`: shared runner and hosted-model dependencies
- `requirements/`: separate local-model stacks and optional text metrics
- `scripts/run_model.sh`: CLI wrapper for a selected model, dataset root, and CSV
- `outputs/`: prediction and error files are written here.

## Dataset setup

Supply the EgoBlind test CSV and videos externally; `test.csv` is not bundled here.
`--dataset_root` points to the directory containing `split_0` through `split_9`:

```text
/path/to/EgoBlind/Test/
  test.csv
  split_0/00923.mp4
  ... split_9/01578.mp4
```

The CSV must provide `question_id`, `video_name`, `question`, and `start-time/s`.
Video names are mapped to five-digit MP4 filenames using the ranges in `data.py`.
Evaluation also needs `answer0` through `answer3`; the text-metric script requires
the `type` column for category summaries.

## Setup

Install one model-family requirement file in a separate environment. These files
reuse the pinned stacks in [VISTA_benchmark's setup](../../VISTA_benchmark/README.md#environment-setup)
and include EgoBlind's shared runner dependencies:

| Models | Requirement file |
| --- | --- |
| Qwen3-VL | `requirements/requirements_qwen.txt` |
| InternVL3 and Gemma 3 | `requirements/requirements_internvl3.txt` |
| VideoLLaMA3 | `requirements/requirements_videollama.txt` |

LLaVA-Video needs its own compatible LLaVA-NeXT installation, plus the shared
`requirements.txt`. For hosted models, install only `requirements.txt`.

For example, create the InternVL/Gemma environment from the repository root:

```bash
python3.12 -m venv EgoBlind/EGOB_benchmark/venvs/venv_internvl3
source EgoBlind/EGOB_benchmark/venvs/venv_internvl3/bin/activate
python -m pip install -r EgoBlind/EGOB_benchmark/requirements/requirements_internvl3.txt
cd EgoBlind/EGOB_benchmark
```

Local model execution uses CUDA. The runner also imports PyTorch at startup for
API models through `models.py`.

Launch `run.py` with explicit paths, or use `scripts/run_model.sh` from the selected
active environment. GPU visibility can be set with `CUDA_VISIBLE_DEVICES`.

## Smoke Test

Run five questions to check the selected environment and data paths:

```bash
python run.py \
  --dataset_root /path/to/EgoBlind/Test \
  --csv_path /path/to/EgoBlind/Test/test.csv \
  --output_dir outputs/internvl3_1b_smoke_5 \
  --model internvl3_1b \
  --num_frames 8 \
  --limit 5
```

## Full zero-shot inference

Example for InternVL3-1B:

```bash
python run.py \
  --dataset_root /path/to/EgoBlind/Test \
  --csv_path /path/to/EgoBlind/Test/test.csv \
  --output_dir outputs/internvl3_1b_8frames_max_tokens_4096 \
  --model internvl3_1b \
  --num_frames 8
```

Example for VideoLLaMA3 at 8 frames and 640 resolution:

```bash
python run.py \
  --dataset_root /path/to/EgoBlind/Test \
  --csv_path /path/to/EgoBlind/Test/test.csv \
  --output_dir outputs/llama3_video_8frames_640_max_tokens_4096 \
  --model llama3_video \
  --num_frames 8 \
  --llama3_video_resize 640
```

Example for Qwen3-VL-2B with vLLM:

```bash
python run.py \
  --dataset_root /path/to/EgoBlind/Test \
  --csv_path /path/to/EgoBlind/Test/test.csv \
  --output_dir outputs/qwen3vl_2b_8frames_max_tokens_4096_vllm \
  --model qwen3vl_2b \
  --num_frames 8
```

## Outputs

Each run writes two files inside the selected `--output_dir`:

```text
predictions_<model>.jsonl
errors_<model>.jsonl
```

Prediction rows use this schema:

```json
{"question_id": "question_001", "pred": "The door is on the left."}
```

Runs resume by skipping `question_id` values already in the prediction file.
Resume does not check the dataset, prompts, model revision, frame count, or decoding
settings. Use a separate output directory for each experiment. Per-question errors
are logged and inference continues, so check prediction coverage and the error file.

Example:

```text
outputs/internvl3_1b_8frames_max_tokens_4096/predictions_internvl3_1b.jsonl
outputs/internvl3_1b_8frames_max_tokens_4096/errors_internvl3_1b.jsonl
```

## Metric Evaluations

After zero-shot inference finishes, we use `eval.py` to calculate text metrics.

Install `requirements/requirements_metrics.txt` in the environment used for
`eval.py`. Its semantic models are downloaded on first use.

Evaluate `predictions_*.jsonl` files one directory level below an output folder:

```bash
python eval.py \
  --test_path /path/to/EgoBlind/Test/test.csv \
  --outputs_dir outputs
```

Evaluate one model output folder:

```bash
python eval.py \
  --test_path /path/to/EgoBlind/Test/test.csv \
  --pred_path outputs/internvl3_1b_8frames_max_tokens_4096/predictions_internvl3_1b.jsonl
```

Each evaluated model folder gets a summary CSV:

```text
summary_<output_folder_name>.csv
```

The summary includes an overall row and rows grouped by the CSV's `type` category:

```text
bleu
rouge1
rouge2
rougeL
sbert_score
bert_score
```

The semantic evaluation metrics use the following models

```text
SBERT: sentence-transformers/all-mpnet-base-v2
BERTScore: roberta-large
```

## Models List

Use one of these values for `--model`:

```text
internvl3_1b
internvl3_2b
internvl3_8b
internvl3_14b
qwen3vl_2b
qwen3vl_4b
qwen3vl_8b
qwen3vl_32b
gemma3_4b
llava_video
llama3_video
gemini_3_1_flash_lite
gpt_realtime_mini
```

`qwen3vl_32b` resolves to `Qwen/Qwen3-VL-30B-A3B-Instruct`. The CLI does not expose
InternVL3 9B/38B or Gemma 3 12B/27B. Gemma 3 4B defaults to
`google/gemma-3-4b-it`; `EGOB_GEMMA3_4B_MODEL` and `EGOB_GEMMA3_4B_REVISION`
override its repository and revision.

Set the credentials needed by the selected model in the environment:

```bash
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
# For gated Hugging Face models
export HF_TOKEN=...
```

The runner also loads a discovered `.env` using `python-dotenv` with
`override=True`, so that file can replace exported values.

## Sampling and decoding

Pass `--num_frames` to change the requested frame count; use at least two frames.
`--limit` limits questions, and `--llama3_video_resize` defaults to 640 pixels
square (`0` disables this resize).

Frames are normally sampled from the start of the clip to the question timestamp.
Repeated indices are removed. The current inference sampler falls back to the full
clip when the timestamp maps to frame zero or below; check those items before using
results for timestamp-constrained comparisons. The standalone judge clamps those
timestamps to frame zero instead.

The startup seed is 42. InternVL, Gemma, and LLaVA explicitly use greedy generation;
Qwen uses vLLM's default sampling settings, and VideoLLaMA3 inherits its generation
configuration. This runner does not use VISTA's per-item decoding seeds or sidecar
guard. Equal frame counts do not establish matching decoding protocols.
