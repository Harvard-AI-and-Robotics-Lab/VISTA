# VISTA: A Multimodal Egocentric Benchmark for Goal-Oriented Perception and Assistance to Blind and Low-Vision Users

VISTA (**Visually Impaired Scene and Task Annotations**) benchmarks first-person
visual assistance through scene descriptions, question answering, and
goal-conditioned instructions across ten task categories. This repository provides
zero-shot inference, sensor-conditioned model training, and evaluation for VISTA.

[[Homepage]](https://harvard-ai-and-robotics-lab.github.io/VISTA/)

## Dataset Setup

Download the full VISTA dataset, including annotations and all available modalities, from
[Hugging Face](https://huggingface.co/datasets/harvardairobotics/VISTA):

```bash
python -m pip install -U huggingface_hub
hf download harvardairobotics/VISTA --repo-type dataset --local-dir /path/to/VISTA
```

This downloads all three splits with their available video, audio, sensor, and tracking files:

```text
/path/to/VISTA/
  train.csv
  val.csv
  test.csv
  train/
  val/
  test/
```

## Zero-Shot Inference

Run the commands below from this code repository's root, using Linux/Bash and
Python 3.12. Install only the environment for your model. Local models require
CUDA; hosted models require API credentials.

### Environment Setup

For paid API models (GPT Realtime Mini or Gemini), install:

```bash
python3.12 -m venv VISTA_benchmark/venvs/venv_api
VISTA_benchmark/venvs/venv_api/bin/python -m pip install \
  -r VISTA_benchmark/requirements/requirements_api.txt
```

These API models use the launcher's `venv_api` environment without requiring a
local GPU. See the [benchmark setup guide](VISTA_benchmark/README.md) for
open-source model installation.

### Supported Models

| Models | CLI keys |
|---|---|
| Qwen3-VL | `qwen3vl_2b`, `qwen3vl_4b`, `qwen3vl_8b`, `qwen3vl_32b` |
| InternVL3 | `internvl3_1b`, `internvl3_2b`, `internvl3_8b`, `internvl3_9b`, `internvl3_14b` |
| VideoLLaMA3 | `llama3_video` |
| Gemma 3 | `gemma3_4b`, `gemma3_12b`, `gemma3_27b` |
| Gemini 3.1 Flash-Lite | `gemini_3_1_flash_lite` |
| GPT Realtime Mini | `gpt_realtime_mini` |
| LLaVA-Video | `llava_video` |

### Run

```bash
cp .env.example .env
# Set VISTA_DATASET_DIR=/path/to/VISTA and your model's credentials in .env.
bash VISTA_benchmark/run.bash gpt_realtime_mini 5
```

`VISTA_DATASET_DIR` must contain both `test.csv` and `test/`; do not set it to the
`test/` folder itself. Set `OPENAI_API_KEY` or `GEMINI_API_KEY` for hosted models.
Remove `5` to run the full test split.

## VLM-as-a-Judge Evaluation

The default judge is GPT-6 Luna with `high` reasoning effort.
Set up the environment:

```bash
python3 -m venv VISTA_eval/.venv
source VISTA_eval/.venv/bin/activate
python -m pip install -r VISTA_eval/requirements.txt
```

Set `OPENAI_API_KEY` in the repository's `.env`, then run with your prediction file:

```bash
python VISTA_eval/vlm_judge_openrouter.py run \
  --dataset dataset_vista \
  --csv /path/to/VISTA/test.csv --dataset-root /path/to/VISTA/test \
  --preds /path/to/predictions_candidate.jsonl \
  --model-name candidate \
  --out VISTA_eval/out/candidate.jsonl
```

See the [evaluation guide](VISTA_eval/README.md) for aggregation and more options.

## VISTA Sensor-Conditioned Model

Install the model environment on Linux with a CUDA-capable GPU:

```bash
python -m pip install uv
cd VISTA_training
uv sync --python 3.11 --frozen
source .venv/bin/activate
```

Accounts with access to the released VISTA checkpoints can download them with:

```bash
python run.py download-checkpoint --model internvl3_2b
python run.py download-checkpoint --model internvl3_8b
```

Run test inference with the released VISTA 8B checkpoint:

```bash
python run.py predict --model internvl3_8b \
  --checkpoint checkpoints/internvl3_8b/stage2/adapter \
  --prepared /path/to/prepared \
  --encoder-assets /path/to/encoders
```

For training VISTA, data requirements, and other model backbones, see the
[VISTA training guide](VISTA_training/README.md).

## EgoBlind

See the [EgoBlind guide](EgoBlind/README.md) for its dataset setup, zero-shot
inference, text metrics, and judging.
