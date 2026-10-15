# VISTA: A Multimodal Egocentric Benchmark for Goal-Oriented Perception and Assistance to Blind and Low-Vision Users

VISTA (**Visually Impaired Scene and Task Annotations**) benchmarks first-person
visual assistance through scene descriptions, question answering, and
goal-conditioned instructions across ten task categories. This repository provides
zero-shot inference, sensor-conditioned model training, and evaluation for VISTA.

[[Homepage]](https://harvard-ai-and-robotics-lab.github.io/VISTA/)

## Dataset Setup

Load the annotation tables from [Hugging Face](https://huggingface.co/datasets/harvardairobotics/VISTA):

```python
from datasets import load_dataset

# Login using e.g. `hf auth login` to access this dataset
ds = load_dataset("harvardairobotics/VISTA")
```

## Zero-Shot Inference

Install the [model dependencies](VISTA_benchmark/README.md#environment-setup), then run:

```bash
cp .env.example .env
# Edit .env to set VISTA_DATASET_DIR and credentials.
bash VISTA_benchmark/run.bash gpt_realtime_mini
```

Predictions are saved to `VISTA_benchmark/outputs/`. Add a second argument
(e.g. `5`) to limit clips. See the [benchmark README](VISTA_benchmark/README.md)
for more options.

### Supported Models

| Model | CLI keys |
|---|---|
| Qwen3-VL | `qwen3vl_2b`, `qwen3vl_4b`, `qwen3vl_8b`, `qwen3vl_32b` |
| InternVL3 | `internvl3_1b`, `internvl3_2b`, `internvl3_8b`, `internvl3_9b`, `internvl3_14b` |
| Gemma 3 | `gemma3_4b`, `gemma3_12b`, `gemma3_27b` |
| VideoLLaMA3 7B | `llama3_video` |
| LLaVA-Video 7B | `llava_video` |
| GPT Realtime Mini | `gpt_realtime_mini` |
| Gemini 3.1 Flash-Lite | `gemini_3_1_flash_lite` |

## Evaluation

[VISTA_eval](VISTA_eval/README.md) scores predictions with GPT-6 Luna, high reasoning,
and the four-component v5 rubric (0-5, with a 1.95 critical-error cap).

```bash
cd VISTA_eval
python vlm_judge_openrouter.py run --dataset dataset_vista \
  --dataset-root /path/to/VISTA/test --csv /path/to/VISTA/test.csv \
  --preds /path/to/predictions_mymodel.jsonl --model-name mymodel \
  --api-key-file /path/to/openai_key.txt --out out/mymodel.jsonl
python vlm_judge_openrouter.py aggregate out/mymodel.jsonl
```

## Sensor-Conditioned Model Training

[VISTA_training](VISTA_training/README.md) supports InternVL3 2B/8B, Qwen3-VL 4B/8B,
and Gemma 3 12B. The baseline combines RGB with IMU, magnetometer/barometer, gaze,
hands, and trajectory using question-conditioned sensor fusion and LoRA.
Training requires the prepared `native_v2` dataset and pretrained encoder assets.

```bash
cd VISTA_training
uv sync --python 3.11 --frozen
source .venv/bin/activate
python run.py train --model internvl3_2b --stage both \
  --prepared /path/to/native_v2 --encoder-assets /path/to/encoders/v1
```

See the training README for checkpoints, inference, Judge 3, and verification.

The additional EgoBlind benchmark is documented in [EgoBlind](EgoBlind/README.md).
