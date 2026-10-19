# VISTA Model Training and Evaluation

Train sensor-conditioned VISTA models and evaluate their predictions on the test
split. The implementation supports InternVL3, Qwen3-VL, and Gemma 3, combining RGB
video with synchronized wearable sensor inputs for scene description, question
answering, and goal-oriented guidance.

## Supported Models

| Backbone | Model option |
| --- | --- |
| InternVL3 2B | `internvl3_2b` |
| InternVL3 8B | `internvl3_8b` |
| Qwen3-VL 4B | `qwen3_vl_4b` |
| Qwen3-VL 8B | `qwen3_vl_8b` |
| Gemma 3 12B | `gemma3_12b` |

## Installation

Use Linux with Python 3.11, a CUDA-capable NVIDIA GPU, and GNU `patch`.
Dependencies are defined in [pyproject.toml](pyproject.toml) and pinned in
[uv.lock](uv.lock).

From the repository root:

```bash
cd VISTA_training
uv sync --python 3.11 --frozen
source .venv/bin/activate
python run.py verify
```

Run the remaining commands from `VISTA_training` with this environment activated.

## Data and Pretrained Assets

Training and inference require:

- A compatible prepared VISTA cache containing annotations, sensor features,
  normalization statistics, and data manifests.
- The pretrained sensor encoder assets.
- Access to the pretrained backbone and, for inference, a VISTA adapter checkpoint.

See the repository's [dataset setup instructions](../README.md#dataset-setup) for
the source data. The raw dataset alone is insufficient for this package; an
end-to-end raw-data preparation command is not included. Normalization statistics
must be fitted on the training split, and video paths stored in the prepared
cache must resolve in the current environment.

The default data directories are `data/prepared` and `data/encoders`, relative to
`VISTA_training`. Use `--prepared` and `--encoder-assets` to specify other locations.

Authenticate to Hugging Face before accessing restricted model weights. Gemma
backbones also require acceptance of their model access terms.

## Training

VISTA combines eight RGB frames with five sensor streams: IMU,
magnetometer/barometer, gaze, hand tracking, and trajectory. Question-conditioned
reliability gates and a bottleneck fusion network produce sensor tokens for the
language model. Training updates the language LoRA adapters and sensor-fusion
network while keeping the vision and pretrained sensor encoders frozen.

Training has two stages. The first introduces eight bottleneck tokens. The second
adds twenty readout tokens and begins with a sensor-only warmup before joint
adaptation. Model-specific settings are defined in [models.json](models.json).

```bash
python run.py train --model internvl3_8b --stage both \
  --prepared /path/to/prepared \
  --encoder-assets /path/to/encoders
```

Use `--stage stage1` or `--stage stage2` to run an individual stage. InternVL and
Gemma use one GPU. Qwen uses two GPUs by default; `--gpus 1` enables single-GPU
training with gradient accumulation adjusted to preserve the effective batch size.

Checkpoints are written to:

```text
runs/<model>/stage1/adapter/
runs/<model>/stage2/adapter/
```

Set `--training-root` to choose another output directory. Training uses only the
training split and saves the final adapter after the configured number of steps.
The test split is reserved for prediction and evaluation after training.

## Test Inference

All supported models use eight uniformly sampled frames, greedy decoding with one
beam, and a maximum output length of 4,096 tokens. Task prompts are shared with the
[zero-shot benchmark](../VISTA_benchmark/README.md) through its
[prompt templates](../VISTA_benchmark/prompts.py).

```bash
python run.py predict --model internvl3_8b \
  --checkpoint runs/internvl3_8b/stage2/adapter \
  --prepared /path/to/prepared \
  --encoder-assets /path/to/encoders
```

Predictions are saved to `outputs/<model>/test/predictions.jsonl`. Use `--output`
to choose another destination or `--resume` to continue an interrupted run.
Resumption checks that the checkpoint, data, and inference configuration match.

Keep the checkpoint's `experiment.json` and `prepared_manifest.json` files beside
its adapter directory. Training and checkpoint downloads create these metadata
files automatically.

## Evaluation

Prediction and judging operate exclusively on the VISTA test split: 450 clips and
1,828 tasks. The evaluator uses GPT-6 Luna with high reasoning effort and
temperature 1. Prompt wording follows Appendix B.4 of the paper, with prompt
construction, reference loading, image encoding, and verdict parsing shared with
[VISTA_eval](../VISTA_eval/README.md).

```bash
python run.py judge --model internvl3_8b \
  --prepared /path/to/prepared \
  --csv /path/to/VISTA/test.csv \
  --predictions outputs/internvl3_8b/test/predictions.jsonl \
  --key-files /path/to/openrouter_key.txt
```

The training evaluator accesses the judge through OpenRouter with OpenAI-only
routing. Store API credentials in external key files. Judging incurs API charges;
use `--concurrency` and `--max-cost-usd` to control parallelism and spending.
Requests already in progress may take spending beyond the specified threshold.

Each prediction receives one verdict, without format-correction retries. Scores
combine evidence grounding, answer correctness, task completion, and accessible
communication on a 0–5 scale. Critical errors cap the total at 1.95. Malformed
verdicts are recorded and excluded from score aggregation.

Evaluation checks prediction completeness and data consistency before scoring.
Use `--output-dir` to choose where verdicts, summaries, and usage records are saved.

## Pretrained VISTA Adapters

Accounts with access to the released InternVL adapters can download them with:

```bash
python run.py download-checkpoint --model internvl3_2b
python run.py download-checkpoint --model internvl3_8b
```

Adapters are saved under `checkpoints/<model>/stage2/adapter/` and checked against
the recorded checkpoint metadata. Qwen and Gemma adapters must be supplied
separately using `predict --checkpoint`.

## Verification and Command Reference

```bash
python run.py verify
python run.py train --help
python run.py predict --help
python run.py judge --help
```

`verify` checks model configurations, runtime implementations, and shared prompt
and judge sources without training a model or making API calls. Add `--dry-run`
to a training, prediction, or judging command to preview the command without
starting training, inference, or judging.
