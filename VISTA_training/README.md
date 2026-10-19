# VISTA Model Training and Evaluation

Train sensor-conditioned VISTA models, run test inference, and score predictions.

## Supported Models

| Backbone | Model option |
|---|---|
| InternVL3 2B | `internvl3_2b` |
| InternVL3 8B | `internvl3_8b` |
| Qwen3-VL 4B | `qwen3_vl_4b` |
| Qwen3-VL 8B | `qwen3_vl_8b` |
| Gemma 3 12B | `gemma3_12b` |

## Installation

Use Linux, Python 3.11, a CUDA-capable NVIDIA GPU, and GNU `patch`.
From the repository root:

```bash
python -m pip install uv
cd VISTA_training
uv sync --python 3.11 --frozen
source .venv/bin/activate
python run.py verify
```

Run the remaining commands from `VISTA_training`. Dependencies are pinned in
[uv.lock](uv.lock); `verify` checks configurations and source integrity without
training or API calls.

## Data and Pretrained Assets

Training and inference need a prepared `native_v2` cache and pretrained sensor
encoder assets. The cache must include annotations, features, normalization
statistics fitted on the training split, manifests, and valid video paths.
The [raw dataset](../README.md#dataset-setup) needs preprocessing; this package
does not include an end-to-end preparation command. Your account also needs
access to the chosen backbone's weights.

## Released VISTA Checkpoints

Accounts with access can download the released InternVL checkpoints:

```bash
python run.py download-checkpoint --model internvl3_2b
python run.py download-checkpoint --model internvl3_8b
```

These are saved under `checkpoints/<model>/stage2/adapter/` with verified metadata.
Supply Qwen and Gemma checkpoints separately with `predict --checkpoint`.

## Training

Train the adapters and sensor-fusion network in two stages, using settings in
[models.json](models.json):

```bash
python run.py train --model internvl3_8b --stage both \
  --prepared /path/to/prepared \
  --encoder-assets /path/to/encoders
```

Checkpoints go to `runs/<model>/stage1/adapter/` and `runs/<model>/stage2/adapter/`.
Use `--stage stage1` or `--stage stage2` for one stage, or `--training-root` to
change the output location. InternVL and Gemma use one GPU; Qwen defaults to two
and supports `--gpus 1`.

## Test Inference

Run the released checkpoint on the test split:

```bash
python run.py predict --model internvl3_8b \
  --checkpoint checkpoints/internvl3_8b/stage2/adapter \
  --prepared /path/to/prepared \
  --encoder-assets /path/to/encoders
```

For your own trained model, use `runs/internvl3_8b/stage2/adapter` instead.
Predictions go to `outputs/<model>/test/predictions.jsonl`; `--output` changes the
path and `--resume` continues a compatible run. Keep `experiment.json` and
`prepared_manifest.json` beside the adapter directory; training and downloads
create them automatically.

## Evaluation

Use an OpenRouter API key file to score the complete test predictions:

```bash
python run.py judge --model internvl3_8b \
  --prepared /path/to/prepared \
  --csv /path/to/VISTA/test.csv \
  --predictions outputs/internvl3_8b/test/predictions.jsonl \
  --key-files /path/to/openrouter_key.txt
```

This uses GPT-6 Luna with high reasoning effort through OpenRouter, sharing the
[VISTA rubric and parser](../VISTA_eval/README.md#scoring). It checks prediction
completeness and data consistency before scoring. Malformed verdicts are recorded
and excluded. `--output-dir` sets the results directory.

Judging incurs API charges. `--concurrency` controls parallel requests and
`--max-cost-usd` sets a spending threshold; in-flight requests can exceed it.
Use `python run.py <command> --help` for options, or add `--dry-run` to `train`,
`predict`, or `judge` to preview without running it.
