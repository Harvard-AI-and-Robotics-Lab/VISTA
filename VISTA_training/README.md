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

Minimal Ubuntu images also need OpenCV's system libraries:
`sudo apt-get install -y libxcb1 libgl1`.

### Checkpoints and pausing a single-GPU RGB run

For a run that can be interrupted and resumed, use the recovery entry point:

```bash
python train_resumable.py --model internvl3_2b \
  --prepared data/prepared_rgb_trainval
```

It uses the same 210 + 140 steps, learning rates, eight frames and LoRA settings,
with microbatch 1 and accumulation 32. It saves the adapter, optimizer, scheduler,
random-number state and trainer progress after the first step and every five
steps, retaining two checkpoints per stage. Data loading defaults to zero worker
processes. The final adapter remains the fixed final-step checkpoint.

To request a pause, create `runs/rgb/internvl3_2b/PAUSE_REQUESTED`. Training finishes
the current optimizer step, saves a complete checkpoint, and exits with code 75.
Remove that request file and rerun the command with `--resume` to continue from
the latest complete checkpoint. Completed stages are verified and skipped.
Resume requires the same configuration, prepared data and training source.
Use this workflow instead of OS process suspension on Thunder Compute.

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

## RGB-only supervised fine-tuning

All five backbones also support `--input-mode rgb`. This trains language-model
LoRA adapters on RGB frames and text tasks. The vision encoder, visual projector,
and other backbone weights are frozen. No sensor fusion, sensor tokens, sensor
feature cache, normalization statistics, or pretrained sensor encoders are used.
The default input mode remains `multimodal`.

First prepare the downloaded dataset. Run these commands from `VISTA_training`
inside the Linux environment described above (use WSL on Windows):

```bash
python run.py prepare-rgb --dataset-root /path/to/VISTA --output data/prepared_rgb

python run.py train --input-mode rgb --model internvl3_2b --stage both \
  --prepared data/prepared_rgb
```

By default preparation requires `train.csv`, `val.csv`, `test.csv`, and the
corresponding RGB videos. For a local train/validation copy, add
`--splits train val`; validation stays separate and cannot be used for training.
It writes task/clip manifests and checksums, sampling the same eight
frame indices as multimodal preparation. RGB device timestamps are used when
available; otherwise video-FPS timing is recorded explicitly in the manifest.
`--allow-trailing-timestamp` permits exactly one extra timestamp after the final
video frame. Missing videos, unpaired annotations, or overlapping splits fail
preparation. A failed preparation retains `INCOMPLETE.json`; use a new output
directory after correcting its inputs.

The RGB reader discovers all numbered annotation columns, including scene
reference 9 and goals 16/17. Older benchmark/judge readers have different column
limits. To establish a paired comparison against an existing multimodal cache,
add `--compare-mm-prepared /path/to/prepared_mm`: train/test tasks, reference text,
video hashes, frame indices, and timestamps must match, and the existing task
order is reused. A mismatch is reported instead of silently dropping examples.
RGB judging uses the complete RGB task manifest with the existing paper rubric.

The RGB settings are derived from each model's pinned configuration in
`models.json`: rank 32, alpha 64, dropout 0.05, effective batch size 32, seed 17,
BF16, gradient checkpointing, the same visual processing, and the same
answer-only, per-example normalized loss. Stage one uses 210 steps at `8e-5`;
stage two loads its adapter and uses 140 steps at `2e-5`, with a fresh optimizer
and the existing cosine schedule and 5% learning-rate warmup in each stage.
Fixed step counts take precedence over the legacy epoch setting.

Sensor-only warmup is disabled. Consequently RGB receives 350 LoRA update steps,
whereas multimodal training freezes LoRA for 35 of its 350 steps. Run metadata
records the planned update count; this is a matched total-step budget, not an
identical language-update budget. Both modes use the final checkpoint without
test-based checkpoint selection. Validation and test annotations never enter
training.

RGB checkpoints go to `runs/rgb/<model>/stage{1,2}/adapter/`, and predictions go
to `outputs/rgb/<model>/test/`. A custom `--training-root` is used literally.
If a smaller microbatch is needed, `--batch-size 1` increases gradient
accumulation to preserve effective batch size 32. Qwen supports `--gpus 1` or
`--gpus 2`; InternVL and Gemma use one GPU. Smaller batches do not guarantee that
the larger models fit a given GPU.

```bash
python run.py train --input-mode rgb --model internvl3_2b --stage both \
  --prepared data/prepared_rgb --smoke-test

python run.py predict --input-mode rgb --model internvl3_2b \
  --prepared data/prepared_rgb \
  --checkpoint runs/rgb/internvl3_2b/stage2/adapter

python run.py judge --input-mode rgb --model internvl3_2b \
  --prepared data/prepared_rgb --csv /path/to/VISTA/test.csv \
  --key-files /path/to/openrouter_key.txt
```

The smoke run uses two optimizer steps per stage, microbatch 1 and accumulation
1, and writes to `outputs/smoke-runs/rgb/`; it checks execution and checkpoint
handoff separately from the full training command's effective batch of 32.
Prediction and judging use the same deterministic decoding, 4096-token limit,
and scoring rubric as multimodal evaluation. Judging is an explicit paid API
operation, separate from local verification and tests.

Local checks, including small randomly initialized model architectures with
synthetic RGB inputs (no model downloads), can be run with:

```bash
uv pip install pytest
python run.py verify
python -m pytest tests --confcutdir=. -q
```

GPU tests cover LoRA gradients, frozen visual weights, adapter reloads, and
stage-two initialization. They skip when CUDA is unavailable. These tests do
not establish memory requirements or convergence for full-size pretrained runs.

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
