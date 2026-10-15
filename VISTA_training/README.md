# VISTA training and Judge 3

Training, inference, and judging for the five selected models from the archived
VISTA campaigns, using the same no-audio sensor-fusion method.

| CLI model | Source campaign | Training GPUs | Answer cap | Archived Judge 3 score |
| --- | --- | ---: | ---: | ---: |
| `internvl3_2b` | `internvl3_no_audio_seed17` | 1 | 512 | 3.541475 |
| `internvl3_8b` | `internvl3_no_audio_seed17` | 1 | 512 | 3.720737 |
| `qwen3_vl_4b` | `qwen_corrected_frames_20260925` | 2 | 4096 | 3.950691 |
| `qwen3_vl_8b` | `qwen_corrected_frames_20260925` | 2 | 4096 | 3.926728 |
| `gemma3_12b` | `gemma3_no_audio_seed17` | 1 | 512 | 3.239171 |

## Setup and training

Training and model prediction require Linux x86-64, CUDA, the complete prepared
`native_v2` dataset, and pretrained encoder assets. The Hugging Face VISTA annotation
tables and raw clip files are not this prepared cache. This compact package does
not include an end-to-end raw-data preparation command; supply the prepared bundle
and encoder assets, and override their archived absolute paths explicitly.

```bash
cd VISTA_training
uv sync --python 3.11 --frozen
source .venv/bin/activate
python run.py verify
python run.py train --model qwen3_vl_4b --stage both \
  --prepared /path/to/native_v2 --encoder-assets /path/to/encoders/v1
```

Replace `--model` with any model in the table. Qwen defaults to the archived
two-GPU `torchrun` configuration. On one GPU, add `--gpus 1`; the launcher adjusts
gradient accumulation to retain effective batch size 32. Memory requirements
still depend on the model and archived image resolution. InternVL and Gemma use
the original single-GPU trainer. New adapters are written to
`runs/<model>/stage1/adapter` and `runs/<model>/stage2/adapter`.

The selected method uses eight RGB frames and IMU, magbaro, gaze, hands, and
trajectory. It uses question-conditioned reliability gates, width-256 bottleneck
fusion, eight fused slots plus twenty stage-2 readout slots, LoRA rank 32/alpha 64,
seed 17, 210/140 training steps, and 35 stage-2 sensor-only warmup steps.
Backbone revisions, processor settings, and family-specific prompts are preserved.
Qwen retains the corrected sequential frame decoder, original 1408-pixel inputs,
patch projection, padded SDPA, and distributed warmup implementation.

## Training smoke tests

For a training smoke test launched from the **repository root**:

```bash
VISTA_training/.venv/bin/python VISTA_training/run.py train \
  --model internvl3_2b --stage both --smoke-test \
  --prepared /path/to/native_v2 --encoder-assets /path/to/encoders/v1
```

Use any of the five model names. Smoke tests use one GPU, batch/accumulation 1,
two optimizer steps per stage, and one stage-2 sensor-only warmup step. They save
disposable adapters under ignored `outputs/smoke-runs/`. Run again with a new
`--training-root` to avoid overwriting completed runs. These tests exercise the
real backbones, data, backward pass, optimizer and stage-1-to-stage-2 reload;
they do not reproduce a full training run or validate two-GPU execution.

In the recorded campaign, all five models passed this two-stage smoke test from
the repository root on one H100 80GB with the locked environment. Each stage
completed two optimizer steps with finite losses. Saved checkpoints matched their
recorded hashes, stage 2 reloaded stage 1,
and both LoRA and sensor weights changed after warmup. The launcher sets
`PYTHONSAFEPATH=1` so the current directory cannot mask Qwen's selected runtime.

## Full InternVL validation

The results below record the completed training campaign. Offline `verify` checks
the preserved files and configurations; reproducing these scores requires the
prepared data, encoder assets, model execution, and hosted judging described here.

Both InternVL sizes completed the original 210/140-step training schedule on one
H100 80GB, with seed 17, batch 8, accumulation 4, and 35 stage-2 sensor-only warmup
steps. New adapters were evaluated on all 217 validation tasks per model,
using the preserved inference settings and Judge 3 profile.

| Model | Archived Judge 3 score | Fresh training Judge 3 score | Meaningfully correct |
| --- | ---: | ---: | ---: |
| InternVL3 2B | 3.541475/5 | 3.539631/5 | 52.9954% |
| InternVL3 8B | 3.720737/5 | 3.726959/5 | 57.6037% |

All 434 predictions completed without token-limit truncation. The aggregate
scores are close to the archived results. The newly trained weights differ;
139/217 2B answers and 66/217 8B answers exactly match the archived adapters'
answers. This run establishes similar aggregate performance for this seed and
environment. Judge 3 uses a hosted, unpinned model. Full Qwen/Gemma training and
two-GPU execution remain untested. Checkpoints, data, logs, and judge responses
are kept outside version control.

## Inference and judging

```bash
python run.py predict --model gemma3_12b \
  --checkpoint /path/to/stage2/adapter --prepared /path/to/native_v2 \
  --encoder-assets /path/to/encoders/v1
python run.py judge --model gemma3_12b \
  --prepared /path/to/native_v2 --key-files /path/to/openrouter_key.txt
```

Each command targets all 217 validation tasks. Outputs are written to
`outputs/<model>/`. Use `--dry-run` to inspect commands and `predict --resume` to
resume a matching prediction run. The checkpoint's parent must contain its
recorded `experiment.json` and `prepared_manifest.json`; new training creates them.
Prediction requires a trained checkpoint. Use `VISTA_benchmark` for zero-shot
inference from pretrained models.
The launcher checks new local runs, downloaded adapters, then the original
`/workspace/VISTA_campaigns/<campaign>/<model>/stage2/adapter` path by default.

Judge 3 consumes the prediction file together with its `.protocol.json` and
`.complete.json` sidecars. It rejects incomplete or duplicate prediction sets and
checks their hashes and prepared-data identity. The standalone evaluators in
`VISTA_eval` and `EgoBlind/EGOBLIND_eval` accept benchmark JSONL files and use different
request settings; their outputs do not substitute for these Judge 3 artifacts.

Judge 3 uses the exact `assistancevlmjudge3-5-v1` rubric, eight video frames,
`openai/gpt-6-luna`, OpenAI-only routing, high reasoning, and JSON verdicts.
The archived campaign scores in this README were obtained with medium reasoning;
the profile change is recorded in `provenance.json`. Rejudging at high reasoning
requires a new output directory, because resume checks the complete judge protocol.
Judging makes paid API calls, with default concurrency 12 and a $5 limit per run;
in-flight requests can exceed the threshold slightly. Raw responses and usage are
saved. Keys stay in external files. The hosted judge revision is unpinned.
The launcher exposes `--concurrency`, `--max-cost-usd`, `--predictions`, and
`--output-dir` to override these defaults and paths.

The two InternVL adapters can be downloaded from the pinned private HF release:

```bash
python run.py download-checkpoint --model internvl3_2b
python run.py download-checkpoint --model internvl3_8b
```

Authenticate to Hugging Face outside this folder. This command adds the matching
recorded metadata next to each downloaded adapter. Qwen and Gemma weights remain
in the original experiment archive; supply their restored adapter paths.

Training needs the full prepared data and pretrained encoder assets. The selected
prepared ID is `a3048d75632ec5fd222e7d4cbab31637bb5baa6546a42f5d6b54c8529a317c58`.
Prepared clip manifests contain absolute video paths; `--prepared` changes the
dataset directory, not paths inside its manifests. Gemma requires access to its
gated Google backbone. This project uses Linux x86-64, Python 3.11, PyTorch cu128,
Transformers 4.57.6, PEFT 0.18.0, and OpenCV 5.0.0.

## Files and verification

- `run.py`: one launcher for all five models.
- `models.json`: all fifteen original stage/eval configs, shared prepared metadata,
  checkpoint hashes, and judge profile in one file.
- `vista_mm/`: shared fusion, three backbone integrations, trainer, prediction code,
  required data/encoder helpers, and judge helpers.
- `qwen.patch`: only the archived Qwen runtime differences; applied automatically
  to the shared runtime under ignored `outputs/runtime/` before Qwen launches.
- `judge.py` and `rubric.json`: the preserved evaluator and rubric.
- `provenance.json`: source and config hashes; `LICENSES.txt`: used upstream notices.
- `pyproject.toml` and `uv.lock`: pinned setup.

`python run.py verify` checks all fifteen configs and byte-for-byte hashes of the
declared files in both runtime variants. `runtime_cleanup` records the original
hashes for the removed comparison helper and unsupported zero-shot prediction
path, together with the original Qwen patch hash. `unused_preparation_cleanup`
records removal of disconnected raw-preparation helpers and the unused JSONL
writer. The system prompt, schema/recipe metadata, RGB decoder, and prepared-cache
loader remain available. The unchanged shared fusion code is used
by every model. Saved result bundles, duplicate checkpoint metadata, test copies,
unused merge scripts, and the unused BEATs source files are excluded. The selected
five-stream configs never load audio; adding audio requires restoring that code.
Use `run.py` to select the correct runtime rather than calling the shared trainer
directly for Qwen. The generated runtime contains no changes to archived model
calculations. InternVL received full training and evaluation validation as
documented above; Qwen and Gemma received two-stage training smoke tests.

Offline verification requires a compatible GNU `patch` executable on `PATH`.
The Qwen patch includes surrounding context and is verified against the declared
Qwen runtime hashes after application. GNU patch 2.7.6 passed this check; the
Windows GNU patch 2.5.9 build failed while creating the runtime. The repository's
`.gitattributes` keeps preserved Python, JSON, and patch files at LF line endings
so Windows checkout conversion does not invalidate their byte hashes. Training
and model execution still target the Linux/CUDA environment described above.
