# VISTA Zero-Shot Benchmark

Zero-shot inference for multimodal models on the **VISTA** test set.

All open- and closed-source models default to eight RGB frames. Pass
`--num_frames` to override this count. Gemini and GPT Realtime Mini receive
the sampled RGB frames and task text.

Pipeline:
1. **Inference** (this folder) — run a model over the test clips → `predictions_<model>.jsonl`
2. **Evaluation** ([`../VISTA_eval`](../VISTA_eval)) — score predictions with a VLM judge

---

## Quick start

```bash
cd VISTA_benchmark
cp ../.env.example ../.env      # then set the dataset folder and needed credentials
bash run.bash <model_key> [limit]

# examples
bash run.bash qwen3vl_2b        # full test split
bash run.bash internvl3_1b 5    # first 5 clips only (smoke test)
CUDA_VISIBLE_DEVICES=6 bash run.bash internvl3_1b   # pin to a specific GPU
```

`run.bash` picks the model environment, selects visible GPUs, and calls `run.py`.

---

## Configuration

Set `VISTA_DATASET_DIR` in the repository-root `.env` to the downloaded VISTA
repository folder containing `test.csv` and `test/`:

```bash
VISTA_DATASET_DIR=/path/to/VISTA
```

`run.bash` derives the test video directory and annotation CSV from this one path.
Predictions are written to `VISTA_benchmark/outputs/`, and model environments are
loaded from `VISTA_benchmark/venvs/`. Environments may be symlinks to installations
elsewhere. Use `run.py` directly when choosing a custom output directory or split.

Completed `(clip_id, task_type)` pairs are skipped on resume. The decoding sidecar
rejects changed decoding settings. Use a new output directory or `--output_name`
when changing data, prompts, frame counts, or input conditions. Generated outputs
are gitignored. The RGB/text runner rejects predictions marked with a different
input mode on resume.

Uncomment only the credentials your model needs in `.env`; empty credentials are
normalized to unset values. Direct `run.py` calls require exported credentials
and explicit CLI paths. Standalone evaluation uses `--api-key-file` and its CLI
options, as described in [VISTA_eval](../VISTA_eval/README.md).

### Expected dataset layout

```
<VISTA_DATASET_DIR>/test/
└── TC01/
    └── TC01_001/
        ├── TC01_001.mp4        # regular RGB clip used by the benchmark
        ├── vrs/                # raw sensor streams (not read by the benchmark)
        └── mps/                # Aria MPS derivatives (not read by the benchmark)
```

For the Hugging Face release described in [Dataset Setup](../README.md#dataset-setup),
the CSVs are at the downloaded repository root and videos are under split folders.
Set `VISTA_DATASET_DIR=/path/to/VISTA` in `.env`. For direct Python calls, pass
the derived paths explicitly:

```bash
python run.py --dataset_root /path/to/VISTA/test \
  --csv_path /path/to/VISTA/test.csv --output_dir outputs/smoke \
  --model internvl3_1b --max_predictions 10
```

The RGB benchmark reads `<clip_id>.mp4` and the CSV. Its loader scans scene
references 1–8, questions 1–8, and goals 1–17. The released Hub test CSV contains
450 clips and 1,828 tasks within these ranges. The standalone evaluator scans
scene references/questions through 9 and goals through 15; those different limits
do not omit populated question or goal columns in the current Hub test CSV.

---

## Supported models

| Model | Key | Environment | Loader |
|---|---|---|---|
| Qwen3-VL 2B / 4B / 8B / 32B | `qwen3vl_2b` … `qwen3vl_32b` | `venv_qwen` | vLLM |
| InternVL3 1B / 2B / 8B / 14B | `internvl3_1b`, `_2b`, `_8b`, `_14b` | `venv_internvl3` | transformers |
| InternVL3 9B | `internvl3_9b` | **`venv_internvl3_9b`** | transformers |
| VideoLLaMA3 7B | `llama3_video` | `venv_videollama` | transformers |
| LLaVA-Video 7B | `llava_video` | `venv_llava` | LLaVA builder |
| Gemma 3 4B / 12B / 27B | `gemma3_4b`, `_12b`, `_27b` | `venv_qwen` | vLLM |
| Gemini 3.1 Flash-Lite | `gemini_3_1_flash_lite` | `venv_qwen` | Google Gen AI API |
| GPT Realtime Mini | `gpt_realtime_mini` | `venv_qwen` | HTTP / WebSocket API |

The historical key `qwen3vl_32b` resolves to `Qwen/Qwen3-VL-30B-A3B-Instruct`.
The shell environment map assumes the default `VISTA_GEMMA_BACKEND=vllm`; for the
`hf` reference backend, invoke `run.py` directly from a compatible Transformers
environment.

> **`internvl3_9b` is the exception in its family.** It is built on an InternLM2 language
> model rather than Qwen2, and its vendored modeling code assumes `PreTrainedModel` provides
> `.generate()` — which transformers 4.50 moved into `GenerationMixin`. On transformers 4.57
> it loads fine and then fails on every clip with
> `AttributeError: 'InternLM2ForCausalLM' object has no attribute 'generate'`. Because
> `run.py` catches per-clip exceptions, the process still **exits 0 with an empty predictions
> file** — check the line count, not the exit code. It runs correctly on the transformers
> 4.49 environment, which it shares with `llama3_video`.

`run.bash` selects the environment automatically. Model keys are kept in sync across
`run.bash`, `run.py` and `models.py`; an unknown key now fails immediately instead of
reserving GPUs and dying later.

---

## Environment setup

Use Python 3.12 and install only the environment your model needs. Keep the pinned
stacks separate; local models require CUDA, while API models do not need a GPU.

| Models | Environment | Requirements |
|---|---|---|
| Qwen3-VL, Gemma 3, Gemini, GPT Realtime Mini | `venv_qwen` | [requirements_qwen.txt](requirements/requirements_qwen.txt) |
| InternVL3 1B / 2B / 8B / 14B | `venv_internvl3` | [requirements_internvl3.txt](requirements/requirements_internvl3.txt) |
| InternVL3 9B | `venv_internvl3_9b` | [requirements_internvl3_9b.txt](requirements/requirements_internvl3_9b.txt) |
| VideoLLaMA3 | `venv_videollama` | [requirements_videollama.txt](requirements/requirements_videollama.txt) |
| LLaVA-Video | `venv_llava` | See below |

From the repository root, select an environment and requirements file from the
table, then install them (Qwen example):

```bash
cd VISTA_benchmark
VISTA_ENV=venv_qwen
VISTA_REQUIREMENTS=requirements/requirements_qwen.txt
python3.12 -m venv "venvs/$VISTA_ENV"
"venvs/$VISTA_ENV/bin/python" -m pip install -U pip setuptools wheel
"venvs/$VISTA_ENV/bin/python" -m pip install -r "$VISTA_REQUIREMENTS" httpx
```

- **InternVL3 9B:** its requirements use Transformers 4.49.0, matching VideoLLaMA3.
  If VideoLLaMA3 is already installed, reuse it with
  `ln -s venv_videollama venvs/venv_internvl3_9b`.
- **Gemini:** also run `venvs/venv_qwen/bin/python -m pip install google-genai`.
  GPT Realtime Mini's WebSocket dependency is already included.
- **Evaluation:** install [VISTA_eval/requirements.txt](../VISTA_eval/requirements.txt)
  separately; it uses the hosted judge and CPU video decoding.

### `llava_video`

Create `venvs/venv_llava` with Python 3.12 and install a compatible
[LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) version using its LLaVA-Video
instructions. Add the shared runner dependencies:

```bash
venvs/venv_llava/bin/python -m pip install httpx opencv-python-headless pandas Pillow requests tqdm
```

<details>
<summary>Optional cache locations</summary>

Create these directories and export their paths if you need a different cache location:

```bash
export HF_HOME=/path/to/cache/huggingface
export PIP_CACHE_DIR=/path/to/cache/pip
export TMPDIR=/path/to/cache/tmp
```

</details>

---

## Output

```
outputs/predictions_<model>.jsonl
outputs/errors_<model>.jsonl
outputs/decoding_<model>.json
```

One line per (clip, task):

```json
{"clip_id": "TC01_001", "task_type": "scene_description", "prediction": "..."}
{"clip_id": "TC01_001", "task_type": "question_1",        "prediction": "..."}
{"clip_id": "TC01_001", "task_type": "goal_1",            "prediction": "..."}
```

Each clip yields `1 + n_questions + n_goals` items. For the current 450-clip test split
that is **1828 lines** for a complete run — a quick count check:

```bash
wc -l outputs/predictions_<model>.jsonl
```

Also verify that each `(clip_id, task_type)` occurs once and that the identities
exactly match the tasks loaded from the selected test CSV. A matching line count
alone does not establish dataset coverage or the inference protocol.

The six bundled 2,041-item prediction files used obsolete annotations and have
been retired with the unused collection checklist and remain in Git history; see
[archived artifacts](ARCHIVED_ARTIFACTS.md). Verified replacement predictions for
the current dataset are not bundled. Generate them in a fresh output directory
with the current CSV, and retain the decoding sidecar and error log with the results.

Runs **resume**: completed `(clip_id, task_type)` pairs are skipped. To force a fresh run,
use a new output directory or `--output_name`; there is no `--overwrite` flag.
`--limit` limits clips before task expansion, while `--max_predictions` limits the
expanded tasks. Per-task errors are logged and inference normally continues; check
the prediction count and error file instead of relying only on the exit code.

---

## Decoding and reproducibility

Local model paths use sampling with temperature `0.7`, top-p `0.8`, repetition
penalty `1.05`, and an output cap of `4096` tokens. The base seed defaults to `0`;
each task gets a stable seed derived from its `clip_id|task_type` identity.
These settings are configurable through `VISTA_TEMPERATURE`, `VISTA_TOP_P`,
`VISTA_REPETITION_PENALTY`, `VISTA_SEED`, and `VISTA_SEED_MODE`.

`decoding_<model>.json` records the settings and prevents resuming with a different
decoding protocol. `--num_frames` changes the requested frame count from the default
of eight. Hosted backends have their own request settings and do not share the local
per-item seed guarantee; record the provider and model ID with the paper's results.

## RGB + SLAM ablation and LLaVA video input

See [RGB_SLAM.md](RGB_SLAM.md) for paired left/right sampling on the common exported-video timeline, the resumable GPU queue, and the corrected LLaVA-Video input format.
