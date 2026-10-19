# VISTA Evaluation with GPT-6 Luna

Evaluate scene descriptions, question answers, and goal-oriented assistance using **`gpt-6-luna`** through the OpenAI API.
The judge receives sampled video frames, the task or question, reference
answers, and a candidate model's prediction. It returns scores and supporting
explanations, which are saved as JSONL verdicts.

The historical filename `vlm_judge_openrouter.py` is retained, but this standalone
package calls OpenAI directly with reasoning effort fixed to `high`.
[The training evaluator](../VISTA_training/README.md#test-prediction-and-judging)
reuses this prompt builder, verdict parser, and image encoding through OpenRouter,
with the same `high` reasoning requirement and additional provenance checks.

## Installation

Use Python 3.10 or later and an OpenAI API key with access to `gpt-6-luna`.
Video decoding runs locally on the CPU; judging is performed through the API.

```bash
cd VISTA_eval
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Save the API key as a plain-text file and pass its path with `--api-key-file`.
The default is `openai_key.txt` in the current working directory. The script
reads the key from this file, not from `OPENAI_API_KEY`.

## Input data

The reference CSV contains one row per clip:

| Columns | Content |
|---|---|
| `clip_id` | Clip identifier, such as `TC01_001` |
| `scene_description_1` ... `scene_description_9` | Alternative scene descriptions |
| `question_i`, `answer_i` | Question and reference answer, for `i = 1 ... 9` |
| `goal_i`, `instruction_i` | Goal and reference instructions, for `i = 1 ... 15` |

Predictions are joined to references by `(clip_id, task_type)`. Each line of
the prediction JSONL file contains one answer:

```json
{"clip_id": "TC01_001", "task_type": "scene_description", "prediction": "A doorway is ahead."}
{"clip_id": "TC01_001", "task_type": "question_1", "prediction": "The door is on the left."}
{"clip_id": "TC01_001", "task_type": "goal_1", "prediction": "Turn left toward the doorway."}
```

Tasks without reference text or a matching prediction are skipped.

### Video frames

For the Hugging Face release, pass the root-level annotation CSV and the test video
folder separately (see [Dataset Setup](../README.md#dataset-setup)):

```text
/path/to/VISTA/test/TC01/TC01_001/TC01_001.mp4
```

The adapter first looks for `<clip_id>_blurred.mp4`, then `<clip_id>.mp4`.
Use `--csv /path/to/VISTA/test.csv` and `--dataset-root /path/to/VISTA/test`.
It samples frames across the clip, requesting eight frames by default.
Repeated frame indices are removed, so short clips may yield fewer frames.

The v5 rubric selects task-specific instructions for **scene description**,
**question answering**, or **goal guidance**.

## Scoring

Both `run` and `batch` use the **v5** rubric by default, loading
[`prompts/assistancevlmjudge_vista.json`](prompts/assistancevlmjudge_vista.json).
The rubric evaluates four dimensions:

| Dimension | Maximum score |
|---|---:|
| `evidence_grounding` | 1.75 |
| `answer_correctness` | 1.75 |
| `task_completion` | 1.00 |
| `accessible_communication` | 0.50 |

The parser bounds each component to its allowed range and rounds it to a
0.05 increment. The total is the sum of the four components, out of **5**.
When `critical_error` is true, the final score is capped at **1.95**; the
component scores and uncapped total are also recorded.

The verdict includes `meaningfully_correct`, `critical_error`, `gt_conflict`,
and `evidence_limited` flags, plus `visual_evidence` and `rationale` text.
`meaningfully_correct` is an independent judgment rather than a score threshold.
A verdict cannot set both `critical_error` and `meaningfully_correct` to true.

The four-component v5 rubric is the only supported scoring mode. The optional
`--judge-prompt v5` argument makes the selection explicit for single-file runs.
Earlier three-component verdict files are rejected on resume and aggregation;
evaluate those candidates into a new output file with v5.

## Evaluate one prediction file

```bash
python3 vlm_judge_openrouter.py run \
  --dataset dataset_vista \
  --csv /path/to/VISTA/test.csv \
  --dataset-root /path/to/VISTA/test \
  --preds /path/to/predictions_candidate.jsonl \
  --model-name candidate \
  --backend openai \
  --judge-model gpt-6-luna \
  --judge-prompt v5 \
  --rubric-path prompts/assistancevlmjudge_vista.json \
  --num-frames 8 \
  --reasoning-effort high \
  --max-output-tokens 4096 \
  --concurrency 8 \
  --api-key-file /path/to/private/openai_key.txt \
  --out out/candidate.jsonl
```

`--model-name` identifies the candidate being evaluated. The judge is GPT-6
Luna. `run` writes verdicts; use `aggregate` to produce a summary table.

## Evaluate a batch

Batch mode recursively discovers `predictions.jsonl` and
`predictions_*.jsonl` files under a `vista/` directory:

```text
predictions/
└── vista/
    ├── predictions_candidate.jsonl
    └── another_candidate/
        └── predictions.jsonl
```

Pass `--dataset-root /path/to/VISTA/test` and `--csv /path/to/VISTA/test.csv`.
Both arguments are required, including for `--dry-run`. Every candidate in the
batch uses those data paths and the dataset's v5 rubric; no tracked Python files
need path edits.

Preview the files and prediction line counts without calling the API:

```bash
python3 vlm_judge_openrouter.py batch \
  --dataset-root /path/to/VISTA/test \
  --csv /path/to/VISTA/test.csv \
  --root /path/to/predictions \
  --out-root out \
  --only vista \
  --dry-run
```

Run the batch:

```bash
python3 vlm_judge_openrouter.py batch \
  --dataset-root /path/to/VISTA/test \
  --csv /path/to/VISTA/test.csv \
  --root /path/to/predictions \
  --out-root out \
  --only vista \
  --api-key-file /path/to/private/openai_key.txt
```

Verdicts mirror the prediction directory structure under `out/vista/`.
Output filenames are `verdicts.jsonl` or `verdicts_<candidate>.jsonl`.
Candidate names are derived from the relative path. Results are aggregated
automatically after all files have been processed.

`--dry-run` is available only for `batch`. It lists the plan without loading the
reference CSV or decoding videos, so its line counts are not validated task counts.
`bash run.sh run ...` and `bash run.sh batch ...` forward arguments
to the Python script.

## Runtime options

| Option | Default | Purpose |
|---|---|---|
| `--backend` | `openai` | The only standalone API backend |
| `--judge-model` | `gpt-6-luna` | The only registered judge model |
| `--judge-prompt` | `v5` | Only supported scoring protocol; this optional flag is available only for `run` |
| `--num-frames` | `8` | Number of requested video frames per item |
| `--concurrency` | `8` | Number of concurrent evaluation workers |
| `--reasoning-effort` | `high` | Fixed to `high`; other values are rejected |
| `--max-output-tokens` | `4096` | Completion-token limit, including reasoning tokens |
| `--api-key-file` | `openai_key.txt` | Path to the API key file |

Each request sends JPEG images and the evaluation prompt to the OpenAI Chat
Completions endpoint. Requests have a 180-second timeout. Network errors and
HTTP 429/500/502/503/504 responses are retried with exponential backoff, for
up to six attempts.

The progress display reports failed evaluations and estimated cost. Cost is
calculated from API token counts using the coefficients in `OPENAI_PRICES`;
these are configured estimates, not live pricing. The counter is shared
across a batch and resets at each invocation. There is no spending cap.

## Verdicts and resume behavior

One JSONL record is appended and flushed after each completed item. Records
contain the input fields, candidate name, judge backend and model, prompt
mode, reasoning effort, token limit, number of decoded frames, parsed verdict,
and raw judge response when available.

Missing or unreadable videos produce a `no_frames` error. API and parsing
failures are recorded with `parse_error=true` and a null score.

Running the same command again skips every `item_id` already present in the
output file, **including failed items**. Resume checks v5 verdict compatibility
but does not validate the judge model, rubric file, parameters, or input files.
Use a separate output file for each
evaluation configuration.

## Aggregate results

```bash
python3 vlm_judge_openrouter.py aggregate out/candidate.jsonl
python3 vlm_judge_openrouter.py aggregate out/vista/*.jsonl
```

Pass additional paths explicitly for verdicts in nested directories.
The summary reports mean scores by candidate and question/task category,
plus an `ALL` row across that candidate's items. It also reports item counts,
parse errors, reference conflicts, and, for v5, critical errors.

Only v5 verdicts are accepted. Failed items are excluded from score averages. Aggregation does not
deduplicate records or validate matching evaluation configurations; supply
files from the same judge and rubric configuration.
Groups with only failed evaluations do not appear in the printed summary.

## Files

| File | Purpose |
|---|---|
| `vlm_judge_openrouter.py` | API requests, scoring, resume, batch processing, and aggregation |
| `dataset_vista.py` | Reference/prediction loading, video lookup, and frame sampling |
| `prompts/assistancevlmjudge_vista.json` | v5 system instruction, scoring rubric, and task guidance |
| `requirements.txt` | Python dependencies |
| `run.sh` | Shell entry point |
