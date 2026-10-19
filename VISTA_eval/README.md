# VISTA Evaluation

Score predictions with GPT-6 Luna, high reasoning effort, and the paper's v5
rubric. This script calls OpenAI directly; the [training evaluator](../VISTA_training/README.md#evaluation)
uses OpenRouter.

## Installation

Follow the [evaluation setup](../README.md#vlm-as-a-judge-evaluation) and set
`OPENAI_API_KEY` in the repository-root `.env` or your shell. Shell variables
have priority; `--api-key-file` overrides both. Run commands from the repository root.

## Input Data

Use the [downloaded dataset](../README.md#dataset-setup): `test.csv` supplies
references and `--dataset-root /path/to/VISTA/test` supplies videos.
Each prediction JSONL row has this form:

```json
{"clip_id": "TC01_001", "task_type": "question_1", "prediction": "The door is on the left."}
```

Task types are `scene_description`, `question_i`, and `goal_i`. Matching uses
`(clip_id, task_type)`; tasks without references or predictions are skipped.

## Evaluate and Aggregate

Run the [single-file example](../README.md#vlm-as-a-judge-evaluation) with your
prediction path and candidate name, then aggregate:

```bash
python VISTA_eval/vlm_judge_openrouter.py aggregate VISTA_eval/out/candidate.jsonl
```

`--model-name` labels the candidate, not the judge. `--concurrency` sets parallel
requests (default: eight). Judging incurs API charges; displayed costs are
estimates, with no spending cap.

## Scoring

The [rubric](prompts/assistancevlmjudge_vista.json) defines four components:

| Dimension | Maximum |
|---|---:|
| Evidence grounding | 1.75 |
| Answer correctness | 1.75 |
| Task completion | 1.00 |
| Accessible communication | 0.50 |

Components are rounded to 0.05 increments and summed to 0-5. A critical error
caps the score at 1.95. `meaningfully_correct` is an independent judgment,
not a score threshold. Only v5 verdicts are supported.

## Outputs and Resuming

Verdicts are saved as JSONL; failed items have a null score. Rerunning skips
saved items, including failures. Use a new output file when data or judge
settings change. Aggregation excludes failed items from averages, reports
results by candidate and task category, and does not deduplicate records.

## Batch Evaluation

Place `predictions.jsonl` or `predictions_*.jsonl` anywhere below
`/path/to/predictions/vista/`, then preview:

```bash
python VISTA_eval/vlm_judge_openrouter.py batch \
  --root /path/to/predictions --only vista \
  --csv /path/to/VISTA/test.csv --dataset-root /path/to/VISTA/test \
  --out-root VISTA_eval/out --dry-run
```

Remove `--dry-run` to judge and aggregate. Verdicts mirror the input folders
under `VISTA_eval/out/vista/`. The preview counts lines without validating tasks
or decoding videos. Use `run --help` or `batch --help` for other options.
For EgoBlind, follow its [judge instructions](../EgoBlind/README.md#judge-evaluation).
