# Retired benchmark artifacts

The cleanup on 2026-10-12 retired `dataset_check.csv` and six historical prediction
files. Each prediction file has 2,041 items from obsolete annotation identities;
these files are unsuitable for the current 450-clip, 1,828-task test split.

| Prediction file | Archived rows |
| --- | ---: |
| `predictions_internvl3_1b.jsonl` | 2,041 |
| `predictions_internvl3_2b.jsonl` | 2,041 |
| `predictions_internvl3_8b.jsonl` | 2,041 |
| `predictions_qwen3vl_2b.jsonl` | 2,041 |
| `predictions_qwen3vl_4b.jsonl` | 2,041 |
| `predictions_qwen3vl_8b.jsonl` | 2,041 |

The retired checklist, predictions, and six empty error logs are recoverable from the
[last tracked snapshot](https://github.com/Harvard-AI-and-Robotics-Lab/VISTA/tree/4961690be6beaeae860db69d7916ad522451ad9d/VISTA_benchmark).
The placeholder `categories/tags.jsonl` was removed: all 248 records had `Unknown`
task type, environment, and danger level.

Verified replacement predictions have not been supplied. Before publishing a new
result bundle, check exact task identity coverage against its test CSV, duplicates,
nonempty answers, the error log, and the recorded inference settings. Keep data
revision and run metadata with the bundle.
