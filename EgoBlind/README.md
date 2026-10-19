# EgoBlind Benchmark

Run the EgoBlind question-answering benchmark. Use Linux/Bash and Python 3.12;
local models require CUDA. Run commands from the repository root.

## Dataset Setup

Supply the external test CSV and videos:

```text
/path/to/EgoBlind/Test/
  test.csv
  split_0/00923.mp4
  ... split_9/01578.mp4
```

Inference needs `question_id`, `video_name`, `question`, and `start-time/s`.
Evaluation also needs `answer0` through `answer3`; text metrics need `type`.
Video numbers map to five-digit filenames under the split folders.

## Model Setup

Use the [main CLI keys](../README.md#supported-models), excluding `internvl3_9b`,
`gemma3_12b`, and `gemma3_27b`. Install one matching stack from `EgoBlind/EGOB_benchmark/`:

| Models | Requirements |
|---|---|
| Qwen3-VL | `requirements/requirements_qwen.txt` |
| InternVL3 and Gemma 3 4B | `requirements/requirements_internvl3.txt` |
| VideoLLaMA3 | `requirements/requirements_videollama.txt` |
| Gemini and GPT Realtime Mini | `requirements.txt` |

InternVL example:

```bash
python3.12 -m venv EgoBlind/EGOB_benchmark/venvs/venv_internvl3
source EgoBlind/EGOB_benchmark/venvs/venv_internvl3/bin/activate
python -m pip install -r EgoBlind/EGOB_benchmark/requirements/requirements_internvl3.txt
```

Hosted models also require PyTorch, included in their requirements. LLaVA-Video
needs a compatible [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) installation
plus `EgoBlind/EGOB_benchmark/requirements.txt`.

Set `OPENAI_API_KEY` or `GEMINI_API_KEY` for hosted models and `HF_TOKEN` for gated
weights. Inference loads `.env` with `override=True`, replacing matching shell variables.

## Inference

```bash
python EgoBlind/EGOB_benchmark/run.py \
  --dataset_root /path/to/EgoBlind/Test \
  --csv_path /path/to/EgoBlind/Test/test.csv \
  --output_dir EgoBlind/EGOB_benchmark/outputs/my_run \
  --model internvl3_1b --limit 5
```

Remove `--limit 5` for all questions. Outputs are `predictions_<model>.jsonl`
(with `question_id` and `pred`) and `errors_<model>.jsonl`. Rerunning skips saved
question IDs; use a new output directory when changing data or settings.

Frames cover the video start to the question timestamp. At timestamps mapping
to frame zero or below, inference uses the full clip while the judge uses frame
zero. Qwen and VideoLLaMA3 use their default decoding; other local models use
greedy decoding. See [config.py](EGOB_benchmark/config.py) and
[models.py](EGOB_benchmark/models.py) for frame and model settings.

## Text Metrics

In the inference environment:

```bash
python -m pip install -r EgoBlind/EGOB_benchmark/requirements/requirements_metrics.txt
python EgoBlind/EGOB_benchmark/eval.py \
  --test_path /path/to/EgoBlind/Test/test.csv \
  --outputs_dir EgoBlind/EGOB_benchmark/outputs
```

This scores prediction files one folder below `--outputs_dir`, writing
`summary_<folder>.csv` with overall and `type`-grouped BLEU, ROUGE, SBERT, and
BERTScore. Use `--pred_path /path/to/predictions_model.jsonl` for one file.
Metric models download on first use.

## Judge Evaluation

Create a separate judge environment and supply an OpenAI API key file:

```bash
python3 -m venv EgoBlind/EGOBLIND_eval/.venv
source EgoBlind/EGOBLIND_eval/.venv/bin/activate
python -m pip install -r EgoBlind/EGOBLIND_eval/requirements.txt

python EgoBlind/EGOBLIND_eval/vlm_judge_openrouter.py run \
  --dataset dataset_egoblind \
  --csv /path/to/EgoBlind/Test/test.csv --dataset-root /path/to/EgoBlind/Test \
  --preds EgoBlind/EGOB_benchmark/outputs/my_run/predictions_internvl3_1b.jsonl \
  --model-name internvl3_1b \
  --api-key-file /path/to/private/openai_key.txt \
  --out EgoBlind/EGOBLIND_eval/out/internvl3_1b.jsonl

python EgoBlind/EGOBLIND_eval/vlm_judge_openrouter.py aggregate EgoBlind/EGOBLIND_eval/out/internvl3_1b.jsonl
```

The judge uses GPT-6 Luna with high reasoning effort and the
[EgoBlind rubric](EGOBLIND_eval/prompts/assistancevlmjudge_egoblind.json).
It reads the key file, not `.env`. Prediction IDs must match the CSV values
and types; unmatched questions are skipped. See the shared
[scoring](../VISTA_eval/README.md#scoring) and
[resume behavior](../VISTA_eval/README.md#outputs-and-resuming).

For batch judging, place predictions below `/path/to/predictions/egoblind/`:

```bash
python EgoBlind/EGOBLIND_eval/vlm_judge_openrouter.py batch \
  --root /path/to/predictions --only egoblind \
  --csv /path/to/EgoBlind/Test/test.csv --dataset-root /path/to/EgoBlind/Test \
  --out-root EgoBlind/EGOBLIND_eval/out \
  --api-key-file /path/to/private/openai_key.txt --dry-run
```

Remove `--dry-run` to judge and aggregate. This shares VISTA's
[batch behavior](../VISTA_eval/README.md#batch-evaluation). Use `--help` for more options.
