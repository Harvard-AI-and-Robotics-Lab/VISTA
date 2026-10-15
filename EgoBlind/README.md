# EgoBlind Benchmark

EgoBlind provides an additional egocentric question-answering benchmark with setup
separate from VISTA.

| Directory | Purpose |
| --- | --- |
| [EGOB_benchmark](EGOB_benchmark/README.md) | Inference, model backends, and question prompts |
| [EGOBLIND_eval](EGOBLIND_eval/README.md) | GPT-6 Luna judge and dataset adapter |

It supports the [VISTA model keys](../README.md#supported-models) except
`internvl3_9b`, `gemma3_12b`, and `gemma3_27b`. See the benchmark README for
environment setup and [sampling details](EGOB_benchmark/README.md#sampling-and-decoding).

## Dataset Setup

Provide the external test CSV and videos:

```text
EgoBlind/Test/
|-- test.csv
|-- split_0/
|   `-- 00923.mp4
`-- ... split_9/
```

## Inference

From the repository root, with the selected model environment active:

```bash
cd EgoBlind/EGOB_benchmark
python run.py --dataset_root /path/to/EgoBlind/Test --csv_path /path/to/EgoBlind/Test/test.csv --output_dir outputs/my_run --model gemini_3_1_flash_lite
```

Add `--limit 10` for a smoke test. Outputs use `predictions_<model>.jsonl` and
`errors_<model>.jsonl`.

## Evaluation

From the repository root, with the evaluation dependencies installed:

```bash
cd EgoBlind/EGOBLIND_eval
python vlm_judge_openrouter.py run --dataset dataset_egoblind --dataset-root /path/to/EgoBlind/Test --csv /path/to/EgoBlind/Test/test.csv --preds /path/to/predictions_mymodel.jsonl --model-name mymodel --api-key-file /path/to/openai_key.txt --out out/mymodel.jsonl
python vlm_judge_openrouter.py aggregate out/mymodel.jsonl
```
