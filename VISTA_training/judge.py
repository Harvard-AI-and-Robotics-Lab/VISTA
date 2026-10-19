import argparse
import concurrent.futures as cf
import fcntl
import json
import math
import time
from pathlib import Path

from vista_mm.config import Experiment
from vista_mm.data import VistaDataset
from vista_mm.evaluate import task_group
from vista_mm.judge3 import Judge3Rubric, PAPER_ENGINE_PATH, PAPER_DATASET_PATH
from vista_mm.judge_profile import load_profile, request_settings
from vista_mm.openrouter_judge import Client, read_keys
from vista_mm.provenance import atomic_json, digest, read_jsonl, sha256


def mean(values):
    return sum(values) / len(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--csv", help="Original test annotation CSV for the paper judge")
    parser.add_argument("--split", choices=["test"], default="test")
    parser.add_argument("--rubric", required=True)
    parser.add_argument("--judge-profile", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--key-files", nargs="+", required=True)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--max-cost-usd", type=float, default=5)
    parser.add_argument("--model-label")
    parser.add_argument(
        "--runtime-settings",
        type=Path,
        default=Path(__file__).with_name("judge_runtime.json"),
        help="Optional JSON overrides for key_files and concurrency, loaded at judge launch",
    )
    args = parser.parse_args()
    if args.runtime_settings.is_file():
        settings = json.loads(args.runtime_settings.read_text())
        if not isinstance(settings, dict) or set(settings) - {
            "key_files",
            "concurrency",
        }:
            parser.error("Runtime settings may only contain key_files and concurrency")
        if "key_files" in settings:
            paths = settings["key_files"]
            if (
                not isinstance(paths, list)
                or not paths
                or not all(isinstance(p, str) for p in paths)
            ):
                parser.error("Runtime key_files must be a nonempty list of paths")
            args.key_files = paths
        if "concurrency" in settings:
            if type(settings["concurrency"]) is not int:
                parser.error("Runtime concurrency must be an integer")
            args.concurrency = settings["concurrency"]
    if (
        not 1 <= args.concurrency <= 16
        or not math.isfinite(args.max_cost_usd)
        or args.max_cost_usd <= 0
    ):
        parser.error("bad concurrency or budget")

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    profile = load_profile(args.judge_profile)
    rubric = Judge3Rubric(args.rubric)
    config = Experiment.load(args.config)
    config.sensor = {"modalities": []}
    dataset = VistaDataset(config.prepared, args.split, config)
    if dataset.manifest.get("smoke_fixture") or config.num_frames != 8:
        raise ValueError("Requires a complete prepared split and eight frames")
    expected = {r["item_id"]: r for r in dataset.items}
    if len(expected) != len(dataset.items):
        raise ValueError("Duplicate split item")

    pred_path = Path(args.predictions)
    rows = read_jsonl(pred_path)
    predictions = {r["item_id"]: r for r in rows}
    if len(rows) != len(expected) or set(predictions) != set(expected):
        raise ValueError("Incomplete or duplicate predictions")
    if args.csv:
        annotation_csv = Path(args.csv).resolve()
    else:
        data_root = Path(dataset.manifest["dataset_root"])
        annotation_csv = next(
            (p for p in (data_root / f"{args.split}.csv",
                         data_root / args.split / f"{args.split}.csv") if p.is_file()),
            data_root / f"{args.split}.csv",
        )
    if sha256(annotation_csv) != dataset.manifest["annotation_sha256"][args.split]:
        raise ValueError("Annotation CSV differs from the prepared dataset")
    paper_items = rubric.load_items(annotation_csv, pred_path)
    if len(paper_items) != len(expected) or {r["item_id"] for r in paper_items} != set(expected):
        raise ValueError("Paper adapter and prepared split contain different tasks")
    prepared_indices = {item["item_id"]: i for i, item in enumerate(dataset.items)}
    complete = json.loads(pred_path.with_suffix(".complete.json").read_text())
    original = json.loads(pred_path.with_suffix(".protocol.json").read_text())
    model_label = args.model_label or (
        "trained model" if original.get("checkpoint_hashes") else "pretrained model"
    )
    if (
        complete["prediction_sha256"] != sha256(pred_path)
        or original["prepared_id"] != dataset.manifest["prepared_id"]
        or original["split"] != args.split
    ):
        raise ValueError("Prediction provenance mismatch")

    protocol = {
        **profile,
        "rubric_version": rubric.contract["version"],
        "rubric_sha256": sha256(args.rubric),
        "score_range": [0, 5],
        "prepared_id": dataset.manifest["prepared_id"],
        "split": args.split,
        "items": len(expected),
        "items_sha256": sha256(Path(config.prepared) / f"{args.split}.items.jsonl"),
        "predictions_sha256": {"new_model": sha256(pred_path)},
        "config_sha256": sha256(args.config),
        "judge_profile_sha256": sha256(args.judge_profile),
        "key_count": len(read_keys(args.key_files)),
        "concurrency": args.concurrency,
        "max_cost_usd": args.max_cost_usd,
        "num_frames": 8,
        "min_pixels": config.min_pixels,
        "max_pixels": config.max_pixels,
        "image_encoding": "jpeg-quality85-native-resolution",
        "frame_sampling": "same eight prepared frame indices as the evaluated model",
        "annotation_sha256": sha256(annotation_csv),
        "paper_engine_sha256": sha256(PAPER_ENGINE_PATH),
        "paper_dataset_sha256": sha256(PAPER_DATASET_PATH),
        "judging_policy": "one_verdict_no_format_corrections",
        "format_correction_retries": 0,
        "scope": f"Paper VISTA_eval prompt and parser for one completed {model_label} prediction set.",
    }
    protocol_path = root / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Resume protocol differs")
    else:
        atomic_json(protocol_path, protocol)
        atomic_json(root / "rubric.json", rubric.contract)
    identity = digest(protocol)
    client = Client(read_keys(args.key_files), args.max_cost_usd, root / "usage.jsonl")

    def judge(index):
        item = paper_items[index]
        folder = root / "new_model" / "items"
        folder.mkdir(parents=True, exist_ok=True)
        token = digest(item["item_id"])
        saved = folder / f"{token}.json"
        if saved.exists():
            result = json.loads(saved.read_text())
            if result["protocol_identity"] != identity:
                raise ValueError("Saved verdict protocol mismatch")
            try:
                parsed = rubric.parse(result["raw_output"])
            except Exception as exc:
                parsed = {"parse_error": True, "score": None, "exception": str(exc)}
            if any(result[k] != v for k, v in parsed.items()):
                raise ValueError("Saved verdict differs from raw response")
            return result
        example = dataset[prepared_indices[item["item_id"]]]
        payload = {
            **request_settings(profile),
            "messages": [
                {"role": "system", "content": rubric.contract["system"]},
                {
                    "role": "user",
                    "content": rubric.image_content(example["frames"])
                    + [
                        {
                            "type": "text",
                            "text": rubric.prompt(
                                item, predictions[item["item_id"]]["prediction"]
                            ),
                        }
                    ],
                },
            ],
        }
        response_path = folder / f"{token}.response.json"
        if response_path.exists():
            response = json.loads(response_path.read_text())
        else:
            atomic_json(folder / f"{token}.request.json", {"payload_sha256": digest(payload)})
            response = client.complete(
                payload,
                index % max(1, len(client.keys)),
                f"new_model:{item['item_id']}:single_verdict",
            )
            atomic_json(response_path, response)
        if response.get("model") != profile["model"]:
            raise ValueError("Wrong judge model; inspect saved response")
        raw = response.get("choices", [{}])[0].get("message", {}).get("content")
        try:
            verdict = rubric.parse(raw)
        except Exception as exc:
            verdict = {"parse_error": True, "score": None, "exception": str(exc)}
        result = {
            **{k: item[k] for k in ("item_id", "clip_id", "task_type", "question", "references")},
            **verdict,
            "model_name": "new_model",
            "raw_output": raw,
            "protocol_identity": identity,
            "request_id": response.get("id"),
            "usage": response.get("usage"),
            "provider": response.get("provider"),
            "attempt": 1,
        }
        atomic_json(saved, result)
        return result

    started = time.monotonic()
    results = {}
    with cf.ThreadPoolExecutor(args.concurrency) as pool:
        indices = iter(range(len(dataset)))
        pending = {
            pool.submit(judge, i) for i in range(min(args.concurrency, len(dataset)))
        }
        for _ in range(min(args.concurrency, len(dataset))):
            next(indices)
        try:
            while pending:
                finished, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                for future in finished:
                    row = future.result()
                    results[row["item_id"]] = row
                    done = len(results)
                    atomic_json(
                        root / "progress.json",
                        {
                            "status": "judging",
                            "completed": done,
                            "total": len(dataset),
                            "cost_usd": client.spent,
                        },
                    )
                    if done % 20 == 0 or done == len(dataset):
                        print(
                            json.dumps(
                                {
                                    "completed": done,
                                    "total": len(dataset),
                                    "cost_usd": client.spent,
                                }
                            ),
                            flush=True,
                        )
                for _ in finished:
                    index = next(indices, None)
                    if index is not None:
                        pending.add(pool.submit(judge, index))
        except BaseException:
            with client.lock:
                client.stop = True
            for future in pending:
                future.cancel()
            raise
    if set(results) != set(expected):
        raise ValueError("Incomplete final scores")
    out = root / "new_model" / f"{args.split}.scores.jsonl"
    out.write_text(
        "".join(json.dumps(results[k]) + "\n" for k in sorted(expected)),
        encoding="utf-8",
    )
    atomic_json(
        out.with_suffix(".complete.json"),
        {
            "items": len(results),
            "scores_sha256": sha256(out),
            "protocol_identity": identity,
        },
    )
    scores = [row for row in results.values() if not row.get("parse_error") and row.get("score") is not None]
    by_group = {}
    for row in scores:
        by_group.setdefault(task_group(row["task_type"]), []).append(row)
    summary = {
        "status": "complete",
        "judge": profile["model"],
        "split": args.split,
        "rubric": Path(args.rubric).name,
        "protocol_identity": identity,
        "items": len(results),
        "scored_items": len(scores),
        "failed_items": len(results) - len(scores),
        "cost_usd": client.spent,
        "elapsed_seconds": time.monotonic() - started,
        "overall": {
            "score": mean([r["score"] for r in scores]),
            "meaningfully_correct_percent": 100
            * mean([r["meaningfully_correct"] for r in scores]) if scores else None,
        },
        "by_group": {
            g: {
                "items": len(v),
                "score": mean([r["score"] for r in v]),
                "meaningfully_correct_percent": 100
                * mean([r["meaningfully_correct"] for r in v]),
            }
            for g, v in sorted(by_group.items())
        },
        "limitations": [
            "Paper VISTA_eval single-verdict scoring; malformed verdicts are logged and excluded without correction requests."
        ],
    }
    atomic_json(root / "summary.json", summary)
    atomic_json(
        root / "progress.json",
        {
            "status": "complete",
            "completed": len(results),
            "scored": len(scores),
            "failed": len(results) - len(scores),
            "total": len(dataset),
            "cost_usd": client.spent,
        },
    )
    print(json.dumps(summary), flush=True)
    lock.close()


if __name__ == "__main__":
    main()
