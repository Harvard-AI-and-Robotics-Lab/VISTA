"""Offline regressions for the two paper-facing v5 evaluators."""

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ("VISTA_eval", "EgoBlind/EGOBLIND_eval")


def load_engine(folder):
    path = ROOT / folder / "vlm_judge_openrouter.py"
    spec = importlib.util.spec_from_file_location(f"test_{folder.replace('/', '_')}_judge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINES = {folder: load_engine(folder) for folder in FOLDERS}


def verdict(critical=False):
    return {
        "evidence_grounding": 1.71,
        "answer_correctness": 1.23,
        "task_completion": 0.92,
        "accessible_communication": 0.37,
        "meaningfully_correct": not critical,
        "critical_error": critical,
        "gt_conflict": False,
        "evidence_limited": False,
        "visual_evidence": "A doorway is visible.",
        "rationale": "The answer identifies the doorway.",
    }


def record(engine, critical=False):
    contract = engine.load_rubric_contract(engine.DEFAULT_RUBRIC_PATH)
    return {
        "item_id": "TC01_001|question_1",
        "model_name": "candidate",
        "qtype": "qa",
        "judge_prompt": "v5",
        **engine.parse_judge_output_v5(json.dumps(verdict(critical)), contract),
    }


class OpenBuffer(io.StringIO):
    def close(self):
        pass


class V5EvaluationTests(unittest.TestCase):
    def test_cli_defaults_and_removed_option(self):
        for folder, engine in ENGINES.items():
            with self.subTest(folder=folder):
                argv = ["judge", "run", "--dataset", "stub_dataset", "--csv", "refs.csv",
                        "--preds", "preds.jsonl", "--dataset-root", "videos",
                        "--model-name", "candidate", "--out", "verdicts.jsonl"]
                dataset = SimpleNamespace()
                with patch.object(sys, "argv", argv), \
                        patch.object(engine.importlib, "import_module", return_value=dataset), \
                        patch.object(engine, "run") as run:
                    engine.main()
                args = run.call_args.args[1]
                self.assertEqual(args.judge_prompt, "v5")
                self.assertEqual(args.reasoning_effort, "high")
                self.assertEqual(args.num_frames, 8)
                self.assertEqual(args.max_output_tokens, 4096)
                with patch.object(sys, "argv", argv + ["--judge-prompt", "legacy"]), \
                        contextlib.redirect_stderr(io.StringIO()), \
                        self.assertRaises(SystemExit) as caught:
                    engine.main()
                self.assertEqual(caught.exception.code, 2)

    def test_judges_reject_other_effort_before_loading_data_or_calling_api(self):
        commands = [
            ["judge", "run", "--dataset", "stub_dataset", "--csv", "refs.csv",
             "--preds", "preds.jsonl", "--dataset-root", "videos",
             "--model-name", "candidate", "--out", "verdicts.jsonl"],
            ["judge", "batch", "--root", "predictions", "--out-root", "verdicts",
             "--dataset-root", "videos", "--csv", "refs.csv"],
        ]
        for folder, engine in ENGINES.items():
            with self.subTest(folder=folder):
                for command in commands:
                    with patch.object(sys, "argv", command + ["--reasoning-effort", "low"]), \
                            patch.object(engine.importlib, "import_module") as load_data, \
                            patch.object(engine.requests, "post") as post, \
                            contextlib.redirect_stderr(io.StringIO()), \
                            self.assertRaises(SystemExit) as caught:
                        engine.main()
                    self.assertEqual(caught.exception.code, 2)
                    load_data.assert_not_called()
                    post.assert_not_called()
                with patch.object(engine.requests, "post") as post, \
                        self.assertRaisesRegex(ValueError, "must be high"):
                    engine.call_judge_openai("mock-key", "gpt-6-luna", "prompt", [],
                                             reasoning_effort="low")
                post.assert_not_called()

    def test_training_profile_rejects_other_effort(self):
        path = ROOT / "VISTA_training/vista_mm/judge_profile.py"
        spec = importlib.util.spec_from_file_location("test_training_judge_profile", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        profile = json.loads((ROOT / "VISTA_training/models.json").read_text())["judge_profile"]
        self.assertEqual(module.request_settings(profile)["reasoning"], {"effort": "high"})
        profile["reasoning"] = {"effort": "low"}
        with patch.object(Path, "read_text", return_value=json.dumps(profile)), \
                self.assertRaisesRegex(ValueError, "must be high"):
            module.load_profile("mock-profile.json")
        with self.assertRaisesRegex(ValueError, "must be high"):
            module.request_settings(profile)

    def test_four_components_quantization_and_critical_cap(self):
        for folder, engine in ENGINES.items():
            with self.subTest(folder=folder):
                normal = record(engine)
                self.assertFalse(normal["parse_error"])
                self.assertEqual(normal["score"], 4.2)
                self.assertEqual([normal[k] for k in engine.V5_COMPONENTS],
                                 [1.7, 1.25, 0.9, 0.35])
                critical = record(engine, critical=True)
                self.assertEqual(critical["score"], 1.95)
                self.assertEqual(critical["score_uncapped"], 4.2)
                contract = engine.load_rubric_contract(engine.DEFAULT_RUBRIC_PATH)
                invalid = verdict(critical=True)
                invalid["meaningfully_correct"] = True
                self.assertTrue(engine.parse_judge_output_v5(json.dumps(invalid), contract)["parse_error"])
                old = {"visual_grounding": 2, "semantic_consistency": 2, "completeness": 1}
                self.assertTrue(engine.parse_judge_output_v5(json.dumps(old), contract)["parse_error"])

    def test_resume_accepts_v5_and_rejects_old_verdicts(self):
        for folder, engine in ENGINES.items():
            with self.subTest(folder=folder):
                rows = [record(engine), {"item_id": "failed", "judge_prompt": "v5",
                                         "parse_error": True, "score": None}]
                data = "\n".join(json.dumps(row) for row in rows)
                with patch.object(engine.os.path, "exists", return_value=True), \
                        patch.object(engine, "open", return_value=io.StringIO(data), create=True):
                    self.assertEqual(engine._already_done("results.jsonl"),
                                     {"TC01_001|question_1", "failed"})
                for old in [
                    {"item_id": "old", "judge_prompt": "legacy", "score": 5},
                    {"item_id": "old", "score": 5, "visual_grounding": 2},
                ]:
                    with patch.object(engine.os.path, "exists", return_value=True), \
                            patch.object(engine, "open", return_value=io.StringIO(json.dumps(old)), create=True), \
                            self.assertRaisesRegex(ValueError, "only v5 verdicts"):
                        engine._already_done("results.jsonl")

    def test_aggregation_uses_v5_scores_and_excludes_failed_items(self):
        for folder, engine in ENGINES.items():
            with self.subTest(folder=folder):
                rows = [record(engine), record(engine, critical=True),
                        {"model_name": "candidate", "qtype": "qa", "judge_prompt": "v5",
                         "parse_error": True, "score": None}]
                stream = io.StringIO("\n".join(json.dumps(row) for row in rows))
                output = io.StringIO()
                with patch.object(engine, "open", return_value=stream, create=True), \
                        contextlib.redirect_stdout(output):
                    engine.aggregate(["results.jsonl"])
                self.assertIn("3.075", output.getvalue())
                self.assertIn("access", output.getvalue())
                for old in [{"judge_prompt": "legacy", "score": 5},
                            {"judge_prompt": "v5", "score": 5, "visual_grounding": 2}]:
                    with patch.object(engine, "open", return_value=io.StringIO(json.dumps(old)), create=True), \
                            self.assertRaises(ValueError):
                        engine.aggregate(["old-results.jsonl"])

    def test_full_run_sends_high_reasoning_images_and_v5_prompt(self):
        for folder, engine in ENGINES.items():
            with self.subTest(folder=folder):
                item = {"item_id": "item", "qtype": "qa", "question": "Where is the door?",
                        "references": ["Ahead."], "prediction": "Ahead."}
                if folder == "VISTA_eval":
                    item["task_type"] = "question_1"
                frames = [Image.new("RGB", (8, 8)) for _ in range(8)]
                dataset = SimpleNamespace(load_items=lambda *_: [item], get_frames=lambda *_: frames)
                args = SimpleNamespace(backend="openai", judge_model=None, api_key_file="unused-key-file",
                                       reasoning_effort="high", max_output_tokens=4096, judge_prompt="v5",
                                       rubric_path=engine.DEFAULT_RUBRIC_PATH, csv="refs.csv", preds="preds.jsonl",
                                       out="mock-results.jsonl", model_name="candidate-identity-only",
                                       num_frames=8, concurrency=1)
                response = Mock(status_code=200)
                response.json.return_value = {
                    "choices": [{"message": {"content": json.dumps(verdict())}}], "usage": {}}
                written = OpenBuffer()
                real_open = open

                def fake_open(path, mode="r", *a, **kw):
                    if str(path) == args.out:
                        return written
                    return real_open(path, mode, *a, **kw)

                with patch.object(engine, "_read_api_key", return_value="offline-test-key"), \
                        patch.object(engine, "_already_done", return_value=set()), \
                        patch.object(engine.os, "makedirs"), \
                        patch.object(engine, "open", side_effect=fake_open, create=True), \
                        patch.object(engine.requests, "post", return_value=response) as post, \
                        contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    engine.run(dataset, args)
                payload = post.call_args.kwargs["json"]
                self.assertEqual(payload["model"], "gpt-6-luna")
                self.assertEqual(payload["reasoning_effort"], "high")
                self.assertEqual(payload["max_completion_tokens"], 4096)
                content = payload["messages"][-1]["content"]
                self.assertEqual(sum(part["type"] == "image_url" for part in content), 8)
                self.assertIn("evidence_grounding", content[-1]["text"])
                self.assertNotIn(args.model_name, content[-1]["text"])
                saved = json.loads(written.getvalue())
                self.assertEqual(saved["judge_prompt"], "v5")
                self.assertEqual(saved["score"], 4.2)

    def test_provenance_and_removed_adapter_prompts(self):
        for folder, engine in ENGINES.items():
            with self.subTest(folder=folder):
                directory = ROOT / folder
                path = directory / "vlm_judge_openrouter.py"
                source = path.read_text(encoding="utf-8")
                meta = json.loads((directory / "PROVENANCE.json").read_text(encoding="utf-8"))
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), meta["packaged_engine_sha256"])
                self.assertFalse(hasattr(engine, "parse_judge_output"))
                for node in ast.parse(source).body:
                    if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        continue
                    expected = meta["unchanged_function_and_class_sha256"].get(node.name)
                    if expected is None:
                        expected = meta["added_function_and_class_sha256"].get(node.name)
                    if expected is None:
                        expected = meta["modified_function_and_class_sha256"][node.name]["packaged_sha256"]
                    self.assertEqual(hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest(), expected)
                for adapter, details in meta["modified_files"].items():
                    path = directory / adapter
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), details["packaged_sha256"])
                    if path.suffix == ".py":
                        self.assertNotIn("build_prompt", [n.name for n in ast.parse(path.read_text(encoding="utf-8")).body
                                                          if isinstance(n, ast.FunctionDef)])
                for name, details in meta["byte_identical_files"].items():
                    self.assertEqual(hashlib.sha256((directory / name).read_bytes()).hexdigest(), details["sha256"])

    def test_training_profile_is_high_and_archived_configs_are_preserved(self):
        directory = ROOT / "VISTA_training"
        database = json.loads((directory / "models.json").read_text())
        meta = json.loads((directory / "provenance.json").read_text())
        self.assertEqual(database["judge_profile"]["reasoning"]["effort"], "high")
        self.assertEqual(meta["judge"]["reasoning_effort"], "high")
        self.assertEqual(meta["judge"]["format_correction_retries"], 0)
        self.assertEqual(hashlib.sha256((directory / "models.json").read_bytes()).hexdigest(), meta["models_sha256"])
        for name, model in database["models"].items():
            for stage, config in model["configs"].items():
                encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
                self.assertEqual(hashlib.sha256(encoded).hexdigest(), meta["config_digests"][name][stage])


if __name__ == "__main__":
    unittest.main()
