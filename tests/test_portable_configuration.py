"""Offline regressions for explicit batch paths and the Hub-layout launcher."""

import contextlib
import io
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_v5_evaluation import ENGINES, ROOT


BASH = (str(Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe")
        if os.name == "nt" else shutil.which("bash"))
if BASH and not Path(BASH).is_file():
    BASH = None


class PortableConfigurationTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / "VISTA_benchmark/outputs"
        scratch.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="portable-configuration-", dir=scratch)
        self.directory = Path(temporary.name).resolve()
        self.assertTrue(self.directory.is_relative_to(scratch.resolve()))
        self.addCleanup(temporary.cleanup)

    def batch_fixture(self, key, dry_run=False):
        predictions = self.directory / "predictions" / key
        predictions.mkdir(parents=True, exist_ok=True)
        (predictions / "predictions_candidate_a.jsonl").write_text("{}\n", encoding="utf-8")
        nested = predictions / "candidate_b"
        nested.mkdir(exist_ok=True)
        (nested / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(
            root=str(predictions.parent), out_root=str(self.directory / "verdicts"),
            only=key, dataset_root=str(self.directory / "dataset videos"),
            csv=str(self.directory / "references.csv"), backend="openai",
            judge_model=None, reasoning_effort="high", num_frames=8,
            concurrency=2, max_output_tokens=4096, api_key_file="mock-key.txt",
            dry_run=dry_run,
        )

    def test_batch_cli_requires_both_data_paths(self):
        base = ["judge", "batch", "--root", "predictions", "--out-root", "verdicts"]
        for folder, engine in ENGINES.items():
            with self.subTest(folder=folder):
                for supplied in [[], ["--csv", "refs.csv"], ["--dataset-root", "videos"]]:
                    with patch.object(sys, "argv", base + supplied), \
                            contextlib.redirect_stderr(io.StringIO()), \
                            self.assertRaises(SystemExit) as error:
                        engine.main()
                    self.assertEqual(error.exception.code, 2)
                argv = base + ["--dataset-root", "videos", "--csv", "refs.csv", "--dry-run"]
                with patch.object(sys, "argv", argv), patch.object(engine, "run_batch") as run:
                    engine.main()
                args = run.call_args.args[0]
                self.assertEqual(args.dataset_root, "videos")
                self.assertEqual(args.csv, "refs.csv")
                self.assertTrue(args.dry_run)

    def test_batch_passes_data_paths_to_every_candidate(self):
        for folder, engine in ENGINES.items():
            with self.subTest(folder=folder):
                args = self.batch_fixture(next(iter(engine.DATASET_CONFIGS)))
                dataset = SimpleNamespace()
                with patch.object(engine, "_read_api_key", return_value="mock-key"), \
                        patch.object(engine.importlib, "import_module", return_value=dataset), \
                        patch.object(engine, "run") as run, \
                        patch.object(engine, "aggregate") as aggregate, \
                        contextlib.redirect_stdout(io.StringIO()):
                    engine.run_batch(args)
                self.assertEqual(run.call_count, 2)
                self.assertEqual(dataset.DATASET_ROOT, args.dataset_root)
                trackers = []
                for call in run.call_args_list:
                    loaded, candidate, tracker = call.args
                    self.assertIs(loaded, dataset)
                    self.assertEqual(candidate.dataset_root, args.dataset_root)
                    self.assertEqual(candidate.csv, args.csv)
                    self.assertEqual(candidate.judge_prompt, "v5")
                    trackers.append(tracker)
                self.assertIs(trackers[0], trackers[1])
                self.assertEqual(len(aggregate.call_args.args[0]), 2)

    def test_batch_preview_avoids_keys_dataset_loading_and_api_calls(self):
        for folder, engine in ENGINES.items():
            with self.subTest(folder=folder):
                args = self.batch_fixture(next(iter(engine.DATASET_CONFIGS)), dry_run=True)
                with patch.object(engine, "_read_api_key") as read_key, \
                        patch.object(engine.importlib, "import_module") as import_dataset, \
                        patch.object(engine, "run") as run, \
                        contextlib.redirect_stdout(io.StringIO()):
                    engine.run_batch(args)
                read_key.assert_not_called()
                import_dataset.assert_not_called()
                run.assert_not_called()

    @unittest.skipUnless(BASH, "Bash is required to exercise the launcher")
    def test_launcher_derives_test_paths_from_one_dataset_folder(self):
        def bash_path(path):
            if os.name == "nt":
                return subprocess.check_output(
                    [BASH, "--noprofile", "--norc", "-c", 'cygpath -u "$1"', "bash", str(path)],
                    text=True,
                ).strip()
            return path.as_posix()

        repository = self.directory / "repository"
        benchmark = repository / "VISTA_benchmark"
        interpreter = benchmark / "venvs/venv_qwen/bin/python"
        interpreter.parent.mkdir(parents=True)
        interpreter.write_text('#!/usr/bin/env bash\nprintf "ARG:%s\\n" "$@"\n', encoding="utf-8")
        interpreter.chmod(0o755)
        shutil.copy2(ROOT / "VISTA_benchmark/run.bash", benchmark / "run.bash")
        dataset = self.directory / "downloaded VISTA"
        (dataset / "test").mkdir(parents=True)
        csv = dataset / "test.csv"
        csv.write_text("clip_id\nTC01_001\n", encoding="utf-8")
        (repository / ".env").write_text(
            "VISTA_DATASET_DIR=" + shlex.quote(bash_path(dataset)) + "\n"
            "HF_TOKEN=\nOPENAI_API_KEY=\nGEMINI_API_KEY=\n",
            encoding="utf-8",
        )
        env = dict(os.environ)
        env.pop("VISTA_DATASET_DIR", None)
        for old in ["VISTA_BASE", "VISTA_DATASET_ROOT", "VISTA_CSV_PATH", "VISTA_OUTPUT_DIR", "VISTA_ENVS_DIR"]:
            env[old] = "/obsolete/configuration"
        command = [BASH, "--noprofile", "--norc", str(benchmark / "run.bash"), "gemini_3_1_flash_lite", "5"]
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        forwarded = [line[4:] for line in result.stdout.splitlines() if line.startswith("ARG:")]
        options = dict(zip(forwarded[1::2], forwarded[2::2]))
        self.assertEqual(options["--dataset_root"], bash_path(dataset / "test"))
        self.assertEqual(options["--csv_path"], bash_path(csv))
        self.assertEqual(options["--output_dir"], bash_path(benchmark / "outputs"))
        self.assertEqual(options["--model"], "gemini_3_1_flash_lite")
        self.assertEqual(options["--limit"], "5")
        csv.unlink()
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must contain test/ and test.csv", result.stdout)
        self.assertNotIn("ARG:", result.stdout)


if __name__ == "__main__":
    unittest.main()
