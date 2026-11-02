"""Offline MerchantBench integration regressions; no model/provider calls."""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from harness2 import config
from harness2.benchmarks.ecommerce import EcomAdapter, load_tasks
from harness2.benchmarks.merchantbench_runtime import LocalEpisode, request_json
from harness2.benchmarks.tasks import discover
from harness2.cli import _common, run_sweep, setup
from harness2.cli import judge as judge_cli
from harness2.core import context, report, runmark, tree
from harness2.core.judge import Scored
from harness2.core.spec import RunView


class EcomTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "tasks.json"
        self.task = {
            "id": "synthetic/seed42",
            "domain": "synthetic",
            "master_seed": 42,
            "horizon_steps": 24,
        }
        self.write_tasks([self.task])
        self.adapter = EcomAdapter(tasks_file=self.manifest, store_root=self.root / "episodes")

    def write_tasks(self, tasks):
        self.manifest.write_text(json.dumps({"version": 1, "tasks": tasks}))

    def make_scored_run(self, record=None):
        out = self.root / "run"
        (out / "measurement").mkdir(parents=True)
        record = record or {
            "task_id": self.task["id"],
            "terminal_status": "finished",
            "master_seed": 42,
            "horizon_steps": 24,
            "merchant_section": {},
        }
        (out / "measurement/episode.json").write_text(json.dumps(record))
        (out / "trajectory.jsonl").write_text('{"type":"text","part":{"text":"operated"}}\n')
        (out / "operations.jsonl").write_text('{"kind":"terminal","payload":{"terminal":true}}\n')
        runmark.mark_complete(
            out,
            run_id="test",
            extra={
                "artifact_fingerprint": self.adapter._artifact_fingerprint(out),
            },
        )
        return out

    def test_discovery_and_cli_dry_run_without_credentials(self):
        with patch.object(config, "ECOM_TASKS_FILE", self.manifest):
            self.assertEqual(discover("ecom"), {"synthetic/seed42": "synthetic"})
            self.assertIsInstance(_common.make_adapter("ecom", None, "oc"), EcomAdapter)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                code = run_sweep.main(["--bench", "ecom", "--max-tasks", "1", "--dry-run"])
            self.assertEqual(code, 0)
            self.assertIn("synthetic/seed42", output.getvalue())
        self.assertEqual(self.adapter.task_files(self.task["id"]), None)

    def test_manifest_validation(self):
        for bad in [
            [self.task, self.task],
            [{**self.task, "id": "../seed42"}],
            [{**self.task, "master_seed": True}],
            [{**self.task, "master_seed": -1}],
            [{**self.task, "horizon_steps": 0}],
            [{**self.task, "horizon_steps": True}],
            [{**self.task, "domain": "other"}],
            [{**self.task, "scenario": "../private"}],
            [],
        ]:
            with self.subTest(tasks=bad):
                self.write_tasks(bad)
                with self.assertRaises(ValueError):
                    load_tasks(self.manifest)

    def test_split_selection(self):
        split = self.root / "split.json"
        split.write_text(json.dumps({"benchmark": "ecom", "train": [], "test": [self.task["id"]]}))
        adapter = EcomAdapter(tasks_file=self.manifest, split_file=split)
        self.assertEqual(adapter.tasks("train"), [])
        self.assertEqual(adapter.tasks("test"), [self.task["id"]])
        with self.assertRaises(ValueError):
            self.adapter.tasks("test")

    def test_operating_horizon_is_not_the_settlement_cutoff(self):
        prompt = self.adapter.task_prompt(self.task["id"])
        self.assertIn("order-drain period", prompt)
        self.assertIn("settled after the operating horizon still affects", prompt)
        for substrate in ("oc", "cx"):
            adapter = EcomAdapter(substrate)
            contents = "\n".join(tree.read_tree(adapter.substrate.surface(), adapter.base_harness()).values())
            self.assertIn("drain automatically afterward", contents)
            self.assertIn("last\nagent-visible snapshot is not the final score", contents)

    def test_base_surfaces_and_transport_immutability(self):
        for substrate in ("oc", "cx"):
            adapter = EcomAdapter(substrate)
            surface = adapter.substrate.surface()
            self.assertTrue(tree.is_tree(surface, adapter.base_harness()))
            adapter.validate_tree(tree.read_tree(surface, adapter.base_harness()))
            with self.assertRaises(ValueError):
                adapter.validate_tree({"scripts/merchantbench.py": "pass"})
            install = adapter.substrate.materialise(adapter.base_harness(), adapter.spec, {})
            self.addCleanup(shutil.rmtree, install.staging_dir, True)
        self.assertNotEqual(
            self.adapter._episode_dir("a/b", "x/y"), self.adapter._episode_dir("a/b", "x__y")
        )

    @patch.object(EcomAdapter, "check_environment")
    def test_native_metric_keeps_zero_negative_and_large_values(self, _check):
        out = self.make_scored_run()
        for value in (0.0, -123.45, 10500.25):
            with (
                self.subTest(value=value),
                patch(
                    "harness2.benchmarks.ecommerce.subprocess.run",
                    return_value=Mock(
                        returncode=0,
                        stdout=json.dumps(
                            {
                                "final_net_assets": value,
                                "n_steps": 24,
                                "net_profit": value - 3000,
                                "is_alive": True,
                            }
                        ),
                    ),
                ),
            ):
                got = self.adapter.score(out, self.task["id"])
                self.assertEqual(got["n_passed"], value)
                self.assertEqual(got["n_criteria"], 1)
                self.assertFalse(got["all_pass"])

    @patch.object(EcomAdapter, "check_environment")
    def test_invalid_or_failed_measurement_is_ungraded(self, _check):
        out = self.make_scored_run()
        for result in [
            Mock(returncode=1, stdout=""),
            Mock(returncode=0, stdout="invalid json"),
            Mock(returncode=0, stdout='{"final_net_assets": NaN, "n_steps": 24}'),
            Mock(returncode=0, stdout='{"final_net_assets": 3000, "n_steps": 0}'),
        ]:
            with patch("harness2.benchmarks.ecommerce.subprocess.run", return_value=result):
                self.assertIsNone(self.adapter.score(out, self.task["id"]))
        self.assertIsNone(self.adapter.score(out, "different/task"))
        (out / "measurement/episode.json").write_text("changed")
        self.assertFalse(self.adapter.artifacts_intact(out))
        self.assertIsNone(self.adapter.score(out, self.task["id"]))

    def test_report_is_assets_not_percentage(self):
        rows = [
            Scored(
                "task",
                "agg",
                1,
                "agg1",
                "id",
                False,
                n_passed=value,
                n_criteria=1,
                raw={
                    "metric": "final_net_assets",
                    "net_profit": value - 3000,
                    "is_alive": value > 0,
                },
            )
            for value in (-100, 6100)
        ]
        summary = report.summarise(rows)[0]
        self.assertEqual(summary.score, 3000)
        self.assertEqual(summary.net_profit, 0)
        text = report.render([summary], bench="ecom")
        self.assertIn("mean final net assets 3,000.00", text)
        self.assertIn("survived 50.0%", text)
        self.assertNotIn("300000", text)
        self.assertNotIn("score 3000.00%", text)

    def test_mixed_metric_report_is_rejected(self):
        rows = [
            Scored("task", "agg", 1, "agg1", "id", False, n_passed=value, n_criteria=1, raw=raw)
            for value, raw in ((3000, {"metric": "final_net_assets"}), (1, {}))
        ]
        with self.assertRaisesRegex(ValueError, "Cannot aggregate"):
            report.summarise(rows)

    def test_evidence_never_contains_measurement(self):
        out = self.make_scored_run(
            {
                "task_id": self.task["id"],
                "terminal_status": "finished",
                "secret_native_outcome": 987654.32,
            }
        )
        dest = self.root / "context"
        context.build(
            dest, self.adapter, self.task["id"], [RunView("base1", "id", out, True, True)]
        )
        text = "\n".join(p.read_text() for p in dest.rglob("*") if p.is_file())
        self.assertNotIn("secret_native_outcome", text)
        self.assertNotIn("987654", text)
        self.assertIn("operated", text)

    def test_judge_needs_no_model_or_api_credentials(self):
        out = self.make_scored_run()
        spec = replace(self.adapter.spec, runs_root=self.root / "results")
        self.adapter.spec = spec
        ns = _common.runs_root_for(self.adapter) / _common.summary_namespace(
            self.adapter, 1, seed=None
        )
        task_dir = ns / "task"
        task_dir.mkdir(parents=True)
        (task_dir / "summary.json").write_text(
            json.dumps(
                {
                    "selections": {
                        "agg": {
                            "task_id": self.task["id"],
                            "mode": "agg",
                            "k": 1,
                            "ok": True,
                            "chosen_label": "agg1",
                            "chosen_run_id": "id",
                            "chosen_run_dir": str(out),
                        }
                    }
                }
            )
        )
        with (
            patch.object(_common, "make_adapter", return_value=self.adapter),
            patch.object(
                config,
                "validate_models",
                side_effect=AssertionError("must not require judge model"),
            ),
            patch.object(
                self.adapter,
                "score",
                return_value={
                    "n_passed": 3000,
                    "n_criteria": 1,
                    "metric": "final_net_assets",
                },
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(judge_cli.main(["--bench", "ecom"]), 0)
        self.assertTrue((ns / "judged.json").is_file())

    def test_env_only_setup_does_not_install_model_tools(self):
        with (
            patch("harness2.cli.setup.install.merchantbench") as install,
            patch(
                "harness2.cli.setup.install.clone",
                side_effect=AssertionError("must not clone model tools"),
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(setup.main(["ecom", "--env-only", "--dry-run"]), 0)
            install.assert_called_once_with(dry_run=True)


@unittest.skipUnless(config.MERCHANTBENCH_PYTHON.is_file(), "run harness2-setup ecom --env-only")
class NativeEpisodeTests(unittest.TestCase):
    setUp = EcomTests.setUp
    write_tasks = EcomTests.write_tasks

    def test_live_episode_bridge_resume_and_native_score(self):
        fake = self.root / "fake_executor.py"
        fake.write_text("""import json, subprocess, sys
from pathlib import Path
assert (Path.cwd() / "AGENTS.md").is_file()
while True:
    packet = json.loads(subprocess.check_output([sys.executable, "merchantbench.py", "observe"], text=True))
    if packet.get("terminal"):
        break
    subprocess.check_call([sys.executable, "merchantbench.py", "act", "end_of_step", "{}"], stdout=subprocess.DEVNULL)
print(json.dumps({"type":"text", "part":{"text":"Finished all activations"}}))
""")
        with patch.object(
            self.adapter.substrate,
            "run_argv",
            return_value=[str(config.MERCHANTBENCH_PYTHON), str(fake)],
        ):
            out = self.adapter.execute(self.task["id"], None, "native-test", timeout=90)
            first = runmark.read_mark(out)
            self.assertTrue(self.adapter.delivered(out))
            scored = self.adapter.score(out, self.task["id"])
            self.assertEqual(scored["final_net_assets"], 3000)
            self.assertGreaterEqual(scored["n_steps"], 24)
            self.assertEqual(self.adapter.execute(self.task["id"], None, "native-test"), out)
            self.assertEqual(first, runmark.read_mark(out))
            (out / "operations.jsonl").write_text("tampered")
            self.assertFalse(self.adapter.artifacts_intact(out))
            self.assertEqual(
                self.adapter.execute(self.task["id"], None, "native-test", timeout=90), out
            )
            self.assertTrue(self.adapter.artifacts_intact(out))

    def test_codex_launch_credentials_and_trajectory(self):
        fake = self.root / "fake_codex.py"
        fake.write_text("""import json, os, subprocess, sys
from pathlib import Path
assert os.environ["VERTEX_PROXY_KEY"] == "test-endpoint-key"
assert (Path(os.environ["CODEX_HOME"]) / "config.toml").is_file()
assert (Path.cwd() / "AGENTS.md").is_file()
while True:
    packet = json.loads(subprocess.check_output([sys.executable, "merchantbench.py", "observe"], text=True))
    if packet.get("terminal"):
        break
    subprocess.check_call([sys.executable, "merchantbench.py", "act", "end_of_step", "{}"], stdout=subprocess.DEVNULL)
print(json.dumps({"type":"item.completed", "item":{"type":"agent_message", "text":"Codex activations complete"}}))
""")
        home = self.root / "configured_codex_home"
        home.mkdir()
        (home / "config.toml").write_text('model = "test-model"\n')
        adapter = EcomAdapter("cx", tasks_file=self.manifest, store_root=self.root / "cx-episodes")
        with (
            patch.object(config, "CODEX_HOME_DIR", home),
            patch.object(config, "VERTEX_PROXY_KEY", "test-endpoint-key"),
            patch.object(
                adapter.substrate,
                "exec_argv",
                return_value=[str(config.MERCHANTBENCH_PYTHON), str(fake)],
            ),
        ):
            out = adapter.execute(self.task["id"], None, "cx-native", timeout=90)
        self.assertEqual(adapter.score(out, self.task["id"])["final_net_assets"], 3000)
        self.assertIn("Codex activations complete", adapter.trajectory(out, obs_trunc=1000))
        dest = self.root / "native-context"
        context.build(
            dest, adapter, self.task["id"], [RunView("base1", "cx-native", out, True, True)]
        )
        self.assertFalse(any("measurement" in str(p.relative_to(dest)) for p in dest.rglob("*")))

    def test_agent_token_cannot_access_administration_or_measurements(self):
        from urllib.error import HTTPError

        with LocalEpisode(
            config.merchantbench_repo(),
            config.MERCHANTBENCH_PYTHON,
            self.root / "server",
            self.task,
        ) as episode:
            for path in (
                "/runs",
                f"/runs/{episode.run_id}",
                f"/runs/{episode.run_id}/agents/agent_0/sections/merchant",
            ):
                with self.assertRaises(HTTPError) as error:
                    request_json(episode.base, path, episode.agent_token)
                self.assertEqual(error.exception.code, 403)
            env = episode.agent_environment({"MERCHANTBENCH_ADMIN_TOKEN": "secret", "PATH": "ok"})
            self.assertNotIn("MERCHANTBENCH_ADMIN_TOKEN", env)
            self.assertEqual(env["MERCHANTBENCH_AGENT_TOKEN"], episode.agent_token)
            self.assertEqual(env["PATH"], "ok")
            self.assertIn("127.0.0.1", env["NO_PROXY"])
            self.assertEqual(env["NO_PROXY"], env["no_proxy"])
            self.assertIsNone(self.adapter.score(self.root / "missing", self.task["id"]))

    def test_early_executor_exit_is_not_cached_or_scored(self):
        with patch.object(
            self.adapter.substrate,
            "run_argv",
            return_value=[str(config.MERCHANTBENCH_PYTHON), "-c", "print('early exit')"],
        ):
            with self.assertRaisesRegex(RuntimeError, "before MerchantBench reported terminal"):
                self.adapter.execute(self.task["id"], None, "early", timeout=30)
        out = self.adapter._episode_dir(self.task["id"], "early")
        self.assertIsNone(runmark.read_mark(out))
        self.assertIsNone(self.adapter.score(out, self.task["id"]))


if __name__ == "__main__":
    unittest.main()
