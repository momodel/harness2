"""No paid calls: lifecycle fixtures, native artifact layouts and loopback HTTP."""

from __future__ import annotations

import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from harness2.cli import ecom_train as T
from harness2.core import ecom_monitor as M
from harness2.core import schedule


class TrainingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.temp = Path(temporary.name).resolve()
        self.root = self.temp / "experiment"
        self.profile = self.temp / "models.json"
        self.profile.write_text('{"provider":{}}')
        self.args = T.parser().parse_args(
            [
                "run",
                "--output",
                str(self.root),
                "--tasks",
                "2",
                "--seed-start",
                "60",
                "--hours",
                "72",
                "--model-config",
                str(self.profile),
                "--no-dashboard",
            ]
        )

    def prepared(self, **options):
        for name, value in options.items():
            setattr(self.args, name, value)
        T.prepare(self.args)
        return T.validate(self.root)

    def fill(self, plan, fallback=False):
        token = "seq" if plan["mode"] == "sequential" else "agg"
        ns = plan["namespace"]
        run_ns = ns.replace("-agg", "-seq", 1) if token == "seq" else ns
        nsdir = self.root / "ecom/adapt" / ns
        selected, bases = [], []
        for task in plan["tasks"]:
            tid = task["id"]
            work = M.task_dir(self.root, ns, tid)
            work.mkdir(parents=True, exist_ok=True)
            stages = []
            for i in range(1, plan["k"] + 1):
                for label in (
                    f"base{i}",
                    f"inv{i}",
                    f"dom{i}",
                    f"agg{i}" if token == "agg" else f"c{i}",
                ):
                    space = ns if label.startswith("base") else run_ns
                    episode = M.episode_dir(self.root, space, tid, label)
                    (episode / "measurement").mkdir(parents=True, exist_ok=True)
                    (episode / "measurement/episode.json").write_text("{}")
                    (episode / "operations.jsonl").write_text('{"operation":"fixture"}\n')
                    (episode / "trajectory.jsonl").write_text(
                        '{"type":"text","part":{"text":"LOCAL_FIXTURE"}}\n'
                    )
                    (episode / "environment.json").write_text("{}")
                    from harness2.benchmarks.ecommerce import EcomAdapter

                    M.atomic_json(
                        episode / ".adapt_complete",
                        {"artifact_fingerprint": EcomAdapter._artifact_fingerprint(episode)},
                    )
                    if not label.startswith("base"):
                        stages.append(
                            {
                                "label": label,
                                "fell_back": fallback and label == "inv1",
                                "attempts": 1,
                                "error": "timeout" if fallback and label == "inv1" else "",
                            }
                        )
                bases.append(
                    {"task": tid, "label": f"base{i}", "score": {"final_net_assets": 3201.25}}
                )
            M.atomic_json(
                work / "summary.json",
                {
                    "task_id": tid,
                    "status": "ok",
                    "errors": [],
                    "modes": {
                        token: {
                            "ok": True,
                            "errors": [],
                            "fell_back": ["inv1"] if fallback else [],
                            "stages": stages,
                        }
                    },
                    "selections": {
                        token: {
                            "ok": True,
                            "experience": {
                                "cue": "fixture",
                                "lesson": "fixture",
                                "evidence": "fixture",
                            },
                        }
                    },
                },
            )
            variant = f"k{plan['k']}-oc-s{plan['order_seed']}"
            domain = self.root / "ecom/library/ecom" / variant / token / "synthetic"
            entry = domain / "harnesses" / tid.replace("/", "__")
            (entry / "harness").mkdir(parents=True, exist_ok=True)
            (entry / "EXPERIENCE.md").write_text("LOCAL_FIXTURE")
            (domain / "INDEX.md").write_text("LOCAL_FIXTURE")
            selected.append(
                {
                    "task_id": tid,
                    "graded": True,
                    "raw": {"final_net_assets": 3412.75, "net_profit": 412.75, "is_alive": True},
                }
            )
        M.atomic_json(nsdir / "judged.json", selected)
        M.atomic_json(nsdir / "base_judged.json", bases)

    def executor(self, plan, fallback=False, fail=None):
        commands = []

        def run(argv, env, log, on_line):
            command = argv[2]
            commands.append(argv)
            self.assertEqual(env["HARNESS2_ECOM_TIMEOUT"], "1800")
            self.assertEqual(env["HARNESS2_ECOM_TASKS_FILE"], str(self.root / "tasks.json"))
            if command == fail:
                log.write("fixture failure\n")
                return 3
            if command == "run":
                self.fill(plan, fallback)
                for i, task in enumerate(plan["tasks"], 1):
                    on_line(f"  [synthetic] {i}/{len(plan['tasks'])} ok {task['id']}\n")
            log.write(f"LOCAL_FIXTURE_{command}\n")
            return 0

        return run, commands

    def test_dry_run_does_not_write_or_launch(self):
        self.args.dry_run = True
        with patch.object(subprocess, "Popen") as spawn, contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(T.prepare(self.args))
        spawn.assert_not_called()
        self.assertFalse(self.root.exists())

    def test_parameters_and_order_match_core(self):
        plan = self.prepared(k=2, mode="sequential")
        tasks = plan["tasks"]
        self.assertEqual({t["horizon_steps"] for t in tasks}, {72})
        self.assertEqual({t["master_seed"] for t in tasks}, {60, 61})
        actual = schedule.stream_order({"synthetic": [t["id"] for t in tasks]}, 0, ["synthetic"])
        self.assertEqual([t["id"] for t in tasks], [tid for _, tid in actual])
        with self.assertRaises(FileExistsError):
            T.prepare(self.args)
        for option in ("--tasks", "--hours", "--k", "--timeout"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                T.parser().parse_args(["run", option, "0"])

    def test_resume_rejects_changed_inputs(self):
        self.prepared()
        (self.root / "tasks.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            T.validate(self.root)

    def test_resume_rejects_changed_profile_and_sources(self):
        self.prepared()
        self.profile.write_text('{"provider":{"changed":{}}}')
        with self.assertRaisesRegex(ValueError, "profile changed"):
            T.validate(self.root)
        self.profile.write_text('{"provider":{}}')
        with patch.object(T, "sources", return_value={}):
            with self.assertRaisesRegex(ValueError, "source changed"):
                T.validate(self.root)

    def test_worker_full_sequence_and_history(self):
        plan = self.prepared()
        execute, commands = self.executor(plan)
        self.assertEqual(T.run_worker(self.root, executor=execute), 0)
        self.assertEqual(M.read_json(self.root / "job-status.json")["status"], "completed")
        self.assertTrue(M.read_json(self.root / "verification.json")["passed"])
        self.assertEqual([c[2] for c in commands], ["check", "run", "judge", "judge", "report"])
        self.assertNotIn("--bases", commands[2])
        self.assertIn("--bases", commands[3])
        self.assertNotIn("--bases", commands[4])
        self.assertIn("LOCAL_FIXTURE_report", (self.root / "report.txt").read_text())
        self.assertEqual(T.run_worker(self.root, executor=execute), 0)
        self.assertTrue((self.root / "job-history.jsonl").is_file())

    def test_sequential_k2_rollout_paths(self):
        plan = self.prepared(k=2, mode="sequential")
        execute, _ = self.executor(plan)
        self.assertEqual(T.run_worker(self.root, executor=execute), 0)
        view = M.snapshot(self.root)
        self.assertEqual(view["counts"]["rollouts"], 16)
        self.assertEqual(view["counts"]["planned_rollouts"], 16)

    def test_fallback_is_not_reported_as_success(self):
        plan = self.prepared()
        execute, _ = self.executor(plan, fallback=True)
        self.assertEqual(T.run_worker(self.root, executor=execute), 2)
        self.assertEqual(M.snapshot(self.root)["status"], "completed_with_warnings")
        self.assertFalse(M.read_json(self.root / "verification.json")["checks"]["no_fallbacks"])

    def test_technical_failure_stops_before_judge(self):
        plan = self.prepared()
        execute, commands = self.executor(plan, fail="run")
        self.assertEqual(T.run_worker(self.root, executor=execute), 2)
        self.assertEqual([c[2] for c in commands], ["check", "run"])
        self.assertEqual(M.snapshot(self.root)["status"], "failed")

    def test_artifact_tampering_fails_verification(self):
        plan = self.prepared()
        self.fill(plan)
        (self.root / "report.txt").write_text("fixture report")
        episode = M.episode_dir(self.root, plan["namespace"], plan["tasks"][0]["id"], "inv1")
        (episode / "trajectory.jsonl").write_text("tampered")
        self.assertFalse(T.verify(self.root, plan)["checks"]["all_rollout_artifacts_intact"])

    def test_missing_experience_and_duplicate_scores_fail_verification(self):
        plan = self.prepared()
        self.fill(plan)
        (self.root / "report.txt").write_text("fixture report")
        (next(self.root.glob("ecom/library/**/EXPERIENCE.md"))).unlink()
        selected_file = self.root / "ecom/adapt" / plan["namespace"] / "judged.json"
        rows = M.read_json(selected_file)
        M.atomic_json(selected_file, [rows[0], rows[0]])
        checks = T.verify(self.root, plan)["checks"]
        self.assertFalse(checks["all_reflections_complete"])
        self.assertFalse(checks["all_selected_graded"])

    def test_explicit_seeds_do_not_accidentally_start_intermediate_tasks(self):
        args = T.parser().parse_args(["run", "--seeds", "0,7", "--model-config", str(self.profile)])
        plan = T.build_plan(args)
        self.assertEqual({t["master_seed"] for t in plan["tasks"]}, {0, 7})
        self.assertEqual(len(plan["tasks"]), 2)
        for value in ("0,0", "", "1,-2", "0,,7"):
            with (
                self.subTest(value=value),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                T.parser().parse_args(["run", "--seeds", value])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            T.parser().parse_args(["run", "--tasks", "2", "--seeds", "0,7"])

    def test_explicit_no_entry_satisfies_reflection_gate_without_fabricating_library(self):
        plan = self.prepared()
        self.fill(plan)
        (self.root / "report.txt").write_text("fixture report")
        tid = plan["tasks"][0]["id"]
        path = M.task_dir(self.root, plan["namespace"], tid) / "summary.json"
        summary = M.read_json(path)
        summary["selections"]["agg"].update(
            experience={}, reflection_status="no_entry", reflection_error=""
        )
        M.atomic_json(path, summary)
        for artifact in self.root.glob(
            "ecom/library/**/" + tid.replace("/", "__") + "/EXPERIENCE.md"
        ):
            artifact.unlink()
        task = M.snapshot(self.root)["tasks"][0]
        self.assertFalse(task["experience"])
        self.assertEqual(task["reflection_status"], "no_entry")
        self.assertTrue(T.verify(self.root, plan)["passed"])

    def test_empty_legacy_or_failed_reflection_is_not_silently_no_entry(self):
        plan = self.prepared()
        self.fill(plan)
        tid = plan["tasks"][0]["id"]
        path = M.task_dir(self.root, plan["namespace"], tid) / "summary.json"
        summary = M.read_json(path)
        selection = summary["selections"]["agg"]
        selection["experience"] = {}
        for declared, expected in ((None, "unknown"), ("error", "error"), ("distilled", "missing")):
            with self.subTest(declared=declared):
                selection.pop("reflection_status", None)
                if declared:
                    selection["reflection_status"] = declared
                M.atomic_json(path, summary)
                self.assertEqual(M.snapshot(self.root)["tasks"][0]["reflection_status"], expected)

    def test_unfinished_tasks_do_not_show_stale_scores(self):
        plan = self.prepared()
        self.fill(plan)
        tid = plan["tasks"][0]["id"]
        (M.task_dir(self.root, plan["namespace"], tid) / "summary.json").unlink()
        episode = M.episode_dir(self.root, plan["namespace"], tid, "base1")
        (episode / ".adapt_complete").unlink()
        task = M.snapshot(self.root)["tasks"][0]
        self.assertIsNone(task["base_assets"])
        self.assertIsNone(task["final_assets"])
        self.assertIsNone(task["net_profit"])
        self.assertIsNone(task["alive"])

    def test_worker_lock_prevents_duplicates_and_detects_abandoned_state(self):
        self.prepared()
        M.atomic_json(
            self.root / "job-status.json",
            {"status": "running", "pid": os.getpid(), "started_epoch": time.time() - 60},
        )
        self.assertEqual(M.snapshot(self.root)["status"], "interrupted")
        self.assertIsNone(M.snapshot(self.root)["elapsed_seconds"])
        with T.acquire(self.root, "worker.lock"):
            self.assertTrue(M.snapshot(self.root)["worker_alive"])
            with self.assertRaisesRegex(ValueError, "Already running"):
                T.acquire(self.root, "worker.lock")
        self.assertFalse(M.locked(self.root))

    def test_detached_worker_inherits_running_lock(self):
        self.prepared()
        processes = []
        original_spawn = subprocess.Popen

        def spawn(*args, **kwargs):
            process = original_spawn(*args, **kwargs)
            processes.append(process)
            return process

        child = "import os,sys,signal; handle=os.fdopen(int(sys.argv[-1]), 'a+'); signal.pause()"

        def command(*args):
            return [sys.executable, "-c", child, *map(str, args)]

        try:
            with (
                patch.object(T, "_module_command", command),
                patch.object(subprocess, "Popen", spawn),
                patch.object(
                    subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "", "")
                ),
            ):
                pid = T.start_worker(self.root)
            self.assertTrue(M.locked(self.root))
            self.assertEqual(int((self.root / "job.pid").read_text()), pid)
        finally:
            for process in processes:
                process.terminate()
                process.wait(timeout=5)
        self.assertFalse(M.locked(self.root))

    def test_interruption_terminates_detached_descendants(self):
        script = (
            "import subprocess,sys,time; child=subprocess.Popen([sys.executable,'-c',"
            "'import time;time.sleep(60)'],start_new_session=True); "
            "print('READY',child.pid,flush=True);time.sleep(60)"
        )
        children = []

        def stop(line):
            if line.startswith("READY"):
                children.append(int(line.split()[1]))
                raise KeyboardInterrupt("local fixture stopped")

        try:
            with self.assertRaises(KeyboardInterrupt):
                T.execute(
                    [sys.executable, "-u", "-c", script], dict(os.environ), io.StringIO(), stop
                )
            self.assertEqual(len(children), 1)
            state = subprocess.run(
                ["/bin/ps", "-p", str(children[0]), "-o", "stat="], capture_output=True, text=True
            )
            self.assertTrue(
                not state.stdout.strip() or state.stdout.strip().startswith("Z"), state.stdout
            )
        finally:
            for pid in children:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_live_phase_and_stale_sessions(self):
        plan = self.prepared()
        start = time.time() - 1
        M.atomic_json(
            self.root / "job-status.json",
            {
                "status": "running",
                "stage": "run",
                "task_id": plan["tasks"][0]["id"],
                "task_started_at": start,
                "started_epoch": start,
            },
        )
        attempts = (
            self.root / "ecom/adapt/_audit" / plan["namespace"] / "inv1.a1.1234.meta.attempts/x-a1"
        )
        attempts.mkdir(parents=True)
        M.atomic_json(attempts / "process.json", {"phase": "model_cli", "status": "running"})
        (attempts / "stdout.jsonl").write_text('{"type":"tool_use","part":{"tool":"read"}}\n')
        with T.acquire(self.root, "worker.lock"):
            view = M.snapshot(self.root)
            self.assertEqual(view["phase"], "propose")
            self.assertEqual(view["calls"][0]["label"], "inv1")
            self.assertEqual(view["tasks"][0]["status"], "running")
        self.assertEqual(M.snapshot(self.root)["calls"], [])

    def test_port_conflict_does_not_launch_paid_worker(self):
        import socket

        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            with (
                patch.object(T, "start_worker") as worker,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                code = T.main(
                    [
                        "run",
                        "--output",
                        str(self.root),
                        "--model-config",
                        str(self.profile),
                        "--port",
                        str(listener.getsockname()[1]),
                    ]
                )
            self.assertEqual(code, 1)
            worker.assert_not_called()

    def test_log_boundaries_and_redaction(self):
        self.prepared()
        (self.root / "run.log").write_text(
            "long-line\n" * 15000 + "secret-model-key\nBearer abc12345\n"
        )
        with patch.dict(os.environ, {"TEST_API_KEY": "secret-model-key"}):
            value = M.log_view(self.root)
        self.assertTrue(value["truncated"])
        self.assertLess(len(value["text"]), M.TAIL_BYTES + 100)
        self.assertNotIn("secret-model-key", value["text"])
        self.assertNotIn("abc12345", value["text"])
        with self.assertRaises(ValueError):
            M.log_view(self.root, "../../.env")
        with self.assertRaises(ValueError):
            M.log_view(self.root, "base1", "../../other")
        self.assertNotIn("not-in-env", M._redact('api_key="not-in-env"'))
        self.assertNotIn("private-pass", M._redact("https://user:private-pass@example.com"))
        self.assertNotIn("sk-localfixturekey123", M._redact("sk-localfixturekey123"))
        (self.root / "run.log").unlink()
        (self.root / "run.log").symlink_to(self.profile)
        with self.assertRaisesRegex(ValueError, "outside"):
            M.log_view(self.root)

    def test_historical_run_is_read_only_and_not_assumed_alive(self):
        self.root.mkdir()
        ns = self.root / "ecom/adapt/ecom-oc-agg1"
        ns.mkdir(parents=True)
        M.atomic_json(
            ns / "config.json",
            {
                "tasks": [["synthetic", "synthetic/old"]],
                "benchmark_identity": {
                    "tasks": {
                        "synthetic/old": {
                            "id": "synthetic/old",
                            "master_seed": 42,
                            "horizon_steps": 24,
                        }
                    }
                },
            },
        )
        M.atomic_json(self.root / "job-status.json", {"status": "running", "pid": os.getpid()})
        view = M.snapshot(self.root)
        self.assertEqual(view["status"], "unknown")
        self.assertFalse(view["worker_alive"])
        self.assertEqual(view["tasks"][0]["seed"], 42)
        with self.assertRaisesRegex(ValueError, "managed"):
            T.validate(self.root)

    def test_http_is_loopback_read_only_and_non_file_server(self):
        self.prepared()
        server = M.make_server(self.root, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.assertEqual(server.server_address[0], "127.0.0.1")
        base = f"http://127.0.0.1:{server.server_address[1]}"
        with urlopen(base + "/api/status", timeout=5) as response:
            self.assertEqual(json.load(response)["counts"]["total"], 2)
            self.assertEqual(response.headers["Cache-Control"], "no-store")
            self.assertNotIn("Access-Control-Allow-Origin", response.headers)
        with urlopen(base, timeout=5) as response:
            self.assertIn("训练运行台", response.read().decode())
        for request, expected in (
            (Request(base + "/api/status", data=b"{}", method="POST"), 405),
            (Request(base + "/api/status", headers={"Origin": "https://evil.invalid"}), 403),
            (Request(base + "/api/status", headers={"Host": "evil.invalid"}), 403),
            (Request(base + "/.env"), 404),
            (Request(base + "/api/log?source=../../.env"), 400),
        ):
            with self.assertRaises(HTTPError) as caught:
                urlopen(request, timeout=5)
            self.assertEqual(caught.exception.code, expected)
            caught.exception.close()


if __name__ == "__main__":
    unittest.main()
