"""Read-only paired monitoring: no model calls or simulator DB access."""

from __future__ import annotations

import fcntl
import hashlib
import http.client
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from harness2.core import ecom_monitor as M
from harness2.core import ecom_pair_monitor as P


class PairMonitorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        M.atomic_json(
            self.root / "plan.json",
            {
                "method": "frozen-harness-paired-v1",
                "repeats": 3,
                "tasks": [
                    {"id": "synthetic/paired-seed1000", "master_seed": 1000, "horizon_steps": 24}
                ],
                "planned_rollouts": 6,
                "solver_model": "LOCAL_OFFLINE_FIXTURE",
            },
        )
        M.atomic_json(
            self.root / "freeze-provenance.json",
            {"invocation_limit": 2, "candidate_training_seed": 15},
        )

    def complete(self, arm, profit=12.3):
        label = f"seed1000-r0-{arm}"
        ep = P.episode_dir(self.root, "synthetic/paired-seed1000", label)
        (ep / "measurement").mkdir(parents=True)
        (ep / "measurement/episode.json").write_text('{"terminal_status":"finished"}')
        (ep / "trajectory.jsonl").write_text('{"type":"step_finish"}\n')
        (ep / "operations.jsonl").write_text("{}\n")
        content = b"\0".join(
            (ep / name).read_bytes()
            for name in ("measurement/episode.json", "trajectory.jsonl", "operations.jsonl")
        )
        M.atomic_json(
            ep / ".adapt_complete", {"artifact_fingerprint": hashlib.sha256(content).hexdigest()}
        )
        (self.root / "results").mkdir(exist_ok=True)
        M.atomic_json(
            self.root / "results" / f"{label}.json",
            {
                "master_seed": 1000,
                "repeat": 0,
                "arm": arm,
                "status": "ok",
                "episode": str(ep),
                "attempt_history": [{"status": "ok"}],
                "metrics": {"net_profit": profit, "cumulative_fine": 2.5, "is_alive": True},
            },
        )
        return ep

    def test_completed_pilot_is_not_running_or_full_completion(self):
        self.complete("base", 11.1)
        self.complete("agg", 24.7)
        M.atomic_json(
            self.root / "pilot-status.json",
            {
                "status": "completed",
                "verification_passed": True,
                "pid": 999999,
                "started_epoch": 10,
                "finished_epoch": 30,
            },
        )
        snap = P.snapshot(self.root)
        self.assertEqual(snap["status"], "pilot_complete")
        self.assertFalse(snap["worker_alive"])
        self.assertEqual(snap["counts"]["rollouts"], 2)
        self.assertEqual(snap["counts"]["complete_pairs"], 1)
        self.assertEqual(snap["counts"]["not_run"], 4)
        self.assertAlmostEqual(snap["pairs"][0]["delta"], 13.6)
        self.assertIsNone(snap["pairs"][1]["delta"])
        self.assertEqual(snap["elapsed_seconds"], 20)

    def test_old_passed_status_cannot_hide_new_artifact_corruption(self):
        self.complete("base")
        ep = self.complete("agg")
        M.atomic_json(
            self.root / "pilot-status.json", {"status": "completed", "verification_passed": True}
        )
        self.assertEqual(P.snapshot(self.root)["status"], "pilot_complete")
        (ep / "operations.jsonl").write_text("changed")
        self.assertEqual(P.snapshot(self.root)["status"], "verification_failed")

    def test_stale_running_json_is_interrupted_not_alive(self):
        M.atomic_json(
            self.root / "pilot-status.json", {"status": "running", "pid": 123, "started_epoch": 10}
        )
        snap = P.snapshot(self.root)
        self.assertEqual(snap["status"], "interrupted")
        self.assertFalse(snap["worker_alive"])
        self.assertIsNone(snap["elapsed_seconds"])

    def test_lock_and_partial_trajectory_show_real_live_arm(self):
        ep = P.episode_dir(self.root, "synthetic/paired-seed1000", "seed1000-r0-base")
        ep.mkdir(parents=True)
        (ep / "trajectory.jsonl").write_text('{"type":"tool_use","part":{"tool":"bash"}}\n')
        M.atomic_json(self.root / "pilot-status.json", {"status": "running", "started_epoch": 10})
        with (self.root / "pilot.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            snap = P.snapshot(self.root)
            self.assertTrue(snap["worker_alive"])
            self.assertEqual(snap["pairs"][0]["arms"]["base"]["status"], "running")
        self.assertEqual(P.snapshot(self.root)["pairs"][0]["arms"]["base"]["status"], "incomplete")

    def test_tampered_artifact_excluded_from_complete_counts(self):
        ep = self.complete("base")
        (ep / "operations.jsonl").write_text("changed")
        snap = P.snapshot(self.root)
        self.assertEqual(snap["counts"]["rollouts"], 0)
        self.assertEqual(snap["counts"]["invalid"], 1)
        self.assertIsNone(snap["pairs"][0]["arms"]["base"]["net_profit"])

    def test_failed_history_and_missing_are_not_zero_profit(self):
        (self.root / "results").mkdir()
        M.atomic_json(
            self.root / "results/seed1000-r0-base.json",
            {
                "master_seed": 1000,
                "repeat": 0,
                "arm": "base",
                "status": "failed",
                "error": "network disconnected",
                "attempt_history": [{"status": "failed"}],
            },
        )
        snap = P.snapshot(self.root)
        self.assertEqual(snap["counts"]["failed"], 1)
        self.assertEqual(snap["counts"]["failed_attempts"], 1)
        self.assertIsNone(snap["pairs"][0]["arms"]["base"]["net_profit"])
        self.assertIsNone(snap["pairs"][0]["delta"])

    def test_cell_attribution_mismatch_is_invalid(self):
        self.complete("base")
        path = self.root / "results/seed1000-r0-base.json"
        row = M.read_json(path)
        row["master_seed"] = 99
        M.atomic_json(path, row)
        self.assertEqual(P.snapshot(self.root)["counts"]["invalid"], 1)

    def test_logs_are_allowlisted_bounded_and_redacted(self):
        self.complete("base")
        (self.root / "pilot.log").write_text(
            "x" * (M.TAIL_BYTES * 2) + "\nAuthorization: Bearer abcdefghijklmnop\n"
        )
        log = P.log_view(self.root, "run", "seed1000-r0")
        self.assertTrue(log["truncated"])
        self.assertNotIn("abcdefghijklmnop", log["text"])
        self.assertIn("[REDACTED]", log["text"])
        for source, pair in (("../../.env", None), ("base", "../../private")):
            with self.assertRaises(ValueError):
                P.log_view(self.root, source, pair)
        outside = self.root.parent / (self.root.name + "-secret")
        outside.write_text("private sentinel")
        self.addCleanup(outside.unlink)
        (self.root / "pilot.log").unlink()
        (self.root / "pilot.log").symlink_to(outside)
        with self.assertRaises(ValueError):
            P.log_view(self.root, "run")

    def test_background_start_reserves_port_and_refuses_duplicate(self):
        command = [
            sys.executable,
            "-m",
            "harness2.core.ecom_pair_monitor",
            "--output",
            str(self.root),
            "--port",
            "0",
            "--background",
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
        info = json.loads(result.stdout)
        self.addCleanup(os.kill, info["pid"], signal.SIGTERM)
        self.assertTrue(M.locked(self.root, "paired-dashboard.lock"))
        port = int(info["url"].rsplit(":", 1)[1])
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/api/status")
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(response.read())["counts"]["planned_rollouts"], 6)
        conn.close()
        again = subprocess.run(command, capture_output=True, text=True, timeout=10)
        self.assertNotEqual(again.returncode, 0)
        self.assertEqual(M.read_json(self.root / "paired-dashboard.json")["pid"], info["pid"])

    def test_http_same_origin_readonly_and_no_arbitrary_files(self):
        server = P.make_server(self.root, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        port = server.server_address[1]

        def request(method, path, headers=None):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
            conn.request(method, path, headers=headers or {})
            response = conn.getresponse()
            body = response.read()
            code = response.status
            head = dict(response.getheaders())
            conn.close()
            return code, body, head

        for path in ("/", "/style.css", "/api/status", "/api/log"):
            code, _, headers = request("GET", path)
            self.assertEqual(code, 200)
            self.assertEqual(headers["Cache-Control"], "no-store")
            self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        self.assertEqual(request("GET", "/api/status", {"Host": "evil.test"})[0], 403)
        self.assertEqual(request("GET", "/api/status", {"Origin": "https://evil.test"})[0], 403)
        for method in ("POST", "PUT", "PATCH", "DELETE"):
            self.assertEqual(request(method, "/api/status")[0], 405)
        for path in ("/.env", "/server-state/state.db", "/plan.json", "/../../etc/passwd"):
            self.assertEqual(request("GET", path)[0], 404)
        self.assertEqual(request("GET", "/api/log?source=.env")[0], 400)
        self.assertEqual(server.server_address[0], "127.0.0.1")


if __name__ == "__main__":
    unittest.main()
