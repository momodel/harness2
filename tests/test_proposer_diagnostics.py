"""No model/network calls: exercise the real timeout and process-group boundary."""
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from harness2 import config
from harness2.core.proposer import (
    _FETCH_TRACE_JS,
    Proposer,
    ProposerError,
    _call_fingerprint,
    _phase,
    _retain_opencode_state,
    _run_logged,
)
from harness2.core.spec import ChannelSpec


class ProposerDiagnosticsTests(unittest.TestCase):
    def test_timeout_keeps_each_attempt_and_last_event(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            worker = root / "worker.py"
            worker.write_text(
                "import json,sys,time\n"
                "print(json.dumps({'type':'tool_use','part':{'tool':'bash','state':{'status':'running'}}}),flush=True)\n"
                "print('diagnostic-marker',file=sys.stderr,flush=True)\n"
                "time.sleep(30)\n"
            )
            prop = Proposer(ChannelSpec("opencode", "local/test", timeout=1), audit_dir=root / "audit")
            real = subprocess.Popen

            def launch(command, **kwargs):
                return real([sys.executable, str(worker)], **kwargs)

            with (
                patch("harness2.core.proposer._opencode_dir", return_value=root),
                patch("harness2.core.proposer.subprocess.Popen", side_effect=launch),
                patch("harness2.core.proposer.time.sleep"),
                self.assertRaises(ProposerError),
            ):
                prop.call("test", None, "timeout", max_attempts=2)
            logs = sorted((root / "audit").glob("*.attempts/*/process.json"))
            self.assertEqual(len(logs), 2)
            for log in logs:
                data = json.loads(log.read_text())
                self.assertEqual(data["status"], "timeout")
                self.assertEqual(data["phase"], "model_cli")
                self.assertEqual(data["last_event"]["tool"], "bash")
                self.assertEqual(data["last_event"]["status"], "running")
                self.assertGreaterEqual(data["wall_s"], 1)
                self.assertIn('"tool_use"', (log.parent / "stdout.jsonl").read_text())
                self.assertIn("diagnostic-marker", (log.parent / "stderr.log").read_text())
                self.assertEqual(log.parent.stat().st_mode & 0o777, 0o700)
            meta = json.loads(next((root / "audit").glob("*.meta.json")).read_text())
            self.assertEqual(len(meta["diagnostics"]), 2)
            with (
                patch("harness2.core.proposer._opencode_dir", return_value=root),
                patch("harness2.core.proposer.subprocess.Popen", side_effect=launch),
                self.assertRaises(ProposerError),
            ):
                prop.call("test", None, "timeout", max_attempts=1)
            self.assertEqual(len(list((root / "audit").glob("*.attempts/*/process.json"))), 3,
                             "A later invocation must not overwrite earlier timeout evidence")

    def test_no_first_event_retains_native_log_and_committed_wal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            worker = root / "worker.py"
            worker.write_text(
                "import os,sqlite3,sys,time\n"
                "from pathlib import Path\n"
                "root=Path(os.environ['XDG_DATA_HOME'])/'opencode'\n"
                "(root/'log').mkdir(parents=True)\n"
                "(root/'log'/'init.log').write_text('service=provider init marker')\n"
                "(root/'auth.json').write_text('DO_NOT_COPY_AUTH')\n"
                "db=sqlite3.connect(root/'opencode.db')\n"
                "db.execute('PRAGMA journal_mode=WAL')\n"
                "db.execute('PRAGMA wal_autocheckpoint=0')\n"
                "db.execute('CREATE TABLE probe(value TEXT)')\n"
                "db.execute(\"INSERT INTO probe VALUES ('committed-before-first-event')\")\n"
                "db.commit()\n"
                "print('Database migration complete.',file=sys.stderr,flush=True)\n"
                "time.sleep(30)\n"
            )
            prop = Proposer(ChannelSpec("opencode", "local/test", timeout=1),
                            audit_dir=root / "audit", diagnostics=True)
            real = subprocess.Popen
            workspaces = []

            def launch(command, **kwargs):
                workspaces.append(kwargs['cwd'])
                self.assertTrue(any(str(arg).endswith("fetch-trace.mjs") for arg in command))
                self.assertIn("--print-logs", command)
                return real([sys.executable, str(worker)], **kwargs)

            with (
                patch("harness2.core.proposer._opencode_dir", return_value=root),
                patch("harness2.core.proposer.subprocess.Popen", side_effect=launch),
                self.assertRaisesRegex(ProposerError, "timeout.*during model_cli"),
            ):
                prop.call("test", None, "no-first-event", max_attempts=1)
            log = next((root / "audit").glob("*.attempts/*/process.json"))
            state = json.loads(log.read_text())
            self.assertEqual(state["last_event"], {})
            self.assertEqual(state["runtime_evidence"]["errors"], [])
            self.assertEqual((log.parent / "stdout.jsonl").stat().st_size, 0)
            self.assertIn("provider init marker", (log.parent / "native/init.log").read_text())
            with sqlite3.connect(log.parent / "native/opencode.db") as db:
                self.assertEqual(db.execute("SELECT value FROM probe").fetchone()[0],
                                 "committed-before-first-event")
                self.assertEqual(db.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            self.assertFalse(any(p.exists() for p in workspaces))
            self.assertFalse((log.parent / "native/auth.json").exists())
            self.assertEqual((log.parent / "native/opencode.db").stat().st_mode & 0o777, 0o600)

    def test_database_and_transport_are_opt_in_and_cache_is_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prop = Proposer(ChannelSpec("opencode", "local/test"))
            self.assertFalse(prop.diagnostics)
            plain = _call_fingerprint("probe", "mission", "", None)
            self.assertEqual(plain, _call_fingerprint("probe", "mission", "", None,
                                                     diagnostics=False))
            self.assertNotEqual(plain, _call_fingerprint("probe", "mission", "", None,
                                                        diagnostics=True))
            workspace = root / "workspace"
            native = workspace / ".ocdata/opencode"
            (native / "log").mkdir(parents=True)
            (native / "log/init.log").write_text("init")
            with sqlite3.connect(native / "opencode.db") as db:
                db.execute("CREATE TABLE marker(value)")
            logs = root / "audit"
            _phase(logs, "model_cli", status="exited")
            _retain_opencode_state(workspace, logs, database=False)
            self.assertTrue((logs / "native/init.log").is_file())
            self.assertFalse((logs / "native/opencode.db").exists())
            self.assertFalse(json.loads((logs / "process.json").read_text())
                             ["runtime_evidence"]["database_requested"])

    def test_unexpected_retention_error_does_not_mask_reply(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prop = Proposer(ChannelSpec("opencode", "local/test"), audit_dir=root / "audit")
            workspaces = []

            def fake(cwd, *args, **kwargs):
                workspaces.append(cwd)
                return "local-reply", 0.0, ""

            with (
                patch.object(prop, "_opencode", side_effect=fake),
                patch("harness2.core.proposer._retain_opencode_state", side_effect=PermissionError),
            ):
                reply = prop.call("test", None, "retention-error", max_attempts=1)
            self.assertEqual(reply.text, "local-reply")
            state = json.loads(next((root / "audit").glob("*.attempts/*/process.json")).read_text())
            self.assertEqual(state["retention_error"], "PermissionError")
            self.assertFalse(any(p.exists() for p in workspaces))

    def test_bad_snapshot_does_not_hide_original_failure_or_copy_external_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = root / "workspace"
            native = workspace / ".harness2-env/data/opencode"
            (native / "log").mkdir(parents=True)
            (native / "opencode.db").write_bytes(b"not a database")
            secret = root / "external-secret"
            secret.write_text("HOST_SECRET")
            (native / "log/unsafe.log").symlink_to(secret)
            logs = root / "audit"
            _phase(logs, "model_cli", status="timeout")
            _retain_opencode_state(workspace, logs, database=True)
            state = json.loads((logs / "process.json").read_text())
            self.assertEqual(state["status"], "timeout")
            self.assertEqual(state["runtime_evidence"]["errors"][0]["kind"], "DatabaseError")
            self.assertFalse((logs / "native/opencode.db").exists())
            self.assertFalse((logs / "native/unsafe.log").exists())
            self.assertEqual(secret.read_text(), "HOST_SECRET")

    @unittest.skipUnless(config.BUN.is_file(), "local Bun required")
    def test_transport_boundaries_with_local_http_only(self):
        release = threading.Event()
        received = []
        payload = "data: BODY_SECRET_测试\\n\\n".encode()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                received.append(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path.startswith("/no-headers"):
                    release.wait(5)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                if self.path.startswith("/ok"):
                    self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.flush()
                if self.path.startswith("/no-body"):
                    release.wait(5)
                    return
                self.wfile.write(payload)
                self.wfile.flush()

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            preload = root / "trace.mjs"
            preload.write_text(_FETCH_TRACE_JS)
            worker = root / "worker.mjs"
            worker.write_text(
                "const r = await fetch(process.argv[2], {method:'POST', "
                "headers:{Authorization:'Bearer HEADER_SECRET'}, body:'REQUEST_SECRET'});\n"
                "if (r.status !== 200 || !r.url.includes('/ok')) { await r.text(); }\n"
                "else { if (!(await r.text()).includes('BODY_SECRET_测试')) throw Error('bytes changed'); }\n"
                "console.log('consumed');\n"
            )
            for route, expected in (
                ("ok", ["fetch_call", "response_headers", "first_body_chunk", "body_end"]),
                ("no-headers", ["fetch_call"]),
                ("no-body", ["fetch_call", "response_headers"]),
            ):
                with self.subTest(route=route):
                    release.clear()
                    logs = root / route
                    env = {**os.environ, "HARNESS2_HTTP_TRACE": str(logs / "transport.jsonl")}
                    command = [str(config.BUN), "--preload", str(preload), str(worker),
                               f"http://127.0.0.1:{server.server_port}/{route}/PATH_SECRET?key=QUERY_SECRET"]
                    try:
                        if route == "ok":
                            result = _run_logged(command, cwd=root, env=env, timeout=5, logs=logs)
                            self.assertEqual(result.returncode, 0, result.stderr)
                            self.assertIn("consumed", result.stdout)
                        else:
                            with self.assertRaises(subprocess.TimeoutExpired):
                                _run_logged(command, cwd=root, env=env, timeout=1, logs=logs)
                    finally:
                        release.set()
                    raw = (logs / "transport.jsonl").read_text()
                    events = [json.loads(line) for line in raw.splitlines()]
                    self.assertEqual(events[0]["event"], "preload_ready")
                    self.assertEqual([e["event"] for e in events[1:]], expected)
                    for token in ("HEADER_SECRET", "REQUEST_SECRET", "BODY_SECRET", "PATH_SECRET", "QUERY_SECRET"):
                        self.assertNotIn(token, raw)
                    self.assertEqual({e["id"] for e in events[1:]}, {1})
                    self.assertEqual([e["elapsed_ms"] for e in events],
                                     sorted(e["elapsed_ms"] for e in events))
        self.assertEqual(received, [b"REQUEST_SECRET"] * 3, "Positive control: request bodies reached server")

    @unittest.skipUnless(os.name == "posix", "process groups require POSIX")
    def test_timeout_terminates_tool_descendants(self):
        from harness2.core.proposer import _run_logged

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            # A surviving descendant would write this sentinel after the parent times out.
            sentinel = root / "survived"
            child = f"import time;from pathlib import Path;time.sleep(1.5);Path({str(sentinel)!r}).touch()"
            parent = (
                "import subprocess,sys,time;"
                f"subprocess.Popen([sys.executable,'-c',{child!r}]);"
                "time.sleep(30)"
            )
            with self.assertRaises(subprocess.TimeoutExpired):
                _run_logged([sys.executable, "-c", parent], cwd=root, env=os.environ.copy(),
                            timeout=0.5, logs=root / "logs")
            # A real bounded wait is needed for this negative process-liveness assertion.
            subprocess.run([sys.executable, "-c", "import time;time.sleep(1.5)"], check=True)
            self.assertFalse(sentinel.exists())
            # Positive control: the same child does produce the sentinel without cleanup.
            subprocess.run([sys.executable, "-c", child], check=True)
            self.assertTrue(sentinel.exists())

    def test_success_keeps_logs_and_parses_reply(self):
        from harness2.core.proposer import _run_logged

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            p = _run_logged([sys.executable, "-c", "print('ok')"], cwd=root,
                            env=os.environ.copy(), timeout=3, logs=root / "logs")
            self.assertEqual(p.stdout.strip(), "ok")
            self.assertEqual(p.returncode, 0)
            self.assertEqual(json.loads((root / "logs/process.json").read_text())["status"], "exited")


if __name__ == "__main__":
    unittest.main()
