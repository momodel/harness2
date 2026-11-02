"""Read-only frozen-pair observatory. No model calls or native DB reads.

Run with ``python -m harness2.core.ecom_pair_monitor --output RUN --background``.
The evaluation runner, its immutable identity and existing training monitor stay unchanged.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from . import ecom_monitor as M

ASSET = Path(__file__).resolve().parents[1] / "assets/ecom-pair-monitor.html"
CONTROLLERS = (("pilot-status.json", "pilot.lock"), ("evaluation-status.json", "evaluation.lock"))


def inside(root, path):
    if not Path(path).resolve().is_relative_to(root):
        raise ValueError("Artifact path is outside the experiment")
    return Path(path)


def read(root, path, default=None):
    return M.read_json(inside(root, path), default)


def load_plan(root):
    plan = read(root, root / "plan.json", {})
    if plan.get("method") != "frozen-harness-paired-v1":
        raise ValueError("Select a frozen-harness paired experiment")
    tasks, repeats = plan.get("tasks", []), plan.get("repeats")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError("Invalid paired repetition count")
    if not tasks or len(tasks) * repeats * 2 > 2000:
        raise ValueError("Monitor requires 1–2000 planned rollout cells")
    ids, seeds = set(), set()
    for task in tasks:
        seed = task.get("master_seed")
        if (
            not isinstance(seed, int)
            or isinstance(seed, bool)
            or seed < 0
            or not isinstance(task.get("id"), str)
            or not task["id"]
            or seed in seeds
            or task["id"] in ids
        ):
            raise ValueError("Invalid or duplicate paired task")
        ids.add(task["id"])
        seeds.add(seed)
    if plan.get("planned_rollouts") != len(tasks) * repeats * 2:
        raise ValueError("Paired plan cell count mismatch")
    return plan


def episode_dir(root, task_id, label):
    digest = hashlib.sha256(f"{task_id}\0paired/{label}".encode()).hexdigest()[:24]
    return Path(root) / "episodes" / digest


@lru_cache(maxsize=256)
def artifact_hash(directory, signatures):
    # Cache only while inode, size, mtime AND ctime remain unchanged.
    return hashlib.sha256(
        b"\0".join((Path(directory) / name).read_bytes() for name, *_ in signatures)
    ).hexdigest()


def intact(root, episode):
    mark = read(root, episode / ".adapt_complete", {})
    names = ["measurement/episode.json", "trajectory.jsonl", "operations.jsonl"]
    if inside(root, episode / "environment.json").is_file():
        names.append("environment.json")
    try:
        signatures = []
        for name in names:
            stat = inside(root, episode / name).stat()
            signatures.append((name, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
        return bool(mark.get("artifact_fingerprint")) and mark[
            "artifact_fingerprint"
        ] == artifact_hash(str(episode), tuple(signatures))
    except OSError:
        return False


def number(value):
    return (
        value
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        else None
    )


def snapshot(root):
    root = Path(root).resolve()
    plan = load_plan(root)
    controllers = [
        (read(root, root / filename, {}), M.locked(root, inside(root, root / lock)))
        for filename, lock in CONTROLLERS
    ]
    state, alive = max(
        controllers, key=lambda item: (item[1], number(item[0].get("started_epoch")) or 0)
    )
    provenance = read(root, root / "freeze-provenance.json", {})
    pairs, latest, attempts, failed_attempts = [], 0, 0, 0
    for task in plan["tasks"]:
        for repeat in range(plan["repeats"]):
            pair_id = f"seed{task['master_seed']}-r{repeat}"
            arms = {}
            for arm in ("base", "agg"):
                label = f"{pair_id}-{arm}"
                episode = inside(root, episode_dir(root, task["id"], label))
                row = read(root, root / "results" / f"{label}.json", {})
                history = row.get("attempt_history", [{"status": row["status"]}] if row else [])
                attempts += len(history)
                failed_attempts += sum(entry.get("status") == "failed" for entry in history)
                events = M._events(inside(root, episode / "trajectory.jsonl"))
                changed = max(
                    M.mtime(inside(root, episode / name))
                    for name in ("trajectory.jsonl", ".adapt_complete", "operations.jsonl")
                )
                latest = max(latest, changed)
                status, error = "pending", str(row.get("error") or "")[:900]
                if row:
                    attributed = (
                        row.get("master_seed") == task["master_seed"]
                        and row.get("repeat") == repeat
                        and row.get("arm") == arm
                    )
                    if not attributed:
                        status, error = "invalid", "结果归属与计划不一致"
                    elif row.get("status") == "failed":
                        status = "failed"
                    elif (
                        row.get("status") == "ok"
                        and Path(str(row.get("episode", ""))).resolve() == episode.resolve()
                        and intact(root, episode)
                    ):
                        status = "complete"
                    else:
                        status, error = "invalid", "完成记录缺失有效原生产物，或指纹不匹配"
                elif events or episode.exists():
                    status = "running" if alive else "incomplete"
                metrics = row.get("metrics", {}) if status == "complete" else {}
                arms[arm] = {
                    "label": label,
                    "status": status,
                    "error": M._redact(error),
                    "net_profit": number(metrics.get("net_profit")),
                    "fine": number(metrics.get("cumulative_fine")),
                    "alive": metrics.get("is_alive")
                    if isinstance(metrics.get("is_alive"), bool)
                    else None,
                    "last_event": events[-1].get("type") if events else None,
                    "last_activity": changed or None,
                }
            statuses = {entry["status"] for entry in arms.values()}
            status = (
                "complete"
                if statuses == {"complete"}
                else "invalid"
                if "invalid" in statuses
                else "failed"
                if "failed" in statuses
                else "running"
                if "running" in statuses
                else "incomplete"
                if "incomplete" in statuses
                else "pending"
            )
            base, candidate = arms["base"]["net_profit"], arms["agg"]["net_profit"]
            pairs.append(
                {
                    "id": pair_id,
                    "seed": task["master_seed"],
                    "repeat": repeat,
                    "hours": task.get("horizon_steps"),
                    "status": status,
                    "arms": arms,
                    "delta": candidate - base
                    if base is not None and candidate is not None
                    else None,
                }
            )
    flat = [entry for pair in pairs for entry in pair["arms"].values()]
    complete = sum(entry["status"] == "complete" for entry in flat)
    complete_pairs = sum(pair["status"] == "complete" for pair in pairs)
    raw_status = state.get("status", "prepared")
    invalid = sum(entry["status"] == "invalid" for entry in flat)
    failed = sum(entry["status"] == "failed" for entry in flat)
    if alive:
        status = "running"
    elif invalid:
        status = "verification_failed"
    elif complete == plan["planned_rollouts"]:
        status = "completed"
    elif raw_status == "failed" or failed:
        status = "failed"
    elif raw_status in ("running", "starting", "interrupted"):
        status = "interrupted"
    elif state.get("verification_passed") and complete >= (
        number(provenance.get("invocation_limit")) or 2
    ):
        status = "pilot_complete"
    else:
        status = "prepared"
    elapsed = 0
    has_duration = False
    for saved, held in controllers:
        start, end = number(saved.get("started_epoch")), number(saved.get("finished_epoch"))
        if start is not None and (end is not None or held):
            elapsed += max(0, (end or time.time()) - start)
            has_duration = True
    return {
        "root": str(root),
        "status": status,
        "worker_alive": alive,
        "server_time": time.time(),
        "elapsed_seconds": elapsed if has_duration else None,
        "last_activity": latest or None,
        "stale": alive and bool(latest) and time.time() - latest > 120,
        "error": M._redact(str(state.get("error") or ""))[:1200],
        "model": plan.get("solver_model"),
        "candidate_seed": provenance.get("candidate_training_seed"),
        "pilot_limit": provenance.get("invocation_limit"),
        "pairs": pairs,
        "counts": {
            "rollouts": complete,
            "planned_rollouts": plan["planned_rollouts"],
            "complete_pairs": complete_pairs,
            "planned_pairs": len(pairs),
            "failed": failed,
            "invalid": invalid,
            "not_run": sum(entry["status"] == "pending" for entry in flat),
            "attempts": attempts,
            "failed_attempts": failed_attempts,
        },
        "warning": "单对结果只用于管线核验；缺失结果不计零，正收益差不代表统计显著。",
    }


def log_view(root, source="run", pair=None):
    root = Path(root).resolve()
    plan = load_plan(root)
    sources = {
        "run": root / "pilot.log",
        "worker": root / "worker.log",
        "report": root / "report.json",
    }
    if (root / "evaluation.log").exists():
        sources["evaluation"] = root / "evaluation.log"
    if pair is not None:
        matches = [
            (task, repeat)
            for task in plan["tasks"]
            for repeat in range(plan["repeats"])
            if pair == f"seed{task['master_seed']}-r{repeat}"
        ]
        if not matches:
            raise ValueError("Unknown pair")
        task, _ = matches[0]
        for arm in ("base", "agg"):
            sources[arm] = episode_dir(root, task["id"], f"{pair}-{arm}") / "trajectory.jsonl"
    if source not in sources:
        raise ValueError("Unknown log source")
    path = inside(root, sources[source])
    text, truncated = M.tail(path)
    if path.suffix == ".jsonl":
        chunks = []
        for event in M._events(path)[-80:]:
            part = event.get("part", {})
            if part.get("tool"):
                state = part.get("state", {})
                chunks.append(
                    f"{part['tool']} · {state.get('status', '')}\n{json.dumps(state.get('input', {}), ensure_ascii=False)[:420]}\n{str(state.get('output', ''))[:600]}"
                )
            elif event.get("type") in ("text", "error"):
                chunks.append(
                    f"{event['type']}\n{str(part.get('text') or event.get('error', ''))[:1200]}"
                )
            else:
                chunks.append(str(event.get("type", "event")))
        text = "\n\n".join(chunks)
    labels = {
        "run": "pilot 日志",
        "worker": "进程日志",
        "report": "已保存统计报告",
        "evaluation": "后续评测日志",
        "base": "Base 原生轨迹",
        "agg": "候选原生轨迹",
    }
    return {
        "text": M._redact(text),
        "truncated": truncated,
        "updated_at": M.mtime(path) or None,
        "sources": [{"id": key, "label": labels[key]} for key in sources],
    }


def make_server(root, port=0, socket_fd=None):
    root = Path(root).resolve()
    load_plan(root)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, code, body, kind="application/json; charset=utf-8"):
            self.send_response(code)
            for key, value in {
                "Content-Type": kind,
                "Content-Length": str(len(body)),
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
                "Content-Security-Policy": "default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'",
            }.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            port = self.server.server_address[1]
            hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            if self.headers.get("Host") not in hosts or (
                self.headers.get("Origin")
                and self.headers["Origin"] not in {f"http://{host}" for host in hosts}
            ):
                self.send(403, b'{"error":"Loopback, same-origin requests only"}')
                return
            url = urlsplit(self.path)
            query = parse_qs(url.query)
            try:
                if url.path in ("/", "/index.html"):
                    self.send(200, ASSET.read_bytes(), "text/html; charset=utf-8")
                    return
                if url.path == "/style.css":
                    css = M.ASSET.read_text().split("<style>", 1)[1].split("</style>", 1)[0]
                    self.send(200, css.encode(), "text/css; charset=utf-8")
                    return
                if url.path == "/api/status":
                    value = snapshot(root)
                elif url.path == "/api/log":
                    value = log_view(
                        root, query.get("source", ["run"])[0], query.get("pair", [None])[0]
                    )
                else:
                    self.send(
                        404,
                        b'<!doctype html><title>Harness2</title><p>Unknown page. <a href="/">Return to monitor</a></p>',
                        "text/html; charset=utf-8",
                    )
                    return
                self.send(200, json.dumps(value, ensure_ascii=False, allow_nan=False).encode())
            except ValueError as exc:
                self.send(400, json.dumps({"error": M._redact(str(exc))}).encode())
            except (OSError, KeyError, TypeError):
                self.send(503, b'{"error":"Artifacts unavailable; retry shortly"}')

        def do_POST(self):
            self.send(405, b'{"error":"Read-only monitor"}')

        do_PUT = do_DELETE = do_PATCH = do_POST

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler, bind_and_activate=socket_fd is None)
    server.daemon_threads = True
    if socket_fd is not None:
        server.socket.close()
        server.socket = socket.socket(fileno=socket_fd)
        server.server_address = server.socket.getsockname()
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--socket-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = args.output.resolve()
    load_plan(root)
    if not 0 <= args.port <= 65535:
        parser.error("port must be 0–65535")
    if args.status:
        print(json.dumps(snapshot(root), ensure_ascii=False, indent=2))
        return 0
    lock = (
        os.fdopen(args.lock_fd, "a+")
        if args.lock_fd is not None
        else (root / "paired-dashboard.lock").open("a+")
    )
    with lock:
        if args.lock_fd is None:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.error(
                    "A paired dashboard already holds the lock; inspect paired-dashboard.json"
                )
        server = make_server(root, args.port, args.socket_fd)
        port = server.server_address[1]
        info = {"url": f"http://127.0.0.1:{port}", "root": str(root), "pid": os.getpid()}
        if args.background:
            with (root / "paired-dashboard.log").open("ab") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "harness2.core.ecom_pair_monitor",
                        "--output",
                        str(root),
                        "--port",
                        str(port),
                        "--socket-fd",
                        str(server.fileno()),
                        "--lock-fd",
                        str(lock.fileno()),
                    ],
                    cwd=Path(__file__).resolve().parents[2],
                    env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    pass_fds=(server.fileno(), lock.fileno()),
                )
            info["pid"] = process.pid
            M.atomic_json(root / "paired-dashboard.json", info)
            server.server_close()
        else:
            M.atomic_json(root / "paired-dashboard.json", info)
            print(json.dumps(info), flush=True)
            try:
                server.serve_forever()
            finally:
                server.server_close()
        print(json.dumps(info, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
