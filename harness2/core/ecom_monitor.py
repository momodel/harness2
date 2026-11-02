"""Read-only, loopback-only views over MerchantBench experiment artifacts.

No model calls, native simulator DB access, arbitrary file serving, or shell
commands are exposed over HTTP. File/event tails and response sizes are bounded.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .library import domain_dir, has_experience, root_for, slug

ASSET = Path(__file__).resolve().parents[1] / "assets/ecom-monitor.html"
TAIL_BYTES = 48 * 1024


def read_json(path, default=None):
    try:
        value = json.loads(Path(path).read_text())
        return value if default is None or isinstance(value, type(default)) else default
    except (OSError, ValueError):
        return default


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def locked(root, name="worker.lock"):
    """An OS lock, not a potentially reused PID, is authoritative for managed jobs."""
    path = Path(root) / name
    if not path.is_file():
        return False
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def tail(path, size=TAIL_BYTES):
    path = Path(path)
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            length = stream.tell()
            stream.seek(max(0, length - size))
            raw = stream.read(size)
        if length > size:
            raw = raw.partition(b"\n")[2]
        return raw.decode(errors="replace"), length > size
    except OSError:
        return "", False


def mtime(path):
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0


def load_plan(root, namespace=None):
    root = Path(root)
    plan = read_json(root / "train.json")
    if isinstance(plan, dict) and plan:
        return plan
    configs = sorted((root / "ecom/adapt").glob("ecom-oc-*/config.json"))
    if namespace:
        configs = [p for p in configs if p.parent.name == namespace]
    if len(configs) != 1:
        raise ValueError("Select one existing experiment with --output and, if needed, --namespace")
    p = configs[0]
    cfg = read_json(p, {})
    tasks = cfg.get("benchmark_identity", {}).get("tasks", {})
    order = [tid for _, tid in cfg.get("tasks", [])]
    # Old namespaces record k in their name rather than in config.json.
    scale = re.search(r"-agg(\d+)", p.parent.name)
    summaries = list(p.parent.glob("*/summary.json"))
    summary = read_json(summaries[0], {}) if summaries else {}
    modes = summary.get("modes", {})
    mode = "sequential" if "seq" in modes and "agg" not in modes else "parallel"
    return {
        "version": 0,
        "historical": True,
        "namespace": p.parent.name,
        "mode": mode,
        "k": int(scale[1]) if scale else 1,
        "tasks": [tasks[t] for t in order if t in tasks],
        "solver_model": cfg.get("executor", "unknown"),
        "improver_model": cfg.get("improver", "unknown"),
        "order_seed": cfg.get("seed"),
    }


def task_dir(root, namespace, task_id):
    return Path(root) / "ecom/adapt" / namespace / task_id.replace("/", "__")


def episode_dir(root, namespace, task_id, label):
    rid = f"{namespace}/{task_id}/{label}"
    key = hashlib.sha256(f"{task_id}\0{rid}".encode()).hexdigest()[:24]
    return Path(root) / "ecom/episodes" / key


def _events(path):
    text, _ = tail(path)
    result = []
    for line in text.splitlines():
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                result.append(value)
        except ValueError:
            continue
    return result


def snapshot(root, namespace=None):
    root = Path(root).resolve()
    plan = load_plan(root, namespace)
    ns = plan["namespace"]
    token = "seq" if plan["mode"] == "sequential" else "agg"
    run_ns = ns.replace("-agg", "-seq", 1) if token == "seq" else ns
    nsdir = root / "ecom/adapt" / ns
    state = read_json(root / "job-status.json", {})
    managed = not plan.get("historical")
    alive = locked(root) if managed else False
    status = state.get("status", "prepared")
    if managed and status == "running" and not alive:
        status = "interrupted"
    elif plan.get("historical") and status == "running":
        status = "unknown"  # Never infer liveness from an old PID or stale JSON.
    selected = {r["task_id"]: r for r in read_json(nsdir / "judged.json", [])}
    bases = {}
    for row in read_json(nsdir / "base_judged.json", []):
        score = row.get("score")
        if score is not None:
            bases.setdefault(row["task"], []).append(
                (row.get("label"), score.get("final_net_assets"))
            )
    latest = mtime(root / "run.log")
    variant = "oc" + (f"-s{plan['order_seed']}" if plan.get("order_seed") is not None else "")
    library = (
        root_for(root / "ecom/library", benchmark="ecom", k=plan["k"], variant=variant) / token
    )
    tasks = []
    all_rollouts = 0
    for index, item in enumerate(plan["tasks"]):
        tid = item["id"]
        work = task_dir(root, ns, tid)
        summary = read_json(work / "summary.json", {})
        mode = summary.get("modes", {}).get(token, {})
        episodes = []
        for i in range(1, plan["k"] + 1):
            labels = [
                (f"base{i}", ns),
                (f"inv{i}", run_ns),
                (f"dom{i}", run_ns),
                (f"c{i}" if token == "seq" else f"agg{i}", run_ns),
            ]
            for label, space in labels:
                path = episode_dir(root, space, tid, label)
                complete = (path / ".adapt_complete").is_file()
                started = path.is_dir()
                log = path / "trajectory.jsonl"
                last = mtime(log)
                latest = max(latest, last)
                events = _events(log) if started and not complete else []
                event = events[-1] if events else {}
                episodes.append(
                    {
                        "label": label,
                        "status": "complete"
                        if complete
                        else "running"
                        if started and alive and state.get("task_id") == tid
                        else "incomplete"
                        if started
                        else "pending",
                        "updated_at": last or None,
                        "last_event": event.get("type"),
                    }
                )
                all_rollouts += int(complete)
        errors = [_redact(str(error)) for error in summary.get("errors", [])]
        fallbacks = mode.get("fell_back", [])
        task_status = "pending"
        active = alive and state.get("task_id") == tid and state.get("stage") == "run"
        if summary:
            task_status = "complete" if summary.get("status") == "ok" else "failed"
            if fallbacks and task_status == "complete":
                task_status = "warning"
        elif active:
            task_status = "running"
        elif any(e["status"] != "pending" for e in episodes):
            task_status = "incomplete"
        row = selected.get(tid, {})
        raw = row.get("raw") or {}
        completed_labels = {e["label"] for e in episodes if e["status"] == "complete"}
        scores = [
            v for label, v in bases.get(tid, []) if v is not None and label in completed_labels
        ]
        base_score = sum(scores) / len(scores) if scores else None
        graded = row.get("graded") and task_status in ("complete", "warning")
        final_score = raw.get("final_net_assets") if graded else None
        domain = library / domain_dir(item.get("domain", tid.split("/")[0]))
        entry = domain / "harnesses" / slug(tid)
        selection = summary.get("selections", {}).get(token, {})
        experience = has_experience(selection.get("experience"))
        experience = (
            experience
            and (entry / "EXPERIENCE.md").is_file()
            and (entry / "harness").is_dir()
            and (domain / "INDEX.md").is_file()
        )
        declared = selection.get("reflection_status")
        reflection_error = _redact(str(selection.get("reflection_error") or ""))
        if declared == "no_entry":
            reflection_status = (
                "no_entry"
                if selection.get("ok") and not selection.get("experience") and not reflection_error
                else "error"
            )
        elif declared == "error":
            reflection_status = "error"
        elif experience:
            reflection_status = "written"
        elif declared in ("distilled", "written"):
            reflection_status = "missing"
            reflection_error = "Distilled experience is missing from the library"
        elif declared == "pending" or not summary:
            reflection_status = "pending"
        else:
            # Empty legacy experience cannot prove an intentional abstention.
            reflection_status = "unknown"
        tasks.append(
            {
                "id": tid,
                "index": index + 1,
                "seed": item["master_seed"],
                "hours": item.get("horizon_steps", 8760),
                "status": task_status,
                "active": active,
                "fallbacks": fallbacks,
                "errors": errors,
                "stages": [
                    {
                        k: _redact(str(s.get(k) or "")) if k == "error" else s.get(k)
                        for k in ("label", "fell_back", "kept_base", "attempts", "error")
                    }
                    for s in mode.get("stages", [])
                ],
                "episodes": episodes,
                "base_assets": base_score,
                "final_assets": final_score,
                "delta": final_score - base_score
                if final_score is not None and base_score is not None
                else None,
                "alive": raw.get("is_alive") if graded else None,
                "net_profit": raw.get("net_profit") if graded else None,
                "experience": experience,
                "reflection_status": reflection_status,
                "reflection_error": reflection_error,
            }
        )
    calls = []
    audit = root / "ecom/adapt/_audit" / ns
    # Only the current task can own live audit sessions: tasks in this domain run sequentially.
    task_started = state.get("task_started_at", state.get("started_epoch", 0))
    for p in sorted(audit.glob("*.meta.attempts/*/process.json")):
        proc = read_json(p, {})
        out = p.parent / "stdout.jsonl"
        last = max(mtime(out), mtime(p))
        latest = max(latest, last)
        is_live = (
            alive and proc.get("status") in ("running", "starting") and mtime(p) >= task_started
        )
        if not is_live:
            continue
        label = re.sub(r"\.[a-f0-9]+\.meta\.attempts$", "", p.parent.parent.name)
        label = re.sub(r"\.a\d+$", "", label)
        events = _events(out)
        event = events[-1] if events else {}
        part = event.get("part", {})
        calls.append(
            {
                "label": label,
                "phase": proc.get("phase", "unknown"),
                "last_event": part.get("tool") or event.get("type"),
                "updated_at": last,
                "source": str(out.relative_to(root)),
                "log_source": "call-" + hashlib.sha256(str(out).encode()).hexdigest()[:12],
            }
        )
    phase = state.get("stage", "unknown")
    if calls and phase == "run":
        names = [c["label"] for c in calls]
        phase = (
            "reflect"
            if any(n.startswith("experience") for n in names)
            else ("compose" if any(n.startswith(("agg", "c")) for n in names) else "propose")
        )
    elif phase == "run":
        active_task = next((t for t in tasks if t["active"]), None)
        running = [
            e["label"] for e in (active_task or {}).get("episodes", []) if e["status"] == "running"
        ]
        if running:
            phase = (
                "baseline"
                if all(s.startswith("base") for s in running)
                else ("final" if any(s.startswith(("agg", "c")) for s in running) else "probe")
            )
    started_epoch = state.get("started_epoch")
    finished_epoch = state.get("finished_epoch")
    if not started_epoch and state.get("started_at"):
        try:
            from datetime import datetime

            started_epoch = datetime.fromisoformat(state["started_at"]).timestamp()
            if state.get("finished_at"):
                finished_epoch = datetime.fromisoformat(state["finished_at"]).timestamp()
        except (ValueError, TypeError):
            pass
    finished = sum(t["status"] in ("complete", "warning", "failed") for t in tasks)
    fallback_count = sum(len(t["fallbacks"]) for t in tasks)
    return {
        "root": str(root),
        "namespace": ns,
        "server_time": time.time(),
        "status": status,
        "worker_alive": alive,
        "phase": phase,
        "started_at": state.get("started_at"),
        "finished_at": state.get("finished_at"),
        "elapsed_seconds": max(0, (finished_epoch or time.time()) - started_epoch)
        if started_epoch and (alive or finished_epoch)
        else None,
        "last_activity": latest or None,
        "stale": alive and bool(latest) and time.time() - latest > 120,
        "error": _redact(str(state.get("error") or "")),
        "verification": read_json(root / "verification.json"),
        "config": {
            key: plan.get(key)
            for key in ("solver_model", "improver_model", "k", "mode", "order_seed")
        },
        "counts": {
            "total": len(tasks),
            "finished": finished,
            "complete": sum(t["status"] == "complete" for t in tasks),
            "failed": sum(t["status"] == "failed" for t in tasks),
            "fallbacks": fallback_count,
            "rollouts": all_rollouts,
            "planned_rollouts": len(tasks) * 4 * plan["k"],
        },
        "reflection_counts": {
            key: sum(t["reflection_status"] == key for t in tasks)
            for key in ("written", "no_entry", "error", "missing", "unknown", "pending")
        },
        "tasks": tasks,
        "calls": calls,
    }


def _redact(text):
    # Never dump the environment. Redact credential-shaped values if they occur in a tool/log.
    for name, value in os.environ.items():
        if len(value) >= 6 and any(
            word in name.upper() for word in ("API_KEY", "TOKEN", "PASSWORD", "SECRET")
        ):
            text = text.replace(value, "[REDACTED]")
    text = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)(https?://)[^/\s:@]+:[^/\s@]+@", r"\1[REDACTED]@", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}", "[REDACTED]", text)
    return re.sub(
        r"""(?i)(["']?(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret)["']?\s*[:=]\s*["']?)([^"'\s,}\]]+)""",
        r"\1[REDACTED]",
        text,
    )


def log_view(root, source="run", task=None, namespace=None):
    root = Path(root).resolve()
    plan = load_plan(root, namespace)
    sources = {"run": root / "run.log", "report": root / "report.txt"}
    if task:
        if task not in {t["id"] for t in plan["tasks"]}:
            raise ValueError("Unknown task")
        run_ns = plan["namespace"]
        if plan["mode"] == "sequential":
            run_ns = run_ns.replace("-agg", "-seq", 1)
        for i in range(1, plan["k"] + 1):
            for label in (
                f"base{i}",
                f"inv{i}",
                f"dom{i}",
                f"agg{i}" if plan["mode"] == "parallel" else f"c{i}",
            ):
                ns = plan["namespace"] if label.startswith("base") else run_ns
                sources[label] = episode_dir(root, ns, task, label) / "trajectory.jsonl"
    live = snapshot(root, namespace)
    for call in live["calls"]:
        sources[call["log_source"]] = root / call["source"]
    if source not in sources:
        raise ValueError("Unknown log source")
    path = sources[source]
    if not path.resolve().is_relative_to(root):
        raise ValueError("Log path is outside the experiment")
    text, truncated = tail(path)
    if path.suffix == ".jsonl":
        lines = []
        for event in _events(path)[-80:]:
            part = event.get("part", {})
            state = part.get("state", {})
            stamp = event.get("timestamp")
            clock = (
                time.strftime("%H:%M:%S", time.localtime(stamp / 1000))
                if isinstance(stamp, (float, int))
                else ""
            )
            if part.get("tool"):
                detail = json.dumps(state.get("input", {}), ensure_ascii=False)[:420]
                output = str(state.get("output", ""))[:600]
                lines.append(
                    f"{clock}  {part['tool']}  {state.get('status', '')}\n{detail}\n{output}"
                )
            elif event.get("type") in ("text", "error"):
                lines.append(
                    f"{clock}  {event['type']}\n{str(part.get('text') or event.get('error', ''))[:1200]}"
                )
            else:
                lines.append(f"{clock}  {event.get('type', 'event')}")
        text = "\n\n".join(lines)
    return {
        "source": source,
        "text": _redact(text),
        "truncated": truncated,
        "updated_at": mtime(path) or None,
        "sources": [
            {
                "id": key,
                "label": next(
                    ("改进器 · " + c["label"] for c in live["calls"] if c["log_source"] == key), key
                ),
            }
            for key in sources
        ],
    }


def make_server(root, port=8765, namespace=None, socket_fd=None):
    root = Path(root).resolve()
    load_plan(root, namespace)  # Fail before binding if the root is not an experiment.

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _send(self, code, body, content_type="application/json; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            port = self.server.server_address[1]
            allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
            origin = self.headers.get("Origin")
            if self.headers.get("Host") not in allowed or (
                origin and origin not in {f"http://{h}" for h in allowed}
            ):
                self._send(403, b'{"error":"Loopback, same-origin requests only"}')
                return
            url = urlsplit(self.path)
            query = parse_qs(url.query)
            try:
                if url.path in ("/", "/index.html"):
                    self._send(200, ASSET.read_bytes(), "text/html; charset=utf-8")
                    return
                if url.path == "/api/status":
                    value = snapshot(root, namespace)
                elif url.path == "/api/log":
                    value = log_view(
                        root,
                        query.get("source", ["run"])[0],
                        query.get("task", [None])[0],
                        namespace,
                    )
                else:
                    self._send(
                        404,
                        b'<!doctype html><title>Not found</title><p>Unknown monitor page. <a href="/">Return to monitor</a></p>',
                        "text/html; charset=utf-8",
                    )
                    return
                self._send(200, json.dumps(value, ensure_ascii=False, allow_nan=False).encode())
            except ValueError as exc:
                self._send(400, json.dumps({"error": str(exc)}).encode())
            except (OSError, KeyError, TypeError):
                self._send(503, b'{"error":"Artifacts unavailable, retry shortly"}')

        def do_POST(self):
            self._send(405, b'{"error":"Read-only monitor"}')

        do_PUT = do_DELETE = do_PATCH = do_POST

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler, bind_and_activate=socket_fd is None)
    server.daemon_threads = True
    if socket_fd is not None:
        server.socket.close()
        server.socket = socket.socket(fileno=socket_fd)
        server.server_address = server.socket.getsockname()
    return server
