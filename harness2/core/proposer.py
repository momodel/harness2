# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Improver sessions with file access in an isolated context workspace.

OpenCode reads output/answer.md or the final message; the Claude CLI reads the
final message. Prompts use stdin, and empty replies raise an error.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .. import config
from . import execution_env as EE
from .spec import ChannelSpec

SYSTEM_ARG_MAX = 120_000


MAX_CONCURRENCY = int(config.PROPOSER_CONCURRENCY)
_SEM = threading.BoundedSemaphore(MAX_CONCURRENCY) if MAX_CONCURRENCY > 0 else None


_DROP_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_BEDROCK_BASE_URL",
    "CLAUDE_CODE_USE_BEDROCK",
)


def _phase(logs: Path, phase: str, **fields) -> None:
    logs.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = logs / "process.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**data, "phase": phase, **fields}, indent=2) + "\n")


def _run_logged(command, *, cwd, env, timeout, logs):
    """Stream evidence to disk and reap the entire CLI/tool process group.

    Logs may contain task data. Keep them under a private per-attempt directory;
    never include credentials/environment values in the diagnostic summary.
    """
    _phase(logs, "model_cli", status="starting")
    started = time.monotonic()
    process = None
    status = "launch_error"
    try:
        with (logs / "stdout.jsonl").open("w") as out, (logs / "stderr.log").open("w") as err:
            process = subprocess.Popen(
                command, cwd=cwd, env=env, stdout=out, stderr=err,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
            _phase(logs, "model_cli", status="running", pid=process.pid)
            try:
                process.wait(timeout=timeout)
                status = "exited"
            except subprocess.TimeoutExpired:
                status = "timeout"
                raise
            finally:
                # Also clean up descendants after a normal leader exit.
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(process.pid, sig)
                    except ProcessLookupError:
                        break
                    if sig == signal.SIGTERM:
                        try:
                            process.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            pass
                process.wait()
    finally:
        last = {}
        stdout = (logs / "stdout.jsonl").read_text(errors="replace")
        for line in reversed(stdout.splitlines()):
            try:
                event = json.loads(line)
                part = event.get("part") or {}
                last = {"type": event.get("type"), "timestamp": event.get("timestamp"),
                        "tool": part.get("tool"), "status": (part.get("state") or {}).get("status")}
                break
            except (ValueError, AttributeError):
                continue
        _phase(logs, "model_cli", status=status, wall_s=round(time.monotonic() - started, 3),
               returncode=process.returncode if process is not None else None, last_event=last)
    return subprocess.CompletedProcess(command, process.returncode, stdout,
                                      (logs / "stderr.log").read_text(errors="replace"))


# Opt-in Bun preload: metadata only. Never inspect headers, bodies or URL paths.
# A fetch_call is a JS invocation, NOT proof of bytes sent or provider acceptance.
_FETCH_TRACE_JS = r"""
import { appendFileSync } from "node:fs";
const file = process.env.HARNESS2_HTTP_TRACE;
const started = performance.now();
let serial = 0;
function record(event, fields = {}) {
  try {
    appendFileSync(file, JSON.stringify({event, pid: process.pid,
      elapsed_ms: Math.round(performance.now() - started), ...fields}) + "\n", {mode: 0o600});
  } catch {} // Diagnostics must not change request success/failure.
}
function errorKind(e) {
  return ["AbortError", "TimeoutError", "TypeError", "Error"].includes(e?.name) ? e.name : "Other";
}
const original = globalThis.fetch;
record("preload_ready");
globalThis.fetch = async function(input, init) {
  const id = ++serial;
  let origin = "unknown";
  try { origin = new URL(input instanceof Request ? input.url : String(input)).origin; } catch {}
  const rawMethod = init?.method ?? (input instanceof Request ? input.method : "GET");
  const method = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"].includes(rawMethod)
    ? rawMethod : "OTHER";
  record("fetch_call", {id, origin, method});
  let response;
  try { response = await original.call(globalThis, input, init); }
  catch (e) { record("fetch_error", {id, kind: errorKind(e)}); throw e; }
  record("response_headers", {id, status: response.status});
  if (!response.body) { record("body_end", {id}); return response; }
  const reader = response.body.getReader();
  let first = true;
  // Pull only when consumed; no tee, SSE decoding, buffering of the full body, or retries.
  const body = new ReadableStream({
    async pull(controller) {
      try {
        const {done, value} = await reader.read();
        if (done) { record("body_end", {id}); controller.close(); return; }
        if (first && value.byteLength) {
          first = false;
          record("first_body_chunk", {id, bytes: value.byteLength});
        }
        controller.enqueue(value);
      } catch (e) { record("body_error", {id, kind: errorKind(e)}); controller.error(e); }
    },
    cancel(reason) { record("body_cancel", {id}); return reader.cancel(reason); }
  }, {highWaterMark: 0});
  const wrapped = new Response(body, {
    status: response.status, statusText: response.statusText, headers: response.headers
  });
  for (const key of ["url", "redirected", "type"]) {
    Object.defineProperty(wrapped, key, {value: response[key]});
  }
  return wrapped;
};
"""


def _retain_opencode_state(workspace: Path, logs: Path, *, database: bool) -> None:
    """Preserve native logs (and opt-in SQLite snapshot) before deleting the workspace.

    Only inspect private per-attempt XDG roots, never host HOME/auth/config. Snapshot
    after process-group cleanup so WAL data is included without copying live sidecars.
    Retention failures are diagnostic evidence, not a replacement for the original error.
    """
    artifacts, errors = [], []
    for root in (workspace / ".harness2-env/data/opencode", workspace / ".ocdata/opencode"):
        sources = list((root / "log").glob("*.log"))
        if database:
            sources += list(root.glob("opencode*.db"))
        for source in sources:
            if source.is_symlink() or not source.resolve().is_relative_to(workspace.resolve()):
                continue
            dest = logs / "native" / source.name
            try:
                dest.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                if source.suffix == ".db":
                    deadline = time.monotonic() + 3

                    def progress(*_):
                        if time.monotonic() > deadline:
                            raise TimeoutError("SQLite snapshot deadline")

                    with (
                        contextlib.closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True,
                                                          timeout=1)) as src,
                        contextlib.closing(sqlite3.connect(dest)) as dst,
                    ):
                        src.backup(dst, pages=128, progress=progress, sleep=0.05)
                else:
                    shutil.copyfile(source, dest)
                dest.chmod(0o600)
                artifacts.append(str(dest.relative_to(logs)))
            except Exception as exc:
                dest.unlink(missing_ok=True)
                errors.append({"file": source.name, "kind": type(exc).__name__})
    state = json.loads((logs / "process.json").read_text())
    _phase(logs, state["phase"], runtime_evidence={"files": artifacts, "errors": errors,
                                               "database_requested": database})


def _obj(d: dict, key: str) -> dict:
    """`d[key]` when it is a dict, else {}."""
    v = d.get(key)
    return v if isinstance(v, dict) else {}


class ProposerError(RuntimeError):
    """A call did not produce a usable reply. Never swallowed into an empty string."""

    def __init__(self, message: str, *, attempts: int = 0, cost_usd: float = 0.0):
        super().__init__(message)
        self.attempts = attempts
        self.cost_usd = cost_usd


@dataclass
class Reply:
    """One session: what came back, what it cost, and where the audit trail is."""

    label: str
    text: str
    ok: bool
    cost_usd: float = 0.0
    wall_s: float = 0.0
    attempts: int = 0
    cached: bool = False
    channel: str = ""
    error: str = ""
    audit: Path | None = None
    workspace: Path | None = None

    def __bool__(self) -> bool:
        return self.ok


@dataclass
class Proposer:
    """Runs agentic sessions on one channel, with an audit trail and a cache."""

    spec: ChannelSpec
    audit_dir: Path | None = None
    attempts: int = 2
    diagnostics: bool = field(default=False, kw_only=True)
    project: str = field(default_factory=lambda: config.VERTEX_PROJECT)
    region: str = field(default_factory=lambda: config.VERTEX_REGION)
    _calls: list[Reply] = field(default_factory=list)

    def __post_init__(self) -> None:
        if EE.enabled() and self.spec.kind != "opencode":
            raise config.ConfigError("Clean improver sessions currently require OpenCode")
        if not self.spec.model.strip():
            raise ValueError("Set HARNESS2_IMPROVER_MODEL before starting model sessions")

    def env(self) -> dict[str, str]:
        e = {k: v for k, v in os.environ.items() if k not in _DROP_ENV}
        if self.spec.kind == "claude_cli":
            e.update(
                {
                    "CLAUDE_CODE_USE_VERTEX": "1",
                    "ANTHROPIC_VERTEX_PROJECT_ID": self.project,
                    "CLOUD_ML_REGION": self.region,
                    "ANTHROPIC_MODEL": self.spec.model,
                }
            )
        else:
            e.update(
                {
                    "GOOGLE_VERTEX_PROJECT": self.project,
                    "GOOGLE_VERTEX_LOCATION": self.region,
                    "VERTEXAI_LOCATION": self.region,
                }
            )
        return e

    def call(
        self,
        mission: str,
        context_dir: Path | str | None,
        label: str,
        *,
        system: str = "",
        timeout: int | None = None,
        max_attempts: int | None = None,
        harvest: str | None = None,
    ) -> Reply:
        """Run one session. Raises ProposerError if no usable reply after `attempts`.

        `harvest` names a subpath of the session's working directory to copy out before that
        directory is destroyed -- e.g. "context/harness", which is how an edited tree comes back.
        """

        cache_key = _call_fingerprint(label, mission, system, context_dir,
                                      diagnostics=self.diagnostics)
        cached = self._cached(label, cache_key, require_workspace=bool(harvest))
        if cached is not None:
            self._calls.append(cached)
            return cached

        wall = int(timeout or self.spec.timeout)
        tries = max(1, int(self.attempts if max_attempts is None else max_attempts))
        last = ""
        spent_usd = 0.0
        t0 = time.time()
        for attempt in range(1, tries + 1):
            if attempt > 1:
                time.sleep(10 * (attempt - 1))
            tmp = Path(tempfile.mkdtemp(prefix="harness2-prop-"))
            paths = self._paths(label, cache_key)
            logs = (paths["meta"].with_suffix(".attempts") if paths else tmp / "attempts") / f"{tmp.name}-a{attempt}"
            try:
                _phase(logs, "context_copy", status="starting")
                if context_dir and Path(context_dir).is_dir():
                    shutil.copytree(context_dir, tmp / "context", dirs_exist_ok=True)
                with _SEM if _SEM is not None else contextlib.nullcontext():
                    if self.spec.kind == "claude_cli":
                        text, cost, err = self._claude_cli(tmp, mission, system, wall)
                    else:
                        text, cost, err = self._opencode(tmp, mission, system, wall, logs=logs)
                spent_usd += cost
                if not text.strip():
                    last = err or "empty reply"
                    continue
                r = Reply(
                    label=label,
                    text=text,
                    ok=True,
                    cost_usd=spent_usd,
                    wall_s=round(time.time() - t0, 1),
                    attempts=attempt,
                    channel=self.spec.kind,
                )
                if harvest:
                    r.workspace = self._harvest(tmp, harvest, label, cache_key)
                    if r.workspace is None:
                        _drop_workspace(self._paths(label, cache_key))
                self._write_audit(r, mission, system, cache_key)
                self._calls.append(r)
                return r
            except subprocess.TimeoutExpired as exc:
                phase = json.loads((logs / "process.json").read_text()).get("phase")
                last = f"timeout after {exc.timeout}s during {phase}"
                if (logs / "stdout.jsonl").is_file():
                    spent_usd += _opencode_cost((logs / "stdout.jsonl").read_text(errors="replace"))
                _phase(logs, phase, status="timeout", error=last)
            except Exception as exc:
                last = f"{type(exc).__name__}: {exc}"
                phase = json.loads((logs / "process.json").read_text()).get("phase")
                _phase(logs, phase, status="error", error=last[:1000])
            finally:
                try:
                    if self.spec.kind == "opencode":
                        try:
                            _retain_opencode_state(tmp, logs, database=self.diagnostics)
                        except Exception as exc:
                            # Even unexpected retention errors must not replace a
                            # timeout, usable reply, or other original outcome.
                            with contextlib.suppress(OSError, ValueError):
                                state = json.loads((logs / "process.json").read_text())
                                _phase(logs, state["phase"], retention_error=type(exc).__name__)
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)

        r = Reply(
            label=label,
            text="",
            ok=False,
            wall_s=round(time.time() - t0, 1),
            attempts=tries,
            channel=self.spec.kind,
            error=last,
            cost_usd=spent_usd,
        )
        self._calls.append(r)
        self._write_audit(r, mission, system, cache_key)
        raise ProposerError(
            f"[{label}] {self.spec.kind}/{self.spec.model}: {last}",
            attempts=tries,
            cost_usd=spent_usd,
        )

    def _claude_cli(
        self, cwd: Path, mission: str, system: str, wall: int
    ) -> tuple[str, float, str]:
        cli = shutil.which("claude") or "claude"
        cmd = [
            cli,
            "--dangerously-skip-permissions",
            "-p",
            "--output-format",
            "json",
            "--effort",
            self.spec.effort,
        ]
        stdin = mission
        if system:
            if len(system) > SYSTEM_ARG_MAX:
                stdin = f"# SYSTEM INSTRUCTIONS\n\n{system}\n\n# END\n\n{mission}"
                cmd += [
                    "--append-system-prompt",
                    "Follow the SYSTEM INSTRUCTIONS block at the top of the message.",
                ]
            else:
                cmd += ["--append-system-prompt", system]
        p = subprocess.run(
            cmd,
            input=stdin,
            cwd=cwd,
            env=self.env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=wall,
        )
        payload = _parse_json(p.stdout)
        text = str(payload.get("result") or payload.get("text") or "")
        err = ""
        if payload.get("is_error") or payload.get("subtype") not in (None, "success"):
            err = f"CLI error: {payload.get('subtype')} {str(text)[:200]}"
            text = ""
        elif p.returncode != 0 and not text.strip():
            err = f"rc={p.returncode}: {(p.stderr or '')[-300:]}"
        return text, float(payload.get("total_cost_usd") or 0.0), err

    def _opencode(self, cwd: Path, mission: str, system: str, wall: int, *, logs: Path) -> tuple[str, float, str]:
        (cwd / "MISSION.md").write_text(
            (f"{system}\n\n---\n\n" if system else "") + mission, encoding="utf-8"
        )
        out = cwd / "output"
        out.mkdir(exist_ok=True)

        (cwd / "opencode.json").write_text(
            json.dumps({"agent": {"build": {"steps": self.spec.steps}}}), encoding="utf-8"
        )
        prompt = (
            "Read MISSION.md first. Do what it asks. Reply with the requested output as "
            "your final message AND save it verbatim to output/answer.md."
        )
        oc = _opencode_dir()

        _phase(logs, "prepare", status="running")
        if EE.enabled():
            env = EE.prepare(
                cwd,
                model=self.spec.model,
                tool_dirs=(
                    config.MERCHANTBENCH_PYTHON.parent,
                    config.BUN.parent,
                ),
            )
            audit = cwd / "environment.json"
            _phase(logs, "startup_audit", status="running")
            EE.audit(cwd, env, audit)
            if self.audit_dir:
                dest = self.audit_dir / "environments" / cwd.name
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy(audit, dest / "environment.json")
        else:
            env = self.env()
        env["OPENCODE_PERMISSION"] = json.dumps(
            {"external_directory": {"*": "deny", str(cwd): "allow", f"{cwd}/**": "allow"}}
        )
        if not EE.enabled():
            env["XDG_DATA_HOME"] = str(cwd / ".ocdata")
            env["XDG_STATE_HOME"] = str(cwd / ".ocstate")

        entry = "src/index.ts"
        if self.diagnostics:
            trace_script = (logs / "fetch-trace.mjs").resolve()
            # The pinned package's bunfig preload overrides CLI --preload under
            # --cwd. Import the unchanged native entry *after* the trace instead.
            native_entry = (oc / "packages/opencode/src/index.ts").resolve().as_uri()
            trace_script.write_text(_FETCH_TRACE_JS + f"\nawait import({json.dumps(native_entry)});\n")
            trace_script.chmod(0o600)
            env["HARNESS2_HTTP_TRACE"] = str((logs / "transport.jsonl").resolve())
            entry = str(trace_script)
        cmd = [
            str(config.BUN),
            "run",
            "--cwd",
            str(oc / "packages" / "opencode"),
            "--conditions=browser",
            entry,
            "run",
            prompt,
            "--dir",
            str(cwd),
            "--model",
            self.spec.model,
            "--format",
            "json",
        ]
        if self.spec.effort:
            cmd += ["--variant", self.spec.effort]
        if self.diagnostics:
            cmd += ["--print-logs", "--log-level", "DEBUG"]
        p = _run_logged(cmd, cwd=cwd, env=env, timeout=wall, logs=logs)
        cost = _opencode_cost(p.stdout)
        ans = out / "answer.md"
        if ans.is_file() and ans.read_text(errors="replace").strip():
            return ans.read_text(errors="replace"), cost, ""
        text, err = _opencode_stdout(p.stdout)
        if text.strip():
            return text, cost, ""
        tail = _strip_oc_banner(p.stderr or "").strip()[-300:]
        return "", cost, f"rc={p.returncode}: {err or tail or 'empty reply'}"

    def _harvest(self, tmp: Path, sub: str, label: str, cache_key: str = "") -> Path | None:
        """Copy `tmp/<sub>` somewhere durable before the session directory is removed."""
        src = tmp / sub
        if not src.is_dir():
            return None
        dest = Path(self.audit_dir or tempfile.mkdtemp(prefix="harness2-harvest-"))
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
        if cache_key:
            safe = f"{safe}.{cache_key}"
        dest = dest / f"{safe}.workspace"
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(src, dest)
        return dest

    def _paths(self, label: str, cache_key: str = "") -> dict[str, Path] | None:
        if not self.audit_dir:
            return None
        d = Path(self.audit_dir)
        d.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
        if cache_key:
            safe = f"{safe}.{cache_key}"
        return {k: d / f"{safe}.{k}.md" for k in ("mission", "system", "response")} | {
            "meta": d / f"{safe}.meta.json",
            "workspace": d / f"{safe}.workspace",
        }

    def _cached(
        self, label: str, cache_key: str = "", *, require_workspace: bool = False
    ) -> Reply | None:
        p = self._paths(label, cache_key)
        if not p or not p["response"].is_file():
            return None
        if require_workspace and not p["workspace"].is_dir():
            return None
        txt = p["response"].read_text(encoding="utf-8", errors="replace")
        if not txt.strip():
            return None
        return Reply(
            label=label,
            text=txt,
            ok=True,
            cost_usd=0.0,
            cached=True,
            channel=self.spec.kind,
            audit=p["response"],
            workspace=p["workspace"] if p["workspace"].is_dir() else None,
        )

    def _write_audit(self, r: Reply, mission: str, system: str, cache_key: str = "") -> None:
        p = self._paths(r.label, cache_key)
        if not p:
            return
        p["mission"].write_text(mission, encoding="utf-8")
        if system:
            p["system"].write_text(system, encoding="utf-8")
        p["response"].write_text(r.text, encoding="utf-8")
        p["meta"].write_text(
            json.dumps(
                {
                    "label": r.label,
                    "ok": r.ok,
                    "channel": r.channel,
                    "model": self.spec.model,
                    "effort": self.spec.effort,
                    "diagnostics_enabled": self.diagnostics,
                    "environment": EE.identity(),
                    "attempts": r.attempts,
                    "wall_s": r.wall_s,
                    "cost_usd": r.cost_usd,
                    "error": r.error,
                    "diagnostics": [
                        str(f) for f in sorted(p["meta"].with_suffix(".attempts").glob("*/process.json"))
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        r.audit = p["response"]


def _drop_workspace(paths: dict[str, Path] | None) -> None:
    """Remove a stale harvested workspace so a later cache read cannot serve it."""
    if paths:
        shutil.rmtree(paths["workspace"], ignore_errors=True)


def _opencode_dir() -> Path:
    """The upstream OpenCode checkout to run through bun: config.OPENCODE_DIR."""
    oc = config.OPENCODE_DIR
    if not (oc / "packages" / "opencode" / "src" / "index.ts").is_file():
        raise FileNotFoundError(
            f"no OpenCode checkout at {oc} (config.OPENCODE_DIR); run "
            f"scripts/setup_benchmark.py <benchmark> to clone the pinned commit there"
        )
    return oc


def _parse_json(stdout: str) -> dict:
    """The claude CLI's one JSON object, tolerating banners and trailing noise around it.

    Not for opencode stdout, which is a JSONL event stream: use `_opencode_stdout`.
    """
    s = (stdout or "").strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        for line in reversed(s.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        i, j = s.find("{"), s.rfind("}")
        if 0 <= i < j:
            try:
                return json.loads(s[i : j + 1])
            except json.JSONDecodeError:
                pass
    return {"result": s}


_OC_BANNER = re.compile(
    r"^(?:Performing one time database migration[^\n]*|sqlite-migration:[^\n]*"
    r"|Database migration complete\.)[ \t]*$",
    re.M,
)


def _strip_oc_banner(s: str) -> str:
    """Drop the migration banner lines from OpenCode output. No-op when they are absent."""
    return _OC_BANNER.sub("", s or "")


def _opencode_stdout(stdout: str) -> tuple[str, str]:
    """Fold `opencode run --format json` into (final assistant text, error text).

    The stream is one raw event per line and assistant prose lives at part.text of a `text`
    event. All text events are joined, so a trailing terminator event cannot swallow the
    answer; `error` events become the error string. Plain-text stdout is returned as-is.
    """
    texts: list[str] = []
    errors: list[str] = []
    saw_json = False
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict):
            continue
        saw_json = True
        part = _obj(ev, "part")
        ptype = part.get("type") or ev.get("type")
        if ptype == "text":
            t = part.get("text")
            if isinstance(t, str) and t.strip():
                texts.append(t)
        elif ptype == "error":
            e = ev.get("error")
            if isinstance(e, dict):
                msg = str(_obj(e, "data").get("message") or e.get("name") or "").strip()
            else:
                msg = str(e or "").strip()
            if msg:
                errors.append(msg)
    if not saw_json:
        return _strip_oc_banner(stdout).strip(), ""
    return "\n\n".join(texts), "; ".join(errors)[-300:]


def _opencode_cost(stdout: str) -> float:
    """Sum `part.cost` over the step_finish events of an `opencode run --format json` stream."""
    total = 0.0
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict):
            continue
        part = _obj(ev, "part")
        if (part.get("type") or ev.get("type")) in ("step-finish", "step_finish"):
            c = part.get("cost")
            if isinstance(c, (int, float)):
                total += float(c)
    return round(total, 6)


def _call_fingerprint(label: str, mission: str, system: str, context_dir: Path | str | None,
                      *, diagnostics: bool = False) -> str:
    """Content-address one call, including its evidence and its mode/task workspace path.

    The resolved path is part of the key: the two modes can have byte-identical k=1 evidence
    and must still be independent draws.
    """
    h = hashlib.sha256()
    if diagnostics:
        h.update(b"opencode-diagnostics-v1\0")
    if EE.enabled():
        h.update(json.dumps(EE.identity(), sort_keys=True).encode())
    for value in (label, mission, system):
        h.update(value.encode("utf-8", errors="replace"))
        h.update(b"\0")
    if context_dir:
        root = Path(context_dir)
        h.update(str(root.resolve()).encode("utf-8", errors="replace"))
        h.update(b"\0")
        if root.is_dir():
            for p in sorted(root.rglob("*")):
                if not p.is_file() or "__pycache__" in p.parts:
                    continue
                h.update(str(p.relative_to(root)).encode("utf-8", errors="replace"))
                h.update(b"\0")
                try:
                    with p.open("rb") as fh:
                        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                            h.update(chunk)
                except OSError as exc:
                    h.update(f"<unreadable:{type(exc).__name__}:{exc}>".encode())
                h.update(b"\0")
    return h.hexdigest()[:16]
