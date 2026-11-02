"""MerchantBench controller and isolated Python entrypoints (no model calls).

The simulator and scoring module are imported only in the upstream venv. The
controller itself uses the standard library; model executors receive agent API
credentials only, never the administrative token or the measurement payload.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener


class ProtocolError(RuntimeError):
    pass


def request_json(base: str, path: str, token: str, body=None, *, timeout=30):
    data = None if body is None else json.dumps(body).encode()
    req = Request(
        base + path,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    # Loopback traffic must not go through the operator's outbound HTTP proxy.
    with build_opener(ProxyHandler({})).open(req, timeout=timeout) as response:
        return json.load(response)


def stop_process(process: subprocess.Popen) -> None:
    # Kill the process group even if its leader has exited: an executor may
    # leave shell/tool descendants alive after returning its final response.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_process(command, *, cwd, env, stdout, stderr, timeout):
    process = subprocess.Popen(
        [str(c) for c in command],
        cwd=cwd,
        env=env,
        stdout=stdout,
        stderr=stderr,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        code = process.wait(timeout=timeout)
        if code:
            raise RuntimeError(f"MerchantBench executor exited with status {code}; inspect logs")
    finally:
        stop_process(process)


def clean_environment() -> dict[str, str]:
    """An env server has no need for model/provider or private-data credentials."""
    names = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT")
    return {name: os.environ[name] for name in names if name in os.environ}


class LocalEpisode:
    """One fresh authenticated loopback simulator; trusted local use, NOT a sandbox."""

    def __init__(self, repo: Path, python: Path, state: Path, task: dict):
        self.repo, self.python, self.state, self.task = repo, python, state, task
        self.admin_token = secrets.token_urlsafe(32)
        self.process = None
        self.log = None
        self.base = ""
        self.run_id = ""
        self.agent_token = ""

    def __enter__(self):
        self.state.mkdir(parents=True, exist_ok=False)
        self.state.chmod(0o700)
        ready = self.state / "ready.json"
        self.log = (self.state / "server.log").open("w")
        server_env = clean_environment()
        server_env.update(
            {
                "MERCHANTBENCH_REQUIRE_TOKENS": "1",
                "MERCHANTBENCH_ADMIN_TOKEN": self.admin_token,
            }
        )
        self.process = subprocess.Popen(
            [
                str(self.python),
                str(Path(__file__).resolve()),
                "serve",
                "--repo",
                str(self.repo),
                "--state",
                str(self.state),
            ],
            env=server_env,
            stdout=self.log,
            stderr=self.log,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        deadline = time.monotonic() + 45
        try:
            while not ready.is_file():
                if self.process.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"MerchantBench server failed to start; see {self.state}/server.log"
                    )
                time.sleep(0.05)
            self.base = json.loads(ready.read_text())["base_url"]
            scenario = json.loads(
                subprocess.check_output(
                    [
                        str(self.python),
                        str(Path(__file__).resolve()),
                        "scenario",
                        "--repo",
                        str(self.repo),
                    ],
                    env=clean_environment(),
                    text=True,
                    timeout=30,
                )
            )
            if scenario.get("data", {}).get("source") != "synthetic":
                raise ValueError("EcomAdapter supports the public synthetic catalog only")
            scenario["run"]["master_seed"] = self.task["master_seed"]
            if self.task.get("horizon_steps") is not None:
                scenario["run"]["horizon_steps"] = self.task["horizon_steps"]
            self.horizon = int(scenario["run"]["horizon_steps"])
            self.initial_assets = float(scenario["run"]["initial_cash"]) + float(
                scenario["run"]["initial_deposit"]
            )
            created = self.admin(
                "/runs",
                {
                    "scenario": scenario,
                    "master_seed": self.task["master_seed"],
                    "auto_start": False,
                    "bootstrap_agent": "none",
                    "name": self.task["id"],
                },
            )
            self.run_id, self.agent_token = created["run_id"], created["agent_token"]
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def admin(self, path, body=None):
        return request_json(self.base, path, self.admin_token, body)

    def start(self):
        self.admin(f"/runs/{self.run_id}/start", {"interval_ms": 0})

    def agent_environment(self, base_env=None):
        env = dict(os.environ if base_env is None else base_env)
        for key in list(env):
            if key.startswith(("MERCHANTBENCH_", "REALSHOP_", "RSH_")):
                env.pop(key)
        env.update(
            {
                "MERCHANTBENCH_BASE_URL": self.base,
                "MERCHANTBENCH_RUN_ID": self.run_id,
                "MERCHANTBENCH_AGENT_ID": "agent_0",
                "MERCHANTBENCH_AGENT_TOKEN": self.agent_token,
            }
        )
        # Keep outbound model proxies, but never proxy the agent's loopback SDK.
        bypass = ",".join(
            filter(None, [env.get("NO_PROXY"), env.get("no_proxy"), "127.0.0.1,localhost"])
        )
        env["NO_PROXY"] = env["no_proxy"] = bypass
        return env

    def finish(self, timeout=120):
        """Require settlement completion; stopped/error/truncated runs are not scores."""
        deadline = time.monotonic() + timeout
        while True:
            row = self.admin(f"/runs/{self.run_id}")
            status = row.get("status")
            if status == "finished":
                break
            if status in ("stopped", "error") or time.monotonic() >= deadline:
                raise ProtocolError(
                    f"MerchantBench episode incomplete: {status}; no score recorded"
                )
            time.sleep(0.05)
        section = self.admin(f"/runs/{self.run_id}/agents/agent_0/sections/merchant")
        if not (section.get("series") or {}).get("net_assets"):
            raise ProtocolError("Finished episode has no authoritative net_assets series")
        return {
            "terminal_status": status,
            "run_id": self.run_id,
            "master_seed": self.task["master_seed"],
            "horizon_steps": self.horizon,
            "initial_assets": self.initial_assets,
            "merchant_section": section,
        }

    def __exit__(self, *_):
        if self.process is not None:
            stop_process(self.process)
        if self.log is not None:
            self.log.close()


def _upstream(repo):
    sys.path.insert(0, str(repo / "env"))
    sys.path.insert(0, str(repo))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("serve", "scenario", "score"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args(argv)
    _upstream(args.repo)
    if args.command == "scenario":
        from web.runner import load_default_scenario

        print(json.dumps(load_default_scenario()))
    elif args.command == "score":
        from eval.scoring import compute

        record = json.loads(args.input.read_text())
        print(json.dumps(compute(record["merchant_section"])))
    else:
        from web.app import create_app
        from werkzeug.serving import make_server

        app = create_app(
            db_path=str(args.state / "catalog.db"),
            runs_root=str(args.state / "runs"),
        )
        server = make_server("127.0.0.1", 0, app, threaded=True)
        (args.state / "ready.json").write_text(
            json.dumps(
                {
                    "base_url": f"http://127.0.0.1:{server.server_port}",
                }
            )
        )
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
        try:
            server.serve_forever()
        finally:
            server.server_close()
            app.registry.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
