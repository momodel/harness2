"""MerchantBench public-synthetic episodes behind the BenchmarkAdapter seam.

Uses the upstream simulator, SDK and native final-net-assets metric unchanged.
The local runtime is for trusted agents: API tokens do not sandbox shell access.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .. import config
from ..core import execution_env as EE
from ..core import runmark as RM
from ..core.spec import BenchPrompts, BenchSpec, resolve_channel_model
from ..install import codex_config_error
from ..substrate.codex import CodexSubstrate
from ..substrate.opencode import OUTPUT_TOKEN_MAX, OpencodeSubstrate
from .jobbench import render_opencode_tree
from .merchantbench_runtime import LocalEpisode, run_process
from .tasks import select

HERE = Path(__file__).resolve().parent
TASKS_FILE = HERE / "ecom_tasks.json"
RUNTIME = HERE / "merchantbench_runtime.py"
NATIVE_JUDGE = "merchantbench-native-final-net-assets-v1"

TASK_PROMPT = """Operate a persistent online merchant business using the supplied MerchantBench tools.
Maximize terminal net assets while managing sourcing, listings, pricing, orders,
settlement and supplier changes. Read the environment-issued brief and platform rules
on your first observation. The initial default capital is 2000 cash plus 1000 deposit.
This is the public synthetic catalog, not the private research dataset.

Use the fixed bridge in this workspace through Bash:
- python merchantbench.py tools
- python merchantbench.py observe
- python merchantbench.py act TOOL_NAME '{"argument": "value"}'
Read tools for the exact argument schemas. Observe before acting, use end_of_step
when done with the current activation, then observe again. On a stale-step error,
discard pending actions and observe again. Continue until observe returns
{"terminal": true}; do not stop after a single activation.
The operating horizon ends new business and agent activations, not settlement:
existing orders continue through delivery, refunds and settlement in an automatic
order-drain period. Final net assets are measured only after that drain finishes.
An order delivered or settled after the operating horizon still affects the final
score; do not treat a delivery after the horizon as zero-margin or lost revenue.
After observe returns terminal, exit successfully; do not keep acting or try to
read the controller's final measurement. A last observed balance is not the final
score. Fines are already deducted from assets; never subtract them twice.
Keep your working notes inside this workspace. A written plan is not a substitute for executing operations.
Do not inspect simulator source, server state, other runs, administrative endpoints
or measurement files. Do not modify the bridge or its SDK.
"""

PROMPTS = BenchPrompts(
    work="Operate an online merchant business with delayed orders, supplier changes and cash constraints.",
    domain_word="merchant environment",
    inv_levers="Inspect current state before mutations, retain a concise operational ledger, and verify tool results.",
    dom_levers="Track supplier availability, margin, liquidity, aging listings and delayed fulfillment obligations.",
    yardstick="Did the merchant make consistent evidence-backed decisions and follow up delayed obligations?",
    selection_checks="Compare operational coherence, source coverage, constraint handling and follow-up in the visible histories.",
    deliverable_word="operating history",
)


def load_tasks(path: Path | None = None) -> dict[str, dict]:
    path = Path(path or config.ECOM_TASKS_FILE)
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or type(data.get("version")) is not int
        or data["version"] != 1
        or data.keys() - {"version", "tasks"}
    ):
        raise ValueError(f"{path}: expected version=1 task manifest")
    items = data.get("tasks")
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path}: tasks must be a nonempty list")
    tasks = {}
    for task in items:
        if not isinstance(task, dict):
            raise ValueError(f"{path}: each task must be an object")
        task_id = task.get("id", "")
        domain = task.get("domain", "")
        if not isinstance(task_id, str) or not re.fullmatch(
            r"[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+", task_id
        ):
            raise ValueError(f"{path}: invalid task id {task_id!r}; expected domain/name")
        if domain != task_id.split("/", 1)[0]:
            raise ValueError(f"{path}: domain must match task id prefix")
        if task_id in tasks:
            raise ValueError(f"{path}: duplicate task id {task_id}")
        if type(task.get("master_seed")) is not int or task["master_seed"] < 0:
            raise ValueError(f"{path}: master_seed must be a nonnegative integer")
        horizon = task.get("horizon_steps")
        if horizon is not None and (type(horizon) is not int or horizon < 1):
            raise ValueError(f"{path}: horizon_steps must be a positive integer or null")
        unknown = task.keys() - {"id", "domain", "master_seed", "horizon_steps"}
        if unknown:
            raise ValueError(f"{path}: unsupported task fields {sorted(unknown)}")
        tasks[task_id] = dict(task)
    return dict(sorted(tasks.items()))


class EcomAdapter:
    """Fresh MerchantBench episode per rollout, with post-selection native scoring."""

    judge_identity = NATIVE_JUDGE
    requires_model_judge = False

    def __init__(self, substrate="oc", *, split_file=None, tasks_file=None, store_root=None):
        if config.ECOM_TIMEOUT < 1:
            raise ValueError("HARNESS2_ECOM_TIMEOUT must be positive")
        if substrate not in ("oc", "cx"):
            raise ValueError("MerchantBench substrate must be oc or cx")
        self.tasks_file = Path(tasks_file or config.ECOM_TASKS_FILE)
        self.split_file = split_file
        self.store = Path(store_root or config.RUNS / "ecom" / "episodes")
        self.substrate = (
            OpencodeSubstrate(launcher=self._launch)
            if substrate == "oc"
            else CodexSubstrate(launcher=self._launch)
        )
        channel = resolve_channel_model(None, None)
        self.spec = BenchSpec(
            name="ecom",
            runs_root=config.RUNS / "ecom",
            install_mode="workspace_agents_md" if substrate == "oc" else "codex_root",
            prompts=PROMPTS,
            channels={"proposer": channel, "selector": channel},
            state_deliverable=True,
            allow_subagents=substrate == "oc",
            reserved_scripts=frozenset({"merchantbench.py"}),
            leak_globs=("measurement/**", "server-state/**", "native_result.json", "auth.json"),
        )

    def tasks(self, split="all"):
        tasks = load_tasks(self.tasks_file)
        return select(
            {k: v["domain"] for k, v in tasks.items()}, split, self.split_file, benchmark="ecom"
        )

    def domain_of(self, task_id):
        return load_tasks(self.tasks_file)[task_id]["domain"]

    def reference_domain_order(self):
        return sorted({t["domain"] for t in load_tasks(self.tasks_file).values()})

    def task_prompt(self, task_id):
        task = load_tasks(self.tasks_file)[task_id]
        horizon = task.get("horizon_steps")
        duration = (
            f"{horizon} simulated hours (custom horizon)" if horizon else "365 simulated days"
        )
        return TASK_PROMPT + f"\nOperating horizon: {duration}.\n"

    def task_files(self, task_id):
        return None  # Hidden catalog/scenario details are never copied into improvement context.

    def base_harness(self):
        return HERE / "ecom_base" / self.substrate.name

    def validate_tree(self, tree):
        for path in tree:
            if Path(path).name in ("merchantbench.py", "merchantbench_tool_client.py"):
                raise ValueError(
                    "MerchantBench transport and SDK are fixed runtime files, not editable helpers"
                )

    def experiment_identity(self):
        repo = config.merchantbench_repo()
        scenario = repo / "env/scenarios/default.yaml"
        return {
            "environment": EE.identity(),
            "upstream_pin": config.DEPENDENCY_PINS["merchantbench"],
            "scenario_sha256": hashlib.sha256(scenario.read_bytes()).hexdigest(),
            "tasks": load_tasks(self.tasks_file),
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "runtime_sha256": hashlib.sha256(RUNTIME.read_bytes()).hexdigest(),
            "bridge_sha256": hashlib.sha256(
                (HERE / "merchantbench_tool.py").read_bytes()
            ).hexdigest(),
            "requirements_sha256": hashlib.sha256(
                (HERE / "merchantbench-requirements.txt").read_bytes()
            ).hexdigest(),
            "base_harness": RM.fingerprint(self.base_harness()),
            "timeout": config.ECOM_TIMEOUT,
        }

    def preflight(self):
        if EE.enabled():
            if self.substrate.name != "oc" or config.DEFAULT_PROPOSER_CHANNEL != "opencode":
                raise config.ConfigError(
                    "Clean ecom mode currently requires OpenCode executor and improver"
                )
            EE.validate_credentials()
        self.substrate.preflight()
        if self.substrate.name == "cx" and (error := codex_config_error()):
            raise RuntimeError(error)
        self.check_environment()
        load_tasks(self.tasks_file)

    @staticmethod
    def check_environment():
        py, repo = config.MERCHANTBENCH_PYTHON, config.merchantbench_repo()
        if not py.is_file():
            raise RuntimeError(
                "MerchantBench environment missing; run harness2-setup ecom --env-only"
            )
        result = subprocess.run(
            [str(py), "-c", "import flask, yaml, numpy, jieba, requests; import web.app"],
            env={**os.environ, "PYTHONPATH": str(repo / "env")},
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode:
            raise RuntimeError(
                "MerchantBench runtime dependencies missing; run harness2-setup ecom --env-only"
            )
        head = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        if head != config.DEPENDENCY_PINS["merchantbench"]:
            raise RuntimeError("MerchantBench checkout is not at the supported pin")
        dirty = subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD", "--"], text=True)
        if dirty:
            raise RuntimeError(
                "MerchantBench tracked source is modified; restore it before an experiment"
            )

    def _episode_dir(self, task_id, run_id):
        # Stable hashing avoids path traversal and slash/underscore label collisions.
        digest = hashlib.sha256(f"{task_id}\0{run_id}".encode()).hexdigest()[:24]
        return self.store / digest

    def execute(self, task_id, harness_dir, run_id, *, timeout=None):
        load_tasks(self.tasks_file)[task_id]
        run_dir = self._episode_dir(task_id, run_id)
        identity = self.experiment_identity()
        marker = RM.read_mark(run_dir) or {}
        if (
            RM.is_complete(run_dir, harness_dir=harness_dir)
            and marker.get("identity") == identity
            and self.artifacts_intact(run_dir)
        ):
            return run_dir
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        install = self.substrate.materialise(harness_dir or self.base_harness(), self.spec, {})
        try:
            self.substrate.execute(task_id, install, [run_id], timeout or config.ECOM_TIMEOUT)
        finally:
            shutil.rmtree(install.staging_dir, ignore_errors=True)
        fingerprint = self._artifact_fingerprint(run_dir)
        if fingerprint is None:
            raise RuntimeError("MerchantBench rollout artifacts are incomplete; not caching")
        RM.mark_complete(
            run_dir,
            run_id=run_id,
            harness_dir=harness_dir,
            extra={
                "identity": identity,
                "artifact_fingerprint": fingerprint,
            },
        )
        return run_dir

    @staticmethod
    def _artifact_fingerprint(run_dir):
        paths = ("measurement/episode.json", "trajectory.jsonl", "operations.jsonl")
        if (run_dir / "environment.json").is_file():
            paths += ("environment.json",)
        try:
            return hashlib.sha256(b"\0".join((run_dir / p).read_bytes() for p in paths)).hexdigest()
        except OSError:
            return None

    def artifacts_intact(self, run_dir):
        run_dir = Path(run_dir)
        mark = RM.read_mark(run_dir) or {}
        return bool(mark.get("artifact_fingerprint")) and mark[
            "artifact_fingerprint"
        ] == self._artifact_fingerprint(run_dir)

    def delivered(self, run_dir):
        return self.artifacts_intact(run_dir)

    def _launch(self, request, install):
        if len(request.run_ids) != 1:
            raise ValueError("MerchantBench launches one fresh episode at a time")
        task = load_tasks(self.tasks_file)[request.task_ref]
        run_dir = self._episode_dir(request.task_ref, request.run_ids[0])
        deadline = time.monotonic() + request.timeout
        with tempfile.TemporaryDirectory(prefix="harness2-merchant-agent-") as temp:
            workspace = Path(temp)
            if self.substrate.name == "oc":
                render_opencode_tree(install.staging_dir, workspace / "rendered", {})
                shutil.copytree(workspace / "rendered", workspace, dirs_exist_ok=True)
                shutil.rmtree(workspace / "rendered")
            else:
                shutil.copytree(install.staging_dir, workspace, dirs_exist_ok=True)
            shutil.copy(HERE / "merchantbench_tool.py", workspace / "merchantbench.py")
            shutil.copy(
                config.merchantbench_repo() / "agent/sdk/merchantbench_tool_client.py", workspace
            )
            with LocalEpisode(
                config.merchantbench_repo(),
                config.MERCHANTBENCH_PYTHON,
                run_dir / "server-state",
                task,
            ) as episode:
                if EE.enabled():
                    if self.substrate.name != "oc":
                        raise config.ConfigError("Clean execution is certified for OpenCode only")
                    env = EE.prepare(
                        workspace,
                        model=config.SOLVER_MODEL_OC,
                        tool_dirs=(
                            config.MERCHANTBENCH_PYTHON.parent,
                            config.BUN.parent,
                        ),
                    )
                    EE.audit(workspace, env, run_dir / "environment.json")
                    env = episode.agent_environment(env)
                else:
                    env = episode.agent_environment()
                env["HARNESS2_ECOM_FRAMEWORK"] = f"harness2-{self.substrate.name}"
                if not EE.enabled():
                    env["PATH"] = (
                        str(config.MERCHANTBENCH_PYTHON.parent) + os.pathsep + env.get("PATH", "")
                    )
                    for key in ("XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME"):
                        env[key] = str(workspace / key.lower())
                prompt = self.task_prompt(request.task_ref)
                if self.substrate.name == "oc":
                    env = self.substrate.run_env(
                        env,
                        workspace,
                        output_token_max=(
                            int(env.get("OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX", OUTPUT_TOKEN_MAX))
                            if EE.enabled()
                            else OUTPUT_TOKEN_MAX
                        ),
                    )
                    command = self.substrate.run_argv(prompt, workspace, config.SOLVER_MODEL_OC)
                else:
                    home = workspace / "codex_home"
                    home.mkdir()
                    shutil.copy(config.CODEX_HOME_DIR / "config.toml", home)
                    env = self.substrate.exec_env(env, codex_home=home)
                    env["VERTEX_PROXY_KEY"] = config.VERTEX_PROXY_KEY
                    command = self.substrate.exec_argv(prompt, workspace, config.SOLVER_MODEL_CX)
                episode.start()
                with (
                    (run_dir / "trajectory.jsonl").open("w") as out,
                    (run_dir / "executor.log").open("w") as err,
                ):
                    try:
                        run_process(
                            command,
                            cwd=workspace,
                            env=env,
                            stdout=out,
                            stderr=err,
                            timeout=max(1, deadline - time.monotonic()),
                        )
                    finally:
                        # Keep merchant-visible failure evidence for debugging too.
                        operations = workspace / "operations.jsonl"
                        if operations.is_file():
                            shutil.copy(operations, run_dir / "operations.jsonl")
                operations = workspace / "operations.jsonl"
                if not operations.is_file() or not any(
                    json.loads(line).get("kind") == "terminal"
                    for line in operations.read_text().splitlines()
                ):
                    raise RuntimeError(
                        "Executor exited before MerchantBench reported terminal; rollout is incomplete"
                    )
                record = episode.finish(timeout=max(1, deadline - time.monotonic()))
                record.update({"task_id": request.task_ref, "run_id_label": request.run_ids[0]})
                measurement = run_dir / "measurement"
                measurement.mkdir()
                (measurement / "episode.json").write_text(json.dumps(record), encoding="utf-8")
                shutil.copy(operations, run_dir / "operations.jsonl")
        return [run_dir]

    def trajectory(self, run_dir, *, obs_trunc):
        run_dir = Path(run_dir)
        text = self.substrate.read_trajectory(
            run_dir / "trajectory.jsonl", obs_trunc=obs_trunc, max_chars=8_000_000
        )
        return text + "\n\n# Merchant-visible tool history\n" + self.deliverable_text(run_dir)

    def deliverable_text(self, run_dir):
        path = Path(run_dir) / "operations.jsonl"
        return path.read_text(encoding="utf-8") if path.is_file() else "(no operating history)"

    def deliverable_files(self, run_dir):
        return {"operating_history.jsonl": self.deliverable_text(run_dir)}

    def score(self, run_dir, task_id):
        run_dir = Path(run_dir)
        if not self.artifacts_intact(run_dir):
            return None
        try:
            self.check_environment()
        except (OSError, RuntimeError, subprocess.SubprocessError):
            return None  # Never grade with an absent, unpinned or modified native evaluator.
        path = run_dir / "measurement/episode.json"
        record = json.loads(path.read_text())
        if record.get("terminal_status") != "finished" or record.get("task_id") != task_id:
            return None
        result = subprocess.run(
            [
                str(config.MERCHANTBENCH_PYTHON),
                str(RUNTIME),
                "score",
                "--repo",
                str(config.merchantbench_repo()),
                "--input",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode:
            return None
        try:
            metrics = json.loads(result.stdout)
            value = metrics["final_net_assets"]
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or metrics["n_steps"] <= 0
            ):
                return None
        except (ValueError, KeyError, TypeError):
            return None
        return {
            **metrics,
            "n_passed": value,
            "n_criteria": 1,
            "all_pass": False,
            "metric": "final_net_assets",
            "dataset": "public_synthetic",
            "master_seed": record["master_seed"],
            "horizon_steps": record["horizon_steps"],
        }
