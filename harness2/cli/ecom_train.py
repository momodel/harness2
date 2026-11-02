"""Parameterised ecom RSI training and a detached, read-only local monitor.

run/resume call paid models. --dry-run, status and serve do not. Each experiment
freezes its task manifest/config and uses the existing clean launcher unchanged.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

from ..core import ecom_monitor as M

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts/ecom-clean.sh"
DEFAULT_MODEL = "ecom_deepseek/DeepSeek-V4.1-Flash"


def now():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    return {
        name: digest(ROOT / name)
        for name in (
            "harness2/cli/ecom_train.py",
            "harness2/core/ecom_monitor.py",
            "harness2/core/execution_env.py",
            "harness2/core/selection.py",
            "harness2/core/schedule.py",
            "harness2/benchmarks/merchantbench_tool.py",
            "scripts/ecom-clean.sh",
        )
    }


def positive(raw):
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def nonnegative(raw):
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def port_number(raw):
    value = int(raw)
    if not 0 <= value <= 65535:
        raise argparse.ArgumentTypeError("port must be 0–65535 (0 selects a free port)")
    return value


def seed_list(raw):
    try:
        values = [nonnegative(value) for value in raw.split(",")]
    except (ValueError, argparse.ArgumentTypeError):
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated nonnegative integers"
        ) from None
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("seeds must be nonempty and unique")
    return values


def build_plan(args):
    from ..core.schedule import stream_order

    seeds = (
        args.seeds
        if args.seeds is not None
        else range(args.seed_start, args.seed_start + args.tasks)
    )
    if args.seeds is not None and args.seed_start != 0:
        raise ValueError("--seed-start cannot be combined with --seeds")
    if len(seeds) > 1000:
        raise ValueError("At most 1000 tasks per invocation; split larger experiments explicitly")
    tasks = [
        {
            "id": f"synthetic/train{args.hours}h-seed{s}",
            "domain": "synthetic",
            "master_seed": s,
            "horizon_steps": args.hours,
        }
        for s in seeds
    ]
    order = stream_order({"synthetic": [t["id"] for t in tasks]}, args.order_seed, ["synthetic"])
    by_id = {t["id"]: t for t in tasks}
    profile = args.model_config.expanduser().resolve()
    if not profile.is_file():
        raise ValueError(f"Model profile not found: {profile}")
    model = args.model or os.environ.get("HARNESS2_SOLVER_MODEL") or DEFAULT_MODEL
    improver = args.improver_model or os.environ.get("HARNESS2_IMPROVER_MODEL") or model
    return {
        "version": 1,
        "created_at": now(),
        "tasks": [by_id[t] for _, t in order],
        "hours": args.hours,
        "k": args.k,
        "mode": args.mode,
        "order_seed": args.order_seed,
        "namespace": f"ecom-oc-agg{args.k}-s{args.order_seed}",
        "solver_model": model,
        "improver_model": improver,
        "timeout": args.timeout,
        "model_config": str(profile),
        "profile_sha256": digest(profile),
        "sources": sources(),
    }


def prepare(args):
    plan = build_plan(args)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "planned_rollouts": len(plan["tasks"]) * 4 * plan["k"],
                    "plan": plan,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        print("Preview only: no files, monitor, model calls or experience writes.")
        return None
    root = (
        args.output.expanduser().resolve()
        if args.output
        else (
            ROOT / "runs" / f"ecom-train-{dt.datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        )
    )
    root.mkdir(parents=True, exist_ok=False)
    M.atomic_json(root / "tasks.json", {"version": 1, "tasks": plan["tasks"]})
    plan["manifest_sha256"] = digest(root / "tasks.json")
    M.atomic_json(root / "train.json", plan)
    M.atomic_json(root / "job-status.json", {"status": "prepared", "stage": "prepare"})
    return root


def validate(root):
    root = Path(root).resolve()
    plan = M.read_json(root / "train.json")
    if not plan or plan.get("version") != 1:
        raise ValueError("Resume requires a managed training root containing train.json")
    if plan["sources"] != sources():
        raise ValueError("Runner/environment source changed; use a new experiment directory")
    if digest(root / "tasks.json") != plan["manifest_sha256"]:
        raise ValueError("Task manifest changed; hours/seeds cannot change during resume")
    if digest(plan["model_config"]) != plan["profile_sha256"]:
        raise ValueError("Model profile changed; use a new experiment directory")
    return plan


def environment(root, plan):
    return {
        **os.environ,
        "HARNESS2_RUNS_DIR": str(root),
        "HARNESS2_ECOM_TASKS_FILE": str(root / "tasks.json"),
        "HARNESS2_SOLVER_MODEL": plan["solver_model"],
        "HARNESS2_IMPROVER_MODEL": plan["improver_model"],
        "HARNESS2_CLEAN_MODEL_CONFIG": plan["model_config"],
        "HARNESS2_ECOM_TIMEOUT": str(plan["timeout"]),
        "PYTHONUNBUFFERED": "1",
    }


def acquire(root, name):
    handle = (Path(root) / name).open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise ValueError(f"Already running ({name}); refusing a duplicate process") from None
    return handle


def _module_command(*args):
    return [sys.executable, "-u", "-m", "harness2.cli.ecom_train", *map(str, args)]


def start_dashboard(root, port, namespace=None):
    if M.locked(root, "dashboard.lock"):
        info = M.read_json(Path(root) / "server.json", {})
        if not info.get("url"):
            raise ValueError("Dashboard is starting; check server.json shortly")
        return info
    with acquire(root, "dashboard.lock") as lock:
        server = M.make_server(root, port, namespace)
        try:
            argv = _module_command(
                "serve",
                "--output",
                root,
                "--socket-fd",
                server.fileno(),
                "--lock-fd",
                lock.fileno(),
            )
            if namespace:
                argv += ["--namespace", namespace]
            with (Path(root) / "dashboard.log").open("ab") as log:
                proc = subprocess.Popen(
                    argv,
                    cwd=ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    pass_fds=(server.fileno(), lock.fileno()),
                )
            info = {
                "pid": proc.pid,
                "host": "127.0.0.1",
                "port": server.server_address[1],
                "url": f"http://127.0.0.1:{server.server_address[1]}",
                "started_at": now(),
            }
            M.atomic_json(Path(root) / "server.json", info)
            return info
        finally:
            server.server_close()


def start_worker(root):
    plan = validate(root)
    with acquire(root, "worker.lock") as lock:
        # Preflight cannot authenticate the endpoint, but catches local setup and missing keys.
        checked = subprocess.run(
            ["sh", str(LAUNCHER), "check"],
            cwd=ROOT,
            env=environment(root, plan),
            capture_output=True,
            text=True,
        )
        if checked.returncode:
            raise ValueError(checked.stderr.strip() or checked.stdout.strip())
        with (root / "worker.log").open("ab") as log:
            proc = subprocess.Popen(
                _module_command("_worker", "--output", root, "--lock-fd", lock.fileno()),
                cwd=ROOT,
                env=environment(root, plan),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(lock.fileno(),),
            )
        (root / "job.pid").write_text(str(proc.pid) + "\n")
        return proc.pid


def terminate_tree(pid):
    """OpenCode/tools start separate process groups; terminate descendants, not just the sweep."""
    text = subprocess.check_output(["/bin/ps", "-Ao", "pid=,ppid=,pgid="], text=True)
    rows = {
        int(a): (int(b), int(c))
        for line in text.splitlines()
        if len(parts := line.split()) == 3
        for a, b, c in [parts]
    }
    ids = {pid}
    while True:
        expanded = ids | {p for p, (parent, _) in rows.items() if parent in ids}
        if expanded == ids:
            break
        ids = expanded
    groups = {rows[p][1] for p in ids if p in rows} - {os.getpgrp()}
    # Stop the scheduler before descendants, so it cannot create replacement work.
    for p in sorted(ids, key=lambda p: p != pid):
        try:
            os.kill(p, signal.SIGSTOP)
        except ProcessLookupError:
            pass
    for group in groups:
        try:
            os.killpg(group, signal.SIGTERM)
            os.killpg(group, signal.SIGCONT)
        except ProcessLookupError:
            pass
    # A blocked shell/native child must not survive cancellation. These are only
    # groups captured as descendants of this sweep, never global name matches.
    for group in groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass


def execute(argv, env, log, on_line):
    proc = subprocess.Popen(
        argv,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        for line in proc.stdout:
            log.write(line)
            log.flush()
            on_line(line)
        return proc.wait()
    except BaseException:
        terminate_tree(proc.pid)
        proc.wait()
        raise
    finally:
        proc.stdout.close()


def verify(root, plan):
    view = M.snapshot(root)
    ns = root / "ecom/adapt" / plan["namespace"]
    selected = M.read_json(ns / "judged.json", [])
    bases = M.read_json(ns / "base_judged.json", [])
    intact = True
    run_ns = (
        plan["namespace"].replace("-agg", "-seq", 1)
        if plan["mode"] == "sequential"
        else plan["namespace"]
    )
    for task in plan["tasks"]:
        for i in range(1, plan["k"] + 1):
            for label in (
                f"base{i}",
                f"inv{i}",
                f"dom{i}",
                f"agg{i}" if plan["mode"] == "parallel" else f"c{i}",
            ):
                space = plan["namespace"] if label.startswith("base") else run_ns
                episode = M.episode_dir(root, space, task["id"], label)
                marker = M.read_json(episode / ".adapt_complete", {})
                paths = [
                    episode / p
                    for p in ("measurement/episode.json", "trajectory.jsonl", "operations.jsonl")
                ]
                if (episode / "environment.json").is_file():
                    paths.append(episode / "environment.json")
                try:
                    fingerprint = hashlib.sha256()
                    for index, path in enumerate(paths):
                        if index:
                            fingerprint.update(b"\0")
                        with path.open("rb") as stream:
                            while chunk := stream.read(1024 * 1024):
                                fingerprint.update(chunk)
                    intact = (
                        intact and marker.get("artifact_fingerprint") == fingerprint.hexdigest()
                    )
                except OSError:
                    intact = False
    expected_ids = {t["id"] for t in plan["tasks"]}
    expected_bases = Counter(
        (tid, f"base{i}") for tid in expected_ids for i in range(1, plan["k"] + 1)
    )
    checks = {
        "all_rollout_artifacts_intact": intact,
        "all_reflections_complete": all(
            t["reflection_status"] in ("written", "no_entry") for t in view["tasks"]
        ),
        "all_tasks_complete": len(view["tasks"]) == len(plan["tasks"])
        and all(t["status"] in ("complete", "warning") for t in view["tasks"]),
        "no_fallbacks": view["counts"]["fallbacks"] == 0,
        "all_rollouts_complete": view["counts"]["rollouts"] == view["counts"]["planned_rollouts"],
        "all_selected_graded": len(selected) == len(expected_ids)
        and {r.get("task_id") for r in selected} == expected_ids
        and all(r.get("graded") for r in selected),
        "all_bases_graded": Counter((r.get("task"), r.get("label")) for r in bases)
        == expected_bases
        and all(r.get("score") is not None for r in bases),
        "report_present": (root / "report.txt").is_file()
        and (root / "report.txt").stat().st_size > 0,
    }
    result = {"passed": all(checks.values()), "checks": checks, "verified_at": now()}
    M.atomic_json(root / "verification.json", result)
    return result


def run_worker(root, lock_fd=None, executor=execute):
    root = Path(root).resolve()
    handle = os.fdopen(lock_fd, "a+") if lock_fd is not None else acquire(root, "worker.lock")
    with handle:
        plan = validate(root)
        previous = M.read_json(root / "job-status.json", {})
        if previous.get("started_at"):
            with (root / "job-history.jsonl").open("a") as history:
                history.write(json.dumps(previous, ensure_ascii=False) + "\n")
        state = {
            "status": "running",
            "pid": os.getpid(),
            "started_at": now(),
            "started_epoch": time.time(),
            "stage": "check",
            "stages": [],
            "task_id": plan["tasks"][0]["id"],
            "task_started_at": time.time(),
        }

        def save():
            M.atomic_json(root / "job-status.json", state)

        def on_line(line):
            match = re.search(r"\[synthetic\]\s+(\d+)/(\d+)\s+\S+\s+(synthetic/\S+)", line)
            if match:
                index = int(match[1])
                state["task_id"] = (
                    plan["tasks"][index]["id"] if index < len(plan["tasks"]) else None
                )
                state["task_started_at"] = time.time()
                save()

        def interrupted(*_):
            raise KeyboardInterrupt("Operator stopped training")

        old_signal = signal.signal(signal.SIGTERM, interrupted)
        save()
        flags = ["--k", str(plan["k"]), "--mode", plan["mode"], "--seed", str(plan["order_seed"])]
        try:
            with (root / "run.log").open("a", buffering=1) as log:
                for label, command, extra in (
                    ("check", "check", []),
                    (
                        "run",
                        "run",
                        [
                            *flags,
                            "--split",
                            "all",
                            "--workers",
                            "1",
                            "--max-tasks",
                            str(len(plan["tasks"])),
                        ],
                    ),
                    ("judge_selected", "judge", flags),
                    ("judge_base", "judge", [*flags, "--bases"]),
                    ("report", "report", flags),
                ):
                    state["stage"] = label
                    save()
                    log.write(f"\n=== {label} {now()} ===\n")
                    before = log.tell()
                    code = executor(
                        ["sh", str(LAUNCHER), command, *extra],
                        environment(root, plan),
                        log,
                        on_line,
                    )
                    state["stages"].append({"stage": label, "returncode": code})
                    if command == "report":
                        with (root / "run.log").open() as stream:
                            stream.seek(before)
                            (root / "report.txt").write_text(stream.read())
                    save()
                    if code:
                        raise RuntimeError(
                            f"{label} exited with code {code}; inspect run.log before resume"
                        )
                state["stage"] = "verify"
                save()
                result = verify(root, plan)
                state["status"] = "completed" if result["passed"] else "completed_with_warnings"
                state["checks"] = result["checks"]
        except KeyboardInterrupt as exc:
            state.update(status="interrupted", error=str(exc))
        except Exception as exc:
            state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            state.update(finished_at=now(), finished_epoch=time.time())
            save()
            signal.signal(signal.SIGTERM, old_signal)
        return 0 if state["status"] == "completed" else 2


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser(
        "run", help="PAID: prepare a fresh experiment, then launch detached training"
    )
    task_selection = run.add_mutually_exclusive_group()
    task_selection.add_argument("--tasks", type=positive, default=10)
    task_selection.add_argument(
        "--seeds", type=seed_list, help="explicit unique seeds, e.g. 0,7; no intermediate tasks"
    )
    run.add_argument("--seed-start", type=nonnegative, default=0)
    run.add_argument("--hours", "--horizon-steps", type=positive, default=24)
    run.add_argument("--k", type=positive, default=1)
    run.add_argument("--mode", choices=("parallel", "sequential"), default="parallel")
    run.add_argument("--order-seed", type=nonnegative, default=0)
    run.add_argument("--model", "--solver-model", dest="model")
    run.add_argument("--improver-model")
    run.add_argument(
        "--model-config",
        type=Path,
        default=Path(
            os.environ.get("HARNESS2_CLEAN_MODEL_CONFIG", ROOT / "config/ecom-models.json")
        ),
    )
    run.add_argument(
        "--timeout",
        type=positive,
        default=1800,
        help="wall seconds per business rollout, not simulation hours",
    )
    run.add_argument("--output", type=Path, help="fresh directory; must not exist")
    run.add_argument("--dry-run", action="store_true", help="preview only, no files or paid calls")
    resume = sub.add_parser("resume", help="PAID: resume matching artifacts, refuse changed inputs")
    resume.add_argument("--output", type=Path, required=True)
    for target in (run, resume):
        target.add_argument("--port", type=port_number, default=8765)
        target.add_argument("--no-dashboard", action="store_true")
    serve = sub.add_parser("serve", help="read-only: view a managed or historical run")
    serve.add_argument("--output", type=Path, required=True)
    serve.add_argument("--namespace")
    serve.add_argument("--port", type=port_number, default=8765)
    serve.add_argument("--background", action="store_true")
    serve.add_argument("--socket-fd", type=int, help=argparse.SUPPRESS)
    serve.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    status = sub.add_parser("status", help="read-only: print a JSON snapshot")
    status.add_argument("--output", type=Path, required=True)
    status.add_argument("--namespace")
    stop = sub.add_parser(
        "stop", help="stop this managed worker and its subprocesses, keep the dashboard"
    )
    stop.add_argument("--output", type=Path, required=True)
    worker = sub.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--lock-fd", type=int)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command in ("run", "resume"):
            root = prepare(args) if args.command == "run" else args.output.expanduser().resolve()
            if root is None:
                return 0
            validate(root)
            if M.locked(root):
                raise ValueError(
                    "Training already running; use status/serve, not a duplicate resume"
                )
            # Reserve the listener before any paid work. A port conflict must not
            # accidentally start an invisible training job.
            info = None if args.no_dashboard else start_dashboard(root, args.port)
            try:
                pid = start_worker(root)
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                if not M.locked(root):
                    M.atomic_json(
                        root / "job-status.json",
                        {"status": "failed", "stage": "check", "error": str(exc)},
                    )
                if info:
                    print(f"Dashboard: {info['url']}")
                raise
            print(f"Training launched: {root}\nWorker PID: {pid}")
            if info:
                print(f"Dashboard: {info['url']}")
            print(
                f"Log: {root / 'run.log'}\nThis run calls models. Closing this terminal does not stop it."
            )
            return 0
        root = args.output.expanduser().resolve()
        if args.command == "_worker":
            return run_worker(root, args.lock_fd)
        if args.command == "status":
            print(json.dumps(M.snapshot(root, args.namespace), ensure_ascii=False, indent=2))
            return 0
        if args.command == "stop":
            if not M.locked(root):
                raise ValueError("No active managed worker")
            state = M.read_json(root / "job-status.json", {})
            pid = state.get("pid") or int((root / "job.pid").read_text())
            cmd = subprocess.check_output(["/bin/ps", "-p", str(pid), "-o", "command="], text=True)
            if "harness2.cli.ecom_train _worker" not in cmd or str(root) not in cmd:
                raise ValueError(
                    "Worker identity does not match; refusing to signal a different process"
                )
            os.kill(pid, signal.SIGTERM)
            print("Stop requested. Dashboard remains available; verify interrupted before resume.")
            return 0
        if args.background:
            print(start_dashboard(root, args.port, args.namespace)["url"])
            return 0
        handle = (
            os.fdopen(args.lock_fd, "a+")
            if args.lock_fd is not None
            else acquire(root, "dashboard.lock")
        )
        with handle:
            server = M.make_server(root, args.port, args.namespace, args.socket_fd)
            print(f"Read-only dashboard: http://127.0.0.1:{server.server_address[1]}", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
        return 0
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
