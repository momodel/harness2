"""Frozen-harness paired evaluation; prepare/report are offline, run calls models.

No recursive refinement, selector, or library writes occur during this experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .. import config
from ..benchmarks import ecom_analysis
from ..benchmarks.ecom_analysis import episode_audit, paired_summary
from ..benchmarks.ecommerce import EcomAdapter
from ..core import runmark, tree


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare(args):
    if not 10 <= args.seeds <= 20 or args.repeats < 3:
        raise ValueError("Use 10–20 seeds and at least 3 repeats")
    if args.seed_start < 0 or args.horizon < 1:
        raise ValueError("Seed start must be nonnegative; horizon must be positive")
    root = args.output.resolve()
    if root.exists():
        raise ValueError("Refusing to overwrite an experiment directory")
    adapter = EcomAdapter()
    surface = adapter.substrate.surface()
    harnesses = {}
    for arm, source in (("base", args.base or adapter.base_harness()), ("agg", args.agg)):
        rejected = []
        content = tree.read_tree(surface, source, rejected=rejected)
        if not content or rejected:
            raise ValueError(f"Invalid {arm} harness: {rejected}")
        adapter.validate_tree(content)
        harnesses[arm] = content
    root.mkdir(parents=True)
    tasks = [{"id": f"synthetic/paired-seed{seed}", "domain": "synthetic",
              "master_seed": seed, "horizon_steps": args.horizon}
             for seed in range(args.seed_start, args.seed_start + args.seeds)]
    write_json(root / "tasks.json", {"version": 1, "tasks": tasks})
    hashes = {}
    for arm, content in harnesses.items():
        target = tree.write_tree(surface, content, root / arm, stubs=False)
        hashes[arm] = runmark.fingerprint(target)
    adapter = EcomAdapter(tasks_file=root / "tasks.json", store_root=root / "episodes")
    plan = {"version": 1, "method": "frozen-harness-paired-v1", "k": 1,
            "library_updates": False, "repeats": args.repeats, "tasks": tasks,
            "harness_fingerprints": hashes, "solver_model": config.SOLVER_MODEL_OC,
            "identity": adapter.experiment_identity(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "analysis_sha256": hashlib.sha256(Path(ecom_analysis.__file__).read_bytes()).hexdigest(),
            "planned_rollouts": len(tasks) * args.repeats * 2}
    write_json(root / "plan.json", plan)
    return plan


def validate(root):
    plan = json.loads((root / "plan.json").read_text())
    adapter = EcomAdapter(tasks_file=root / "tasks.json", store_root=root / "episodes")
    if (plan["identity"] != adapter.experiment_identity()
            or plan["solver_model"] != config.SOLVER_MODEL_OC
            or plan["runner_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            or plan["analysis_sha256"] != hashlib.sha256(Path(ecom_analysis.__file__).read_bytes()).hexdigest()):
        raise ValueError("Experiment code/environment/model changed; prepare a new experiment root")
    for arm in ("base", "agg"):
        if runmark.fingerprint(root / arm) != plan["harness_fingerprints"][arm]:
            raise ValueError(f"Frozen {arm} harness changed")
    return plan, adapter


def cells(plan):
    for task in plan["tasks"]:
        for repeat in range(plan["repeats"]):
            arms = ("base", "agg") if (task["master_seed"] + repeat) % 2 == 0 else ("agg", "base")
            for arm in arms:
                yield task, repeat, arm, f"seed{task['master_seed']}-r{repeat}-{arm}"


def run(args):
    if args.limit_rollouts < 1:
        raise ValueError("limit-rollouts must be positive")
    root = args.output.resolve()
    plan, adapter = validate(root)
    adapter.preflight()
    result_dir = root / "results"
    result_dir.mkdir(exist_ok=True)
    attempted = 0
    for task, repeat, arm, label in cells(plan):
        result_path = result_dir / f"{label}.json"
        history = []
        if result_path.exists():
            row = json.loads(result_path.read_text())
            if row["status"] == "ok":
                if not adapter.artifacts_intact(Path(row["episode"])):
                    raise ValueError(f"Changed or missing completed artifacts: {label}")
                continue
            history = row.get("attempt_history", [{"status": row["status"], "error": row.get("error")}])
            if not args.retry_failed:
                raise RuntimeError(f"{label} has an unresolved failure; inspect evidence before --retry-failed")
        if attempted >= args.limit_rollouts:
            break
        attempted += 1
        row = {"master_seed": task["master_seed"], "repeat": repeat, "arm": arm,
               "attempt_history": history}
        try:
            episode = adapter.execute(task["id"], root / arm, f"paired/{label}")
            if adapter.score(episode, task["id"]) is None:
                raise ValueError("Completed episode failed native scoring/integrity checks")
            row.update(status="ok", episode=str(episode.resolve()), metrics=episode_audit(episode))
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            history.append({"status": "failed", "error": row["error"]})
            write_json(result_path, row)
            # Stop on first failure, retain evidence; do not blindly burn the remaining budget.
            raise RuntimeError(f"{label} failed; inspect evidence before --retry-failed") from exc
        history.append({"status": "ok"})
        write_json(result_path, row)
        print(f"{label}: profit={row['metrics']['net_profit']:.2f}, fines={row['metrics']['cumulative_fine']:.2f}", flush=True)
    return {"attempted": attempted, "planned_rollouts": plan["planned_rollouts"]}


def report(root):
    root = root.resolve()
    plan, adapter = validate(root)
    rows = []
    for _, _, _, label in cells(plan):
        path = root / "results" / f"{label}.json"
        if not path.exists():
            continue
        row = json.loads(path.read_text())
        if row["status"] == "ok":
            if not adapter.artifacts_intact(Path(row["episode"])):
                raise ValueError(f"Changed or missing completed artifacts: {label}")
            # Recheck DB attribution too; never silently trust a stale report cache.
            row["metrics"] = episode_audit(Path(row["episode"]))
        rows.append(row)
    result = paired_summary(rows, planned_pairs=plan["planned_rollouts"] // 2)
    write_json(root / "report.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Offline: snapshot harnesses and generate an evaluation manifest")
    prep.add_argument("--output", required=True, type=Path)
    prep.add_argument("--base", type=Path)
    prep.add_argument("--agg", type=Path, default=config.ROOT / "examples/ecom-budget-harness/oc")
    prep.add_argument("--seeds", type=int, default=10)
    prep.add_argument("--seed-start", type=int, default=100)
    prep.add_argument("--repeats", type=int, default=3)
    prep.add_argument("--horizon", type=int, default=24)
    execute = sub.add_parser("run", help="PAID MODEL CALLS: bounded, resumable run; stop on first failure")
    execute.add_argument("--output", required=True, type=Path)
    execute.add_argument("--limit-rollouts", type=int, default=2)
    execute.add_argument("--retry-failed", action="store_true")
    summary = sub.add_parser("report", help="Offline: reconcile fines and compute seed-clustered paired statistics")
    summary.add_argument("--output", required=True, type=Path)
    audit = sub.add_parser("audit", help="Offline: reconcile one completed episode's fines")
    audit.add_argument("episode", type=Path)
    args = parser.parse_args(argv)
    value = (prepare(args) if args.command == "prepare" else run(args) if args.command == "run"
             else report(args.output) if args.command == "report" else episode_audit(args.episode))
    print(json.dumps(value, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
