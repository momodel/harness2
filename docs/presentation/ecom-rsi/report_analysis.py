#!/usr/bin/env python3
"""Read-only experiment analysis; writes only report_metrics.json beside this script.

Usage accounting reuses the historical collector's pure functions, never its main().
No model calls, native DB writes, experiment reruns, or credentials are required.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True  # Importing the historical collector must not write into runs/.

from harness2.benchmarks.ecom_analysis import paired_summary  # noqa: E402

TRAIN = ROOT / "runs/ecom-train-20261023-102614-1d6e02"
EARLY = ROOT / "runs/ecom-train-20261022-135821-2bc054"
HELDOUT = ROOT / "runs/ecom-heldout-seed15-20261023-170455"
SOURCES = {}


def read(path):
    raw = path.read_bytes()
    SOURCES[path.relative_to(ROOT).as_posix()] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def hms(seconds):
    total = round(seconds)
    return f"{total // 3600:02d}:{total // 60 % 60:02d}:{total % 60:02d}"


def summarize(values):
    values = list(values)
    return {"n": len(values), "mean": st.mean(values), "median": st.median(values),
            "min": min(values), "max": max(values), "sample_stddev": st.stdev(values)}


def main(check=False):
    SOURCES.clear()
    saved_report = read(HELDOUT / "report.json")
    final_status = read(HELDOUT / "evaluation-status.json")
    plan = read(HELDOUT / "plan.json")
    require(final_status["status"] == "completed" and final_status["verification_passed"],
            "Final evaluation is not verified")
    cutoff = final_status["finished_epoch"]
    pairs = defaultdict(dict)
    for task in plan["tasks"]:
        seed = task["master_seed"]
        for repeat in range(plan["repeats"]):
            for arm in ("base", "agg"):
                path = HELDOUT / "results" / f"seed{seed}-r{repeat}-{arm}.json"
                row = read(path)
                require(row["status"] == "ok", "Do not replace an incomplete result with zero")
                native_path = (HELDOUT / "episodes" / Path(row["episode"]).name
                               / "measurement/episode.json")
                native = read(native_path)
                require(SOURCES[native_path.relative_to(ROOT).as_posix()]
                        == row["metrics"]["measurement_sha256"], "Native measurement hash changed")
                series = native["merchant_section"]["series"]
                for key, series_key in (("net_profit", "cum_net_profit"),
                                        ("final_net_assets", "net_assets")):
                    require(math.isclose(row["metrics"][key], series[series_key][-1][1],
                                         abs_tol=1e-8), "Native measurement differs from result")
                pairs[(seed, repeat)][arm] = row
    rows = [row for pair in pairs.values() for row in pair.values()]
    calculated = paired_summary(rows, planned_pairs=30)
    require(calculated == saved_report, "Paired statistics no longer match saved report")
    result_pairs = []
    by_seed = []
    by_arm = {}
    for (seed, repeat), pair in sorted(pairs.items()):
        base, agg = (pair[arm]["metrics"] for arm in ("base", "agg"))
        result_pairs.append({"seed": seed, "repeat": repeat, "base_profit": base["net_profit"],
                             "candidate_profit": agg["net_profit"],
                             "profit_delta": agg["net_profit"] - base["net_profit"],
                             "base_fine": base["cumulative_fine"],
                             "candidate_fine": agg["cumulative_fine"]})
    for seed in sorted({key[0] for key in pairs}):
        items = [r for r in result_pairs if r["seed"] == seed]
        by_seed.append({"seed": seed, "base_profit": st.mean(x["base_profit"] for x in items),
                        "candidate_profit": st.mean(x["candidate_profit"] for x in items),
                        "profit_delta": st.mean(x["profit_delta"] for x in items),
                        "candidate_wins": sum(x["profit_delta"] > 0 for x in items)})
    for arm in ("base", "agg"):
        selected = [r["metrics"] for r in rows if r["arm"] == arm]
        by_arm[arm] = {metric: summarize(r[metric] for r in selected)
                       for metric in ("net_profit", "cumulative_fine", "final_net_assets")}
        by_arm[arm]["refund_only_cost_loss"] = summarize(
            r["refund_only_cost_loss"] for r in selected)
    training = {}
    for label, root, mode in (("latest", TRAIN, "seq"), ("early", EARLY, "agg")):
        config = read(root / "train.json")
        summary = read(root / "results-summary.json")
        summarized = summary["rows"] if label == "latest" else summary["tasks"]
        index = {r["seed"]: r for r in summarized}
        training_rows = []
        for order, task in enumerate(config["tasks"], 1):
            seed = task["master_seed"]
            directory = (root / "ecom/adapt" / config["namespace"]
                         / task["id"].replace("/", "__"))
            task_summary = read(directory / "summary.json")
            current = index[seed]
            item = {"order": order, "seed": seed, "status": current["status"],
                    "base_profit": None, "candidate_profit": None, "profit_delta": None}
            if current.get("delta") is not None:
                selection = task_summary["selections"][mode]
                directories = [task_summary["base"][0]["run_dir"], selection["chosen_run_dir"]]
                profits = []
                assets = []
                for episode in directories:
                    native = read(root / "ecom/episodes" / Path(episode).name
                                  / "measurement/episode.json")
                    profits.append(native["merchant_section"]["series"]["cum_net_profit"][-1][1])
                    assets.append(native["merchant_section"]["series"]["net_assets"][-1][1])
                item.update(base_profit=profits[0], candidate_profit=profits[1],
                            profit_delta=profits[1] - profits[0],
                            base_assets=assets[0], candidate_assets=assets[1])
            training_rows.append(item)
        complete = [r for r in training_rows if r["status"] == "complete"]
        all_available = [r for r in training_rows if r["profit_delta"] is not None]
        training[label] = {"order": [t["master_seed"] for t in config["tasks"]],
                           "mode": config["mode"], "k": config["k"], "rows": training_rows,
                           "complete_pair_means": {
                               key: st.mean(r[key] for r in complete)
                               for key in ("base_profit", "candidate_profit", "profit_delta",
                                           "base_assets", "candidate_assets")},
                           "all_available_pair_means": {
                               key: st.mean(r[key] for r in all_available)
                               for key in ("base_profit", "candidate_profit", "profit_delta",
                                           "base_assets", "candidate_assets")}}
    collector_path = ROOT / "runs/usage-summary-20261026-101727/collect.py"
    SOURCES[collector_path.relative_to(ROOT).as_posix()] = hashlib.sha256(
        collector_path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location("historical_usage_collector", collector_path)
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    resource_runs = {}
    global_usage = {}
    episode_arm = {Path(row["episode"]).name: row["arm"] for row in rows}
    arm_usage = {arm: {field: 0 for field in collector.FIELDS} for arm in ("base", "agg")}
    for label, folder in collector.RUNS.items():
        root = ROOT / "runs" / folder
        episode_root = root / "episodes" if label == "frozen_validation" else root / "ecom/episodes"
        solver = collector.collect_files(episode_root.glob("*/trajectory.jsonl"), cutoff)
        improver = collector.collect_files((root / "ecom/adapt/_audit").glob("**/stdout.jsonl"), cutoff)
        timing = collector.intervals(root, cutoff)
        for filename in ("plan.json", "pilot-status.json", "evaluation-status.json",
                         "job-history.jsonl", "job-status.json"):
            state_path = root / filename
            if state_path.is_file():
                SOURCES[state_path.relative_to(ROOT).as_posix()] = hashlib.sha256(
                    state_path.read_bytes()).hexdigest()
        for component in (solver, improver):
            for file in component["files"]:
                SOURCES[file["path"]] = file["sha256"]
            require(not component["reported_total_mismatches"], "Provider usage totals mismatch")
            for event in component["events"]:
                key = (event["session_id"], event["part_id"])
                require(key not in global_usage, "Cross-file or cross-stage duplicate usage")
                global_usage[key] = event["tokens"]
        resource_runs[label] = {
            "tokens": collector.combine([solver, improver]),
            "solver_tokens": solver["tokens"], "improver_tokens": improver["tokens"],
            "solver_usage_events": solver["unique_usage_events"],
            "improver_usage_events": improver["unique_usage_events"],
            "solver_files": len(solver["files"]), "improver_files": len(improver["files"]),
            "incomplete_messages": solver["started_messages_without_finish"]
                + improver["started_messages_without_finish"],
            "malformed_lines": solver["incomplete_or_malformed_lines"]
                + improver["incomplete_or_malformed_lines"],
            "missing_reported_total": solver["missing_reported_total"]
                + improver["missing_reported_total"],
            "duplicate_events_within_component": solver["duplicate_usage_events_removed"]
                + improver["duplicate_usage_events_removed"],
            "active_window_seconds": timing["active_window_seconds"],
            "active_window_hms": hms(timing["active_window_seconds"]),
        }
        if label == "frozen_validation":
            require(len(solver["files"]) == 60, "Final frozen usage requires 60 trajectory files")
            for event in solver["events"]:
                arm = episode_arm[Path(event["file"]).parent.name]
                for field in collector.FIELDS:
                    arm_usage[arm][field] += event["tokens"][field]
    main_labels = ("latest_training", "frozen_validation")
    totals = {}
    for name, labels in (("latest_plus_frozen", main_labels), ("expanded", tuple(resource_runs))):
        combined = collector.combine([resource_runs[label] for label in labels])
        seconds = sum(resource_runs[label]["active_window_seconds"] for label in labels)
        totals[name] = {"tokens": combined, "active_window_seconds": seconds,
                        "active_window_hms": hms(seconds)}
    main_tokens = totals["latest_plus_frozen"]["tokens"]
    for source in (ROOT / "harness2/benchmarks/ecom_analysis.py", Path(collector.M.__file__),
                   Path(__file__), HERE / "build_figures.py", HERE / "evidence.json"):
        SOURCES[source.relative_to(ROOT).as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    result = {
        "measurement_cutoff": datetime.fromtimestamp(
            cutoff, timezone(timedelta(hours=8))).isoformat(),
        "new_experiment_model_calls": 0,
        "paired_report": saved_report, "seed_means": by_seed, "pairs": result_pairs,
        "pair_delta_distribution": summarize(r["profit_delta"] for r in result_pairs),
        "arm_distributions": by_arm,
        "training": training,
        "resources": {"runs": resource_runs, "totals": totals, "frozen_by_arm": arm_usage,
                      "main_cache_read_fraction": main_tokens["cache_read"] / main_tokens["total"],
                      "training_improver_token_fraction": (
                          resource_runs["latest_training"]["improver_tokens"]["total"]
                          / resource_runs["latest_training"]["tokens"]["total"]),
                      "global_unique_usage_events": len(global_usage),
                      "cross_component_duplicates": 0,
                      "cost_is_vendor_bill": False},
        "input_sha256": dict(sorted(SOURCES.items())),
    }
    # Never export profile, gateway, credentials, raw reasoning, or absolute host paths.
    content = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    require("baseURL" not in content and "/Users/" not in content and "API_KEY" not in content,
            "Private runtime configuration leaked into report metrics")
    output = HERE / "report_metrics.json"
    if check:
        require(output.read_text() == content, "Report metrics changed; inspect before updating prose")
        prose = (HERE / "EXPERIMENT_REPORT.md").read_text()
        for row in by_seed:
            expected = (f"| {row['seed']} | {row['base_profit']:.2f} | "
                        f"{row['candidate_profit']:.2f} | {row['profit_delta']:+.2f} | "
                        f"{row['candidate_wins']}/3 |")
            require(expected in prose, f"Report seed table differs: {row['seed']}")
        for row in training["latest"]["rows"]:
            values = ["缺失" if row[key] is None else f"{row[key]:.2f}"
                      for key in ("base_profit", "candidate_profit")]
            delta = "缺失" if row["profit_delta"] is None else f"{row['profit_delta']:+.2f}"
            expected = f"| {row['order']} | {row['seed']} | {' | '.join(values)} | {delta} |"
            require(expected in prose, f"Report training table differs: {row['seed']}")
        for row in result_pairs:
            expected = (f"| {row['seed']} | {row['repeat']} | {row['base_profit']:.2f} | "
                        f"{row['candidate_profit']:.2f} | {row['profit_delta']:+.2f} |")
            require(expected in prose, "Report repeat appendix differs")
        print("PASS: report_metrics.json reproduced exactly; all 50 main data-table rows verified; "
              "no report or experiment files written")
    else:
        output.write_text(content)
        print(f"Wrote {output.relative_to(ROOT)}")
    print(json.dumps({"cutoff": result["measurement_cutoff"], "resources": result["resources"],
                      "seed_means": by_seed, "training": training,
                      "distributions": result["arm_distributions"],
                      "paired_delta": result["pair_delta_distribution"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Recompute and verify; no output writes")
    main(parser.parse_args().check)
