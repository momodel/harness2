"""External, read-only evaluation evidence. Never installed in solver workspaces."""
from __future__ import annotations

import hashlib
import json
import math
import random
import sqlite3
import statistics as st
from collections import Counter
from pathlib import Path
from urllib.parse import quote


def episode_audit(directory: Path):
    directory = Path(directory)
    measurement = directory / "measurement/episode.json"
    record = json.loads(measurement.read_text())
    if record["terminal_status"] != "finished":
        raise ValueError("Cannot audit an unfinished episode as a completed score")
    merchant = record["merchant_section"]
    series = merchant["series"]
    agent_id = merchant["agent_id"]
    dbs = list((directory / "server-state/runs").glob("*/state.db"))
    if len(dbs) != 1:
        raise ValueError("Expected exactly one native state DB")
    fines, counts, events = Counter(), Counter(), []
    connection = sqlite3.connect("file:" + quote(str(dbs[0].resolve())) + "?mode=ro", uri=True)
    try:
        for t, event_type, entity, payload in connection.execute(
            "SELECT t,event_type,entity_id,payload FROM events WHERE run_id=? AND agent_id=? ORDER BY t,rowid",
            (record["run_id"], agent_id),
        ):
            payload = json.loads(payload)
            penalty = float(payload.get("penalty", 0))
            if not math.isfinite(penalty) or penalty < 0:
                raise ValueError("Invalid penalty event")
            if not penalty:
                continue
            if event_type == "order_stockout_violation":
                category = {"supplier_delisted": "supplier_delisted", "out_of_stock": "inventory_stockout"}.get(
                    payload.get("reason"), "unknown_stockout")
            else:
                category = {"order_insufficient_balance_violation": "insufficient_balance",
                            "order_settled_refund": "refund", "order_settled_bad_review": "bad_review",
                            "order_timeout": "timeout", "order_late": "timeout"}.get(event_type, event_type)
            fines[category] += penalty
            counts[category] += 1
            events.append({"t": t, "event_type": event_type, "category": category,
                           "entity_id": entity, "order_id": payload.get("order_id", entity),
                           "penalty": penalty, "phase": "operating" if t < record["horizon_steps"] else "drain"})
        order_fines = connection.execute(
            "SELECT COALESCE(SUM(total_penalty),0) FROM orders WHERE run_id=? AND agent_id=?",
            (record["run_id"], agent_id),
        ).fetchone()[0]
        only_refund_loss = connection.execute(
            "SELECT COALESCE(SUM(realized_cost-realized_revenue),0) FROM orders "
            "WHERE run_id=? AND agent_id=? AND current_status='settled_only_refund'",
            (record["run_id"], agent_id),
        ).fetchone()[0]
    finally:
        connection.close()
    total = sum(fines.values())
    measured = series["cum_fine"][-1][1]
    if not all(math.isclose(total, value, abs_tol=1e-6) for value in (order_fines, measured)):
        raise ValueError(f"Fine reconciliation failed: events={total}, orders={order_fines}, measurement={measured}")
    return {"version": 1, "episode": str(directory.resolve()), "master_seed": record["master_seed"],
            "horizon_steps": record["horizon_steps"], "final_t": series["net_assets"][-1][0],
            "final_net_assets": series["net_assets"][-1][1], "net_profit": series["cum_net_profit"][-1][1],
            "is_alive": merchant["is_alive"], "cumulative_fine": total,
            "fine_breakdown": dict(fines), "fine_counts": dict(counts), "fine_events": events,
            "refund_only_cost_loss": only_refund_loss, "reconciled": True,
            "measurement_sha256": hashlib.sha256(measurement.read_bytes()).hexdigest()}


def paired_summary(cells, *, planned_pairs):
    """Use complete pairs only; bootstrap seed means, not correlated repeats."""
    grouped = {}
    for cell in cells:
        key = (cell["master_seed"], cell["repeat"])
        arm = cell["arm"]
        if arm not in ("base", "agg") or arm in grouped.setdefault(key, {}):
            raise ValueError("Invalid arm or duplicate paired cell")
        grouped[key][arm] = cell
    complete = [pair for pair in grouped.values()
                if all(pair.get(arm, {}).get("status") == "ok" for arm in ("base", "agg"))]
    attempts = {arm: [attempt for cell in cells if cell["arm"] == arm
                      for attempt in cell.get("attempt_history", [{"status": cell["status"]}])]
                for arm in ("base", "agg")}
    result = {"planned_pairs": planned_pairs, "complete_pairs": len(complete),
              "incomplete_pairs": planned_pairs - len(complete),
              "technical_failures": {arm: sum(c["arm"] == arm and c["status"] == "failed" for c in cells)
                                     for arm in ("base", "agg")},
              "not_run_cells": planned_pairs * 2 - len(cells),
              "failed_attempts": {arm: sum(a["status"] == "failed" for a in attempts[arm])
                                  for arm in ("base", "agg")},
              "technical_failure_rate": {
                  arm: st.mean(a["status"] == "failed" for a in attempts[arm]) if attempts[arm] else None
                  for arm in ("base", "agg")}, "arms": {}, "paired": {}}
    for arm in ("base", "agg"):
        rows = [pair[arm]["metrics"] for pair in complete]
        result["arms"][arm] = {"n": len(rows)}
        if rows:
            profits = [r["net_profit"] for r in rows]
            result["arms"][arm].update({"mean_net_profit": st.mean(profits),
                "profit_stddev": st.stdev(profits) if len(rows) > 1 else None,
                "mean_fine": st.mean(r["cumulative_fine"] for r in rows),
                "fine_stddev": st.stdev(r["cumulative_fine"] for r in rows) if len(rows) > 1 else None,
                "survival_rate": st.mean(bool(r["is_alive"]) for r in rows),
                "mean_fine_by_source": {category: st.mean(r["fine_breakdown"].get(category, 0) for r in rows)
                                        for category in sorted({k for r in rows for k in r["fine_breakdown"]})}})
    if not complete:
        return result
    for metric in ("net_profit", "cumulative_fine", "is_alive"):
        seed_diffs = {}
        for pair in complete:
            delta = float(pair["agg"]["metrics"][metric]) - float(pair["base"]["metrics"][metric])
            seed_diffs.setdefault(pair["base"]["master_seed"], []).append(delta)
        means = [st.mean(values) for values in seed_diffs.values()]
        ci = None
        if len(means) >= 2:
            rng = random.Random(0)
            samples = sorted(st.mean(rng.choices(means, k=len(means))) for _ in range(2000))
            ci = [samples[49], samples[1949]]
        result["paired"][metric] = {"mean_seed_delta_agg_minus_base": st.mean(means),
                                   "seed_count": len(means), "seed_bootstrap_95ci": ci}
    result["warning"] = "Conditional on complete pairs; report technical failures separately. Seeds, not repeats, are bootstrap units."
    return result
