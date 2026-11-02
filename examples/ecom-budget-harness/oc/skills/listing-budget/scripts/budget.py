"""Merchant-visible, forecast-conditional budget audit; not a simulator control."""
import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return value


def evaluate(plan, cash, listings, products):
    balance = number(cash["balance"], "balance")
    requested = number(plan.get("reserve", 0), "reserve")
    window = number(plan["window_hours"], "window_hours")
    if not 0 < window <= 12 or not str(plan.get("rationale", "")).strip():
        raise ValueError("Require a rationale and window_hours in (0,12]")
    reserve = min(balance, max(200, balance * 0.25, requested))
    budget = max(0, balance - reserve)
    active = {row["product_id"] for row in listings}
    retire = set(plan.get("retire", []))
    if len(retire) != len(plan.get("retire", [])) or not retire <= active:
        raise ValueError("retire must contain unique currently active SKU IDs")
    items, errors, seen = [], [], set()
    for item in plan["items"]:
        pid = item["product_id"]
        if pid in seen or pid in retire:
            raise ValueError("Duplicate or simultaneously retired SKU")
        seen.add(pid)
        product = products[pid]
        cost = number(product["price"], "supplier_price")
        price = number(item["sale_price"], "sale_price")
        expected = number(item["expected_units"], "expected_units")
        stress = number(item["stress_units"], "stress_units")
        if stress < expected or stress <= 0:
            raise ValueError("stress_units must be positive and >= expected_units")
        if not product.get("supplier_available", False) or product.get("quantity", 0) <= 0:
            errors.append(f"{pid}: supplier unavailable or quantity zero")
        if price <= cost:
            errors.append(f"{pid}: nonpositive unit margin")
        exposure = cost * stress
        if exposure > budget * 0.25 + 1e-8:
            errors.append(f"{pid}: concentration exceeds 25% of procurement budget")
        items.append({**item, "supplier_price": cost, "unit_margin": price - cost,
                      "expected_occupancy": cost * expected, "stress_occupancy": exposure})
    if not active - retire <= seen:
        raise ValueError("Budget omits active SKUs; include them or explicitly plan their retirement")
    if not active and len(items) > 10:
        errors.append("First tranche exceeds 10 SKUs")
    expected = sum(item["expected_occupancy"] for item in items)
    stress = sum(item["stress_occupancy"] for item in items)
    if stress > budget + 1e-8:
        errors.append("Stress procurement exceeds cash minus reserve")
    return {"version": 1, "available_cash": balance, "reserve": reserve,
            "procurement_budget": budget, "sku_count": len(items),
            "new_sku_count": len(seen - active), "retire": sorted(retire),
            "window_hours": window, "rationale": plan["rationale"], "items": items,
            "expected_occupancy": expected, "stress_occupancy": stress,
            "expected_remaining_cash": balance - expected, "stress_remaining_cash": balance - stress,
            "approved": not errors, "errors": errors}


def query(name, arguments=None):
    result = subprocess.run(
        [sys.executable, "merchantbench.py", "act", name, json.dumps(arguments or {})],
        capture_output=True, text=True, check=True, timeout=30,
    )
    payload = json.loads(result.stdout.splitlines()[-1])
    results = payload.get("tool_results", [])
    if not payload.get("ok") or not results:
        raise ValueError(f"{name} failed: {payload}")
    data = json.loads(results[0]["content"])
    if not isinstance(data, dict) or data.get("ok") is False or "error" in data:
        raise ValueError(f"{name} failed: {data}")
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    try:
        cash = query("query_balance")
        table = query("query_my_listings")
        listings = [dict(zip(table["columns"], row)) for row in table["rows"]]
        products = {item["product_id"]: query("get_product_detail", {"product_id": item["product_id"]})
                    for item in plan["items"]}
        record = evaluate(plan, cash, listings, products)
    except (ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        record = {"version": 1, "approved": False, "errors": [str(exc)], "plan": plan}
    step = Path(".merchantbench_step.json")
    record["observed_step"] = json.loads(step.read_text()).get("step") if step.exists() else None
    record["recorded_at"] = datetime.now(timezone.utc).isoformat()
    with Path("budget_decisions.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    with Path("operations.jsonl").open("a") as stream:
        stream.write(json.dumps({"kind": "budget_decision", "payload": record}) + "\n")
    print(json.dumps(record, indent=2))
    return 0 if record["approved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
