---
name: listing-budget
description: Before listing, repricing, supplier-change handling or ending an activation, record and check next-window procurement exposure across the entire retained shelf, not just one unit per SKU.
---

# Whole-shelf listing budget

## Settlement and liquidity
The operating horizon ends new orders and agent activations, not settlement.
Existing orders drain automatically; later deliveries, settlement, refunds and fines
count toward the final score. Exit on `observe terminal=true`. Never infer the
final score from the last observed cash. Do not subtract fines twice.
`balance` is procurement cash; deposit is locked. In-transit purchases have already
been debited: do not subtract them again from balance. Later receivables are not
spendable now and may not settle in full.

## Before a batch or end_of_step
1. Observe and inspect `query_platform_rules`, `query_my_listings`, open orders,
   and supplier anomalies. Screen supplier availability, quantity, current cost,
   margin and reliability. Delisting/repricing affects future orders only.
2. Estimate expected **and stress-case** order counts per retained/new SKU until
   the next activation (or the operating window end). Explain assumptions using
   observed order velocity or explicitly mark a cold-start conservative assumption.
   One unit per SKU is not a demand forecast. Use at most 10 new SKUs in the first
   tranche; fewer is acceptable. No fixed markup or margin floor guarantees profit.
3. Write `budget-plan.json` containing every retained/new SKU and any `retire` IDs:

```json
{
  "window_hours": 12,
  "reserve": 500,
  "rationale": "Cold start: conservative stress demand; no proceeds assumed before next activation",
  "retire": [],
  "items": [
    {"product_id": "P00127", "sale_price": 31.39, "expected_units": 2, "stress_units": 5}
  ]
}
```

4. Run `python .opencode/skills/listing-budget/scripts/budget.py budget-plan.json`.
   The helper queries live merchant-visible cash, listings and product details,
   checks shelf coverage and current costs, computes reserve/budget/exposure, and
   appends its decision to `operations.jsonl` and `budget_decisions.jsonl` even when
   rejected. It exits nonzero when the plan fails. A passing record is conditional
   on forecast assumptions, not an enforced simulator cash cap.
5. Do not add exposure or end the activation on a failed budget. Reduce retained
   or new listings, or reprice and revise the demand assumptions with justification;
   recompute. Then execute **all** planned retirements/listings/reprices and verify
   the resulting shelf. Never treat a proposed retirement as already executed.
   Record the checked plan before any listing batch. After mutations, rerun a plan
   matching the actual shelf before `end_of_step`.

## Checks and follow-up
The minimum reserve is max(200, 25% of current balance), capped at available cash.
Expected/stress procurement is sum(current supplier cost × forecast units), not
sum(cost per SKU). Stress exposure must fit cash minus reserve. Keep each SKU's
stress exposure at most 25% of that budget to limit concentration.
At every activation, reconcile actual cash/order changes against the preceding
forecast. If cash shrinks or demand exceeds assumptions, reduce future exposure,
including positive-margin listings with orders already in flight. Keep handling
existing orders normally. A supplier delist is a current availability condition,
not proof it can never recover; recheck visible facts before any later relisting.

## End
Exit successfully when observation is terminal; the controller drains orders.
Do not query admin state or final measurement. The external evaluator separately
attributes supplier-delisted stockouts, zero-quantity stockouts, insufficient cash,
timeouts, refunds and bad reviews. Refund-only lost procurement cost is a business
loss, not an extra fine. Do not claim that budget compliance eliminates stochastic
supplier/customer losses.
