# Frozen MerchantBench budget-policy evaluation

This compares **fixed policies**, not a successful recursive-improvement pipeline.
The historical inv1/dom1 improver sessions timed out and fell back to base. Do not
count their delivered solver rollouts as successful improvements. Keep k=1.

## Policy and accounting changes

Both arms receive the same corrected task brief: the operating horizon ends new
business and activations, not existing-order settlement. The controller waits for
`finished` after the order-drain period. Delivery/settlement after the horizon
still affects final assets, including refunds and fines. Agent-visible terminal
means exit successfully; the last observation is not the final score.

The base is `harness2/benchmarks/ecom_base/oc`. The candidate is
`examples/ecom-budget-harness/oc`, a new corrected budget policy; it is **not** the
byte-identical historical agg1. Existing run artifacts must remain unchanged.
The candidate's editable helper queries merchant-visible live costs/cash, records
all retained/new SKUs, planned retirements, reserve, procurement budget, expected
and stress demand/exposure, and an approved/rejected decision. It appends to the
merchant-visible operating history. This is a forecast-conditional audit, **not**
a trusted spending guard or a guarantee of zero fines. It never pre-procures stock.
Both in-flight spending already debited and uncollected receivables must not be
mistaken for new spendable cash. Delisting can stop future demand without altering
existing orders, even when those orders belong to profitable listings.

## Prepare without model calls

First follow [clean-environment setup](CLEAN_ENVIRONMENT.md), including explicit
provisioning of an approved local ripgrep executable. Then:

```bash
sh scripts/ecom-clean.sh pair prepare \
  --output runs/ecom-paired-budget-v1 --seed-start 100 \
  --seeds 10 --repeats 3 --horizon 24
sh scripts/ecom-clean.sh pair report --output runs/ecom-paired-budget-v1
```

This snapshots both harnesses and writes `tasks.json` and `plan.json`. Default
seeds 100–109 avoid seed 42 used to diagnose the policy. The plan contains 60
rollouts (10 seeds × 3 repeats × 2 arms). Twenty seeds would require 120. Native
code, task manifest, models, clean policy and tool hash are identity-checked.
Changing one requires a new experiment directory. Do not edit frozen harnesses.
The plan rejects overwriting an existing experiment. The 24-hour horizon validates
short-window behavior, **not** a surviving full-year strategy; prepare a separate
long-horizon experiment only after short-window reliability and budget checks.

No improver, selector or experience-library writes run during this evaluation.
Master seeds control simulator/catalog randomness; repeated LLM calls are not
assumed deterministic. Run IDs separate all repeats, while the schedule alternates
arm order by seed/repeat. This does not guarantee identical order draws when
policies induce different demand or random-number consumption.

## Bounded model-backed pilot, then full evaluation

The next command **calls the configured model and incurs costs**. `prepare`,
`report`, `audit` and unit tests do not. Start with only one pair:

```bash
sh scripts/ecom-clean.sh pair run --output runs/ecom-paired-budget-v1 --limit-rollouts 2
sh scripts/ecom-clean.sh pair report --output runs/ecom-paired-budget-v1
```

Inspect real skill calls, budget decisions, fulfilled plan retirements/reprices,
forecast-vs-actual cash and terminal status before spending on the remaining cells.
Then explicitly increase the per-invocation rollout count, e.g. 58 for the remaining
cells in the 60-rollout plan. This is a rollout-count cap, **not a monetary cap**;
model token limits and timeout also are not total cost limits.
The runner stops at the first failure and preserves its error. Completed artifacts
resume only if intact; failures are retried only with `--retry-failed`. Retry
history remains in the result, so a successful retry does not erase technical
failure rates. Do not repeatedly retry a failure before inspecting its evidence.

`results/*.json` contains each cell's external metrics and attributed fines.
`report.json` uses complete pairs only and reports:

- paired-arm mean net profit, profit/fine standard deviations, fines by source,
  and business survival rate;
- mean within-seed agg-minus-base differences and deterministic 2,000-resample
  seed-cluster bootstrap intervals (repeats are **not** independent bootstrap units);
- incomplete pairs, unrun cells, unresolved technical failures and all recorded
  failed attempts. One seed cannot support a multi-seed confidence interval.

Do not interpret complete-pair-only means as unbiased if technical failures are
selective; show the missingness and failure rates alongside business outcomes.
Check survival, downside and fines as well as profit. Do not expand k based on a
single profitable pair, an empty report or an interval from one seed.

## Read-only local progress page

Use the paired observatory for roots containing a frozen `plan.json`; the training
observatory expects a different run layout. Starting this page does **not** start,
resume, retry, or expand model-backed evaluation:

```bash
.venv/bin/python -m harness2.core.ecom_pair_monitor \
  --output runs/ecom-heldout-seed15-20261023-170455 --port 18768 --background

# Read-only JSON snapshot; no server or model request is started.
.venv/bin/python -m harness2.core.ecom_pair_monitor \
  --output runs/ecom-heldout-seed15-20261023-170455 --status
```

The URL/PID are saved in `paired-dashboard.json`; the listener holds
`paired-dashboard.lock`. Use `--port 0` for an automatically reserved free port.
Duplicate servers for the same root are refused. The page binds only to
`127.0.0.1`, uses the original training page's CSS and has no external assets.
It polls every two seconds and provides pair filtering, arm-level native event
logs, report/pilot logs, bounded tails, credential redaction and pause-follow.
HTTP is read-only, same-origin, allowlisted, and cannot expose the simulator DB,
`.env`, model profiles or arbitrary files. The page itself never opens a native DB.

Full-plan progress and pilot completion are different: a successful 2-attempt
pilot in a 60-cell plan is **2/60 executions and 1/30 pairs**, not a fully finished
experiment. An unstarted remainder is not a network stall. Old PIDs never prove
liveness; `pilot.lock` (or a managed continuation's `evaluation.lock`) does.
Continuation controllers may record `evaluation-status.json` and `evaluation.log`;
this page does not create such a controller or authorize paid work. The initial
pilot budget is shown separately and never treated as a global monetary cap.

Completed cells require matching attribution and intact native artifact hashes.
Missing/failed/invalid cells do not become zero-profit observations. Business
metrics come from saved result records, not fresh DB queries; run `pair report`
offline for authoritative accounting and paired statistics. No confidence interval
is inferred from one completed seed. This viewer accepts at most 2000 rollout
cells and refuses larger manifests rather than silently truncating coverage.

Viewer code and assets do not modify the frozen evaluator, analyzer, harnesses,
training controller or native transport, so adding the page does not refresh an
existing experiment's identity or permit running it with changed inputs.

## Diagnose an old or new completed episode offline

```bash
sh scripts/ecom-clean.sh pair audit /absolute/path/to/ecom/episodes/EPISODE_ID
```

The external analyzer opens the native DB read-only, filters the run and agent,
attributes positive penalty events, and requires agreement with both order-level
total penalties and measured cumulative fines. It separates supplier-delisted
stockouts from zero-quantity stockouts, insufficient balance, late-shipment fines,
refunds and bad reviews. Unrecognized event types remain explicit categories,
not silently relabeled. Refund-only lost cost is reported separately from fines.
Never install this analyzer or DB in a solver workspace or improvement context.

For the historical seed-42 agg1, the reconciliation is 88 = supplier-delisted
stockout 10 + insufficient balance 50 + bad review 20 + refund 8. Refund-only cost
loss is 38.63, not an extra fine. Assets rose from 2940 at t=24 to 3640.33 at
final t=214; net profit was 640.33. This explains that run, not policy generalization.

## Improver timeouts remain a separate diagnostic track

New OpenCode improver calls persist `<label>.<key>.meta.attempts/<attempt>/`:
`stdout.jsonl`, `stderr.log`, and `process.json` with startup/model phase, PID,
wall time, exit code and the last parsed event. Each retry gets separate files.
The entire CLI/tool process group is terminated after timeout or leader exit.
Treat raw logs as sensitive task evidence; they live inside private attempt dirs.
Startup audit/prepare failures and model-CLI timeouts must not be conflated.

Before deleting the workspace, the caller also retains native `log/*.log` files
from its private XDG data root under `native/`. It never copies host HOME, auth or
config files. Native logs are otherwise lost when that temporary root is deleted.
For an explicitly authorized diagnostic call, construct
`Proposer(spec, audit_dir=private_audit_dir, diagnostics=True)`. This opt-in mode:

- routes native DEBUG logs to the captured `stderr.log` using `--print-logs`;
- writes `transport.jsonl` with `preload_ready`, `fetch_call`, `response_headers`,
  `first_body_chunk`, and end/error/cancel boundaries, linked by process/request ID
  and monotonic elapsed milliseconds; it records origin/method/status/byte count,
  **not** URL paths/queries, credentials, headers, bodies, or reasoning text;
- retains a read-only SQLite backup of the private channel-specific `opencode*.db`
  after process-group cleanup, including committed WAL data. Each backup has a
  three-second deadline; missing artifacts and retention errors remain visible in
  `process.json`. Native files are sensitive task evidence, not sanitized exports.

The wrapper imports the unchanged native entry after installing the fetch probe.
Do not rely on CLI `--preload` with this source checkout: its `bunfig.toml` preload
can override that option under `--cwd`; the real-CLI/local-model integration test
must observe a transport trace, not merely a successful model reply.

A `fetch_call` proves only that JavaScript invoked fetch, **not** that the request
reached the gateway/provider. Headers prove an HTTP response, not a model token;
the first consumed body chunk may contain a heartbeat, reasoning or error rather
than answer text. Its timing includes consumer scheduling, so it is not a pure
network TTFT measurement. The probe wraps the consumed stream without SSE parsing
or full-body buffering; it can affect timing and remains opt-in. It instruments
this CLI's global fetch, not native SDK transports, subprocesses or wire traffic.
An absent `preload_ready` means the probe itself did not start; do not interpret it
as proof that no request occurred. Without `audit_dir`, temporary logs are deleted.

Diagnostics mode has a separate call-cache key. Changing proposer source also
changes clean experiment identity: the prepared `ecom-paired-budget-v1` plan now
belongs to the earlier identity and must not be silently refreshed or executed
under the new one. Keep it unchanged; prepare a new directory only after approval.
Do not rerun the guarded, consumed `improver-probe-1` or interpret this logging
patch as permission for another real model call or an operating rollout.

Scope is the OpenCode improver. Startup audits do not call a model and still use
captured subprocess output; this patch does not add per-command streaming there.
The MerchantBench solver already streams its trajectory/stderr, but still deletes
its private native XDG state; native solver-state retention is a separate unpatched
sibling risk. Claude CLI is outside clean-mode certification and this probe.

These changes preserve evidence; they do **not** retroactively locate the internal
step that stalled in historical inv1/dom1. A bounded paid improver reproduction
still needs a separate explicit execution budget. The frozen-policy pilot neither
requires those improvement stages nor proves the RSI pipeline is reliable.
