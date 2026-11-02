# MerchantBench: EcomAdapter

This local extension adds **`--bench ecom`**, backed by
[KhanCold/merchantbench](https://github.com/KhanCold/merchantbench) at
`d9e5bd7519ab100193c21778aecfb59e30f24651`. It does not route tasks through
JobBench, use an LLM judge, or connect to real shops. It supports the upstream
**public synthetic** catalog (1,000 products / 200 suppliers), not the paper's
unreleased commercial dataset. MerchantBench software is Apache-2.0; keep its
upstream license and notices with the downloaded checkout.

## Current scope and safety

- Fresh authenticated upstream Flask simulator per rollout; fixed upstream SDK
  plus a small agent CLI. Editable harness components cannot replace that CLI.
- OpenCode (`oc`) and Codex (`cx`) use HARNESS²'s normal materialisation, executor,
  refinement, selection and reflection machinery. The adapter does not modify
  model weights or upstream simulator/scoring source.
- **Trusted local debugging only, not a security sandbox.** Agent and simulator
  run as the same OS user. Separating directories and admin/agent tokens prevents
  accidental API exposure, NOT a malicious shell from reading host files,
  inspecting processes, changing code or contacting the internet. Do not run
  adversarial agents, production credentials or publication-grade blinded
  evaluations with this mode. Containers/separate identities and network/file
  isolation are future work; Docker is not currently used by this adapter.
- Solver workspaces receive agent tokens only. Native measurement and server
  state live outside them. Improver evidence contains the merchant-visible tool
  history and solver trajectory, not the stored administrative measurement.
  Observed cash, inventory and commercial outcomes remain legitimate task data.
- Executors may use configured paid model APIs. No real marketplace APIs, real
  customer records, production stock changes, or monetary purchases are used.

## Install and validate without models

Use a POSIX host and Python 3.12 (also follow upstream HARNESS² prerequisites).
From this checkout:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .
harness2-setup ecom --env-only
harness2-ecom-smoke --days 1 --output runs/merchantbench-smoke.json
python -m unittest discover -s tests -v
```

The environment-only command installs the pinned upstream source and a separate
Python runtime under `dependencies/merchantbench/.venv`. It does not install
OpenCode, Codex, a model client, or model credentials. Runtime dependencies are
locked in `harness2/benchmarks/merchantbench-requirements.txt`; its `.in` matches
the upstream environment plus the SDK / rule-based agent's `requests` dependency.
The SDK is copied unmodified from that pinned checkout for each rollout.

The smoke command runs the actual upstream **rule-based agent**, simulator,
settlement and `eval.scoring.compute`. It does not run the model or optimize a
harness. It refuses to overwrite its output. Native integration tests are skipped
if the environment has not been installed; after installation, they also test
both executor launch paths with a deterministic stand-in executor, terminal
handling, authenticated API denial, scoring and resume integrity.

## Clean OpenCode environment (recommended)

For both the OpenCode executor and improver, use the new opt-in launcher:

```bash
sh scripts/ecom-clean.sh prepare-tools --rg /absolute/path/to/approved/rg
sh scripts/ecom-clean.sh verify  # real CLI pollution/skill tests, local fake model only
sh scripts/ecom-clean.sh check
sh scripts/ecom-clean.sh run --k 1 --mode parallel --max-tasks 1
sh scripts/ecom-clean.sh judge --k 1 --mode parallel --bases
sh scripts/ecom-clean.sh report --k 1 --mode parallel --bases
```

Configure model IDs and explicit credentials first; no model endpoint is chosen
for you. The launcher selects the short task and a separate clean run root.
Unlike legacy runs it does not inherit personal skills, global instructions,
MCP/provider config, or arbitrary environment variables. It preserves experiment
skills and records startup configuration evidence. See
[clean environment setup and boundaries](CLEAN_ENVIRONMENT.md). This is still
trusted-local configuration isolation, not a security sandbox.

## Model-backed short run (legacy environment)

Configure the executor and improver models, and their credentials, as described
in [setup](SETUP.md#models-and-credentials). E-commerce does **not** need
`HARNESS2_JUDGE_MODEL`. Full setup installs the OpenCode runtime as usual; Codex
additionally needs the configuration from [the Codex setup guide](SETUP.md#codex-executor).
Preview first:

```bash
harness2-setup ecom --dry-run
harness2-setup ecom
export HARNESS2_ECOM_TASKS_FILE="$PWD/examples/ecom-smoke-tasks.json"
export HARNESS2_RUNS_DIR="$PWD/runs/ecom-model-smoke"
harness2 --bench ecom --max-tasks 1 --dry-run
harness2-check --bench ecom
# These commands call configured executor/improver models and incur costs.
harness2 --bench ecom --k 1 --mode parallel --max-tasks 1
harness2-judge --bench ecom --k 1 --mode parallel --bases
harness2-report --bench ecom --k 1 --mode parallel --bases
```

Keep all exports and run flags identical for judging/reporting. Add
`--substrate cx` to every command, including setup, to use Codex. The real paid-model-backed path and a surviving 365-day model evaluation remain
unverified. Real OpenCode executor/improver CLI requests have since been tested
against a local fake model API for configuration isolation, not business quality;
those checks are not a substitute for model-backed evaluation. A short environment smoke is not evidence of
harness quality or a reproduction of MerchantBench's paper results.

## Corrected budget policy and paired evaluation

The operating horizon ends new business, not existing-order settlement. Both
substrates' baseline prompts and the common task brief now distinguish the
order-drain period and the externally measured final score. Historical run
artifacts are unchanged. A corrected candidate with checked whole-shelf demand
budgets lives at `examples/ecom-budget-harness/oc`.

See [frozen paired evaluation](ECOM_PAIRED.md) for offline fine reconciliation,
10–20 seed × repeated base/candidate snapshots, bounded paid execution, and
complete-pair statistics with seed-clustered intervals. This evaluates a fixed
policy; it neither reruns the failed inv1/dom1 improvers nor claims the recursive
pipeline is repaired. Keep k=1 pending reliable multi-seed evidence.

## Tasks and reproducibility

The bundled manifest `harness2/benchmarks/ecom_tasks.json` defines **five local
experiment instances**, seeds 42–46 in domain `synthetic`. These are not an
upstream published task split. Each uses the unmodified default scenario's
**8,760 hourly horizon steps / 365 days**, initial cash 2,000 and deposit 1,000.
Catalog generation and simulator randomness use the task's master seed. Any
catalog/filtering choices made by an agent should be logged separately.

`HARNESS2_ECOM_TASKS_FILE` selects a custom version-1 JSON manifest:

```json
{
  "version": 1,
  "tasks": [
    {"id": "synthetic/short-seed42", "domain": "synthetic", "master_seed": 42, "horizon_steps": 24}
  ]
}
```

- Task IDs are `domain/name`; seeds are nonnegative integers. Optional
  `horizon_steps` must be a positive integer. Unknown fields, duplicates and
  traversal-like names are rejected. No custom/private scenarios are supported.
- `HARNESS2_ECOM_TIMEOUT` is the **whole rollout** wall-clock budget, default
  10,800 seconds. A full model-driven year may need more time and substantial
  API budget. The upstream hook timeout and all other default scenario settings
  are retained. Late settlement can make recorded `n_steps` exceed the horizon.
- HARNESS² `--seed` controls task ordering; it does **not** replace master seeds
  in the task manifest. `--max-tasks` remains a per-domain limit.
- Run identity records source pin, task manifest, adapter/runtime/transport and
  baseline fingerprints. Reuse requires matching identity and intact native
  measurement/trajectory/history. Truncations, failed executors and missing or
  changed artifacts are not accepted as completed results.
- Change `HARNESS2_RUNS_DIR` when changing manifest, models, code, timeout or
  protocol. Keep the manifest unchanged across refinement, judging and reports.
- `scripts/split_tasks.py --bench ecom` partitions manifest instances using the
  normal HARNESS² protocol. **`--split test` does not freeze library updates**;
  it is not by itself a frozen generalization test. Similar seeds are not
  independent product/supplier distribution shifts.

## Grading and reporting

After the solver observes terminal and exits successfully, the controller waits
for upstream run status `finished`, retrieves authoritative merchant series,
and stores them in `measurement/episode.json`. `harness2-judge` calls the pinned
upstream `eval.scoring.compute` outside the agent workspace. Missing/failed
measurement is **ungraded**, not a zero score. Normal upstream business death
(`is_alive=false`) is different: it is a valid completed outcome and keeps the
upstream net-asset score, including any settlement after early termination.

The headline metric is upstream **final net assets**, in native simulator
currency units. Reports show its arithmetic mean, mean net profit and survival
rate. Zero and negative asset scores remain valid. This is not a pass rate:
there is no all-pass threshold, percentage scaling, or cross-benchmark score
combination. Internal `n_passed=value, n_criteria=1` fields are compatibility
carriers only, with `metric="final_net_assets"` and `all_pass=false`.
The existing 0–100 pass-fraction report behavior is unchanged for other adapters.

Native measurements, server state and executor logs are under
`runs/ecom/episodes/` (relative to the chosen run root). Task summaries, judged
results and the persistent improvement library use the usual HARNESS² layout.
Do not give the episode storage root to an evaluator as a solver workspace.

## Implementation verification

The implementation was checked on macOS / Python 3.12 without paid model calls:

- All **20** regression tests passed, including four native integration tests.
- Ruff, Git whitespace checks, all 47 existing release-asset checksums, and wheel
  inclusion checks for the new runtime assets passed. The upstream wheel build
  still warns about pre-existing absent `.env.example` / packaged third-party
  paths; that packaging cleanup is outside this extension.
- Seed 42, **24-hour** horizon, upstream random rule-based policy: native assets
  **3962.06**, profit **962.06**, alive, terminal `finished`. An independent repeat
  reproduced the saved metrics exactly.
- Seed 42, **8760-hour configured horizon**, same policy: assets **3657.99**,
  profit **657.06**, **business death at t=39**, terminal `finished`. This validates
  normal early-termination measurement, not a surviving 365-day run. Both cases
  recorded 193 steps because upstream settlement continues beyond agent activity.
- Real OpenCode/Codex model calls, model-backed recursive improvement and a
  surviving full-year model trajectory remain unverified.

Local measurement files are `runs/merchantbench-smoke-1d.json`,
`runs/merchantbench-smoke-repeat-1d.json` and
`runs/merchantbench-smoke-365d.json`; they are generated artifacts, not a
redistributed benchmark dataset.
