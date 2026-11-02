# Clean OpenCode experiments

This is an opt-in **configuration-hygiene** policy for the MerchantBench
OpenCode executor **and** OpenCode improver. It does not edit personal skills,
MCP servers, login files, shell configuration or environment files. It is not an
OS sandbox, container, network firewall or publication-grade blinded evaluator.
Codex and the Claude CLI are deliberately rejected in this mode, rather than
claiming unverified isolation for them. Legacy behavior remains available when
`HARNESS2_ENV_MODE` is unset / `legacy`.

## Installed and ready to verify

From this source checkout:

```bash
cd /Users/sevenup/harness2
# Supply only an explicitly approved rg executable (example macOS location).
# Do not replace this with inheritance of your whole personal PATH.
sh scripts/ecom-clean.sh prepare-tools --rg /opt/homebrew/bin/rg
sh scripts/ecom-clean.sh verify
```

This runs unit tests and native OpenCode pollution checks with a **local fake
model API**. It does not call a paid model. Native tests require
`harness2-setup ecom`; they are skipped if the pinned OpenCode dependencies are
absent, or the explicit clean rg snapshot has not been provisioned. Inspect skip
counts before claiming native validation. The current native round-trip also
executes the real skill tool, rather than only enumerating available skills.

`prepare-tools` copies the selected executable into `dependencies/tools/clean-rg`,
records its SHA256 and version, and refuses to silently replace a different
snapshot. Clean runs verify it and copy only that binary into a private bin dir.
Its hash enters experiment identity. This bypasses the pinned OpenCode's broken
cold-download/extraction lifetime path without changing its source or inheriting
personal tool directories. It is trusted-local provenance, not a vendor signature
or a portable static binary: native shared-library dependencies remain host
requirements. Provision once on each host; a changed snapshot requires a new run
root. Missing/tampered tools fail before a model call.

The launcher fixes `--bench ecom --substrate oc` and enables the shared clean
policy for both roles. It defaults to the 24-hour task manifest, a 1,800-second
per-rollout timeout, and `runs/ecom-clean-deepseek-v41-flash-64k-high`.
These are not total cost limits.
The improver uses the existing OpenCode channel, not the Claude CLI.

## Models: explicit and separate from personal configuration

The user-selected local configuration now points to
`http://10.52.16.108:8081/v1`, model `DeepSeek-V4.1-Flash`, via provider
`ecom_deepseek`. The launcher defaults **both** executor and improver to
`ecom_deepseek/DeepSeek-V4.1-Flash`; explicit `HARNESS2_SOLVER_MODEL` /
`HARNESS2_IMPROVER_MODEL` shell settings still take precedence.

The gateway is reachable, identifies itself as Sub2API, and returned HTTP 401
`API_KEY_REQUIRED` for an unauthenticated `/v1/models` request. Set
`ECOM_LLM_API_KEY` securely in the invoking environment before checking/running.
No real key is stored in the profile, and no inference request was sent to this
service during configuration. `/v1` is the chosen OpenAI-compatible route;
authenticated model availability, streaming and tool-call compatibility still
require verification. Missing declared credentials fail before CLI startup.

The profile declares the **official 1,048,576-token context capacity**, with a
user-selected **65,536-token per-response output ceiling** (64K, not the official
393,216-token maximum). This ceiling is not a whole-experiment cost budget.
Thinking is explicitly enabled with the official default effort **high**; `low`, `high`
and `max` variants are declared so the improver's `--variant high` really maps
to a request option. The executor uses the model-level high default. The clean
policy lifts the separate OpenCode 32K/64K ceilings to the selected model's
explicit output limit for both roles, rather than only changing JSON metadata.

Official sources:
- [Model details](https://api-docs.deepseek.com/quick_start/pricing): 1M context,
  maximum 384K output, default thinking mode.
- [Official model config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json):
  `max_position_embeddings=1048576`.
- [Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion):
  exact output maximum 393,216. If `max_tokens` is omitted, the official default
  is 8K non-thinking / 64K thinking / 128K max-effort thinking. This experiment
  explicitly requests **64K**, matching the documented high-thinking default.
- [Thinking mode](https://api-docs.deepseek.com/guides/thinking_mode):
  `thinking.type=enabled`, `reasoning_effort=high` by default, and reasoning
  content must be preserved across tool-call turns.

The LAN model alias `DeepSeek-V4.1-Flash` is retained; the official public API
uses `deepseek-flash`. These official capacities do not prove the private
gateway exposes the same limits or forwards the reasoning parameters. Local
fake-API tests verify both roles send `max_tokens=65536`, thinking controls,
and tool-call reasoning history, not that the private gateway accepts them.
A larger ceiling permits larger cost/latency; it does not force every response
to use the full token allowance. The selected HTTP endpoint sends credentials
without TLS; use only on a trusted LAN, or provide a trusted HTTPS endpoint.

For another built-in provider, the model-only profile may instead contain only
`{"provider": {}}`, with corresponding model/credential settings in the shell.

For a custom OpenAI-compatible Chat Completions service, replace that profile
with the provider object from the service configuration, for example:

```json
{
  "provider": {
    "ecom_custom": {
      "npm": "@ai-sdk/openai-compatible",
      "env": ["ECOM_LLM_API_KEY"],
      "options": {"baseURL": "https://YOUR-ENDPOINT/v1"},
      "models": {
        "solver": {"id": "YOUR-MODEL-ID", "tool_call": true,
                   "limit": {"context": 32768, "output": 4096}}
      }
    }
  }
}
```

Replace endpoint, model ID and limits with actual service values, then set:

```bash
export HARNESS2_SOLVER_MODEL="ecom_custom/solver"
export HARNESS2_IMPROVER_MODEL="$HARNESS2_SOLVER_MODEL"
# Supply ECOM_LLM_API_KEY securely in the invoking environment.
```

To use another profile, set `HARNESS2_CLEAN_MODEL_CONFIG` to its absolute path.
The old `OPENCODE_CONFIG` variable is **not** forwarded: the clean policy renders
its own private model config for every session. Only the providers selected by
the executor/improver are enabled. Model-profile `apiKey`, if needed, must be an
`{env:NAME}` reference; literal secrets, `{file:...}` expansion, arbitrary provider
modules, MCP, instructions, plugins and skill paths are rejected. Supported
bundled SDKs are enumerated in `harness2/core/execution_env.py`. The supported
profile schema is intentionally narrower than general OpenCode configuration.

Direct built-in credential forwarding supports OpenAI (`OPENAI_API_KEY`),
Anthropic (`ANTHROPIC_API_KEY`), Google (`GOOGLE_GENERATIVE_AI_API_KEY`) and Vertex
(`GOOGLE_APPLICATION_CREDENTIALS`). Other services must declare credential names
in their explicit provider profile. Unlike the legacy improver environment,
clean mode can explicitly forward a declared Anthropic credential.

**Vertex requires explicit `GOOGLE_APPLICATION_CREDENTIALS`**; it must not fall
back to the personal gcloud HOME. The credential file is explicitly accessible
to the trusted runtime; it is not made secret from an adversarial shell.
Project/location settings still use the existing HARNESS² configuration.

## Reasoning effort and RSI attribution

The current experiment remains **high/high** (executor/improver). Low effort is
a useful constrained-compute condition, not automatic evidence of stronger
harness RSI: a weaker baseline can inflate percentage gains, and a weak improver
can also fail to discover useful changes. For a self-improvement claim, pair a
low/low unchanged-harness baseline with a low/low improved harness on the same
held-out scenarios, and use high/high as a separate control. Report absolute
native asset differences, survival, failures, all improvement/evaluation tokens,
and repeated seeds rather than choosing the condition with the largest uplift.
High-improver/low-executor tests optimization of a cheaper executor, not an
all-low self-improvement loop. Equal effort labels/output ceilings do not imply
equal actual token or search budgets; include a matched-compute baseline.

Changing only model.options.reasoningEffort to low is **insufficient**: the
improver's explicit `--variant high` overrides that default. A future low/low
experiment must change both role controls and use a separate run root. The
current test split also updates the harness library, so it is **not frozen
held-out evaluation**; do not describe it as one without changing that procedure.

## Supply the gateway key (no environment file required)

The project does not generate a credential-bearing environment file. The Python
configuration optionally loads `/Users/sevenup/harness2/.env`; a missing file is
fine, and an already exported shell variable takes precedence. Dotfiles are
hidden by default in macOS Finder (toggle with Command-Shift-period). Do not
send keys to chat or put literal keys into `config/ecom-models.json`.

Recommended on the default macOS **zsh** terminal: prompt without echoing the
key or putting its value in shell history, then export it for this terminal:

```zsh
cd /Users/sevenup/harness2
read -r -s 'ECOM_LLM_API_KEY?Gateway API key: '
printf '\\n'
export ECOM_LLM_API_KEY
sh scripts/ecom-clean.sh check
```

On **Bash**, replace the `read` line with:

```bash
read -r -s -p 'Gateway API key: ' ECOM_LLM_API_KEY
```

Run subsequent experiment commands in the **same terminal**; exporting a
variable there does not inject it into an already-running Pi process or another
terminal. To remove it afterwards, run `unset ECOM_LLM_API_KEY`.

For persistent local configuration instead, **create/edit `.env` yourself** in
an editor, with just:

```dotenv
ECOM_LLM_API_KEY=YOUR_GATEWAY_API_KEY
```

Restrict the file to your account (e.g. `chmod 600 .env`), never commit it, and
avoid sharing its contents. This file is excluded by the project's `.gitignore`.
Agents must not read or modify protected environment files; the credential stays
under your control. The launcher already supplies both role model defaults, so
no model ID or real key belongs in the JSON profile. `check` validates presence
only; it does not authenticate against the gateway or send an inference call.

## Run and compare

For the corrected budget strategy, use the [frozen paired protocol](ECOM_PAIRED.md)
rather than treating the ordinary adaptive sweep as a frozen generalization test.
Preparing its 10-seed × 3-repeat manifest is offline; model execution requires a
separate bounded `pair run` invocation.


```bash
# Select a fresh root after the snapshot-policy fix; do not reuse old identities.
export HARNESS2_RUNS_DIR="$PWD/runs/ecom-clean-snapshot-fixed-v1"
sh scripts/ecom-clean.sh check
sh scripts/ecom-clean.sh run --k 1 --mode parallel --max-tasks 1
sh scripts/ecom-clean.sh judge --k 1 --mode parallel
sh scripts/ecom-clean.sh judge --k 1 --mode parallel --bases
sh scripts/ecom-clean.sh report --k 1 --mode parallel
```

`judge` without `--bases` scores the selected final execution; `judge --bases`
scores reference executions separately. `report` does not accept `--bases`.

`check` does not call a model and does not prove endpoint availability. `run`
calls configured executor/improver models and incurs costs; even `k=1` uses
multiple sessions. Keep model settings, profile, task manifest and run root
identical for all commands. Change `HARNESS2_RUNS_DIR` for a new experiment when
changing endpoint/model mappings, environment policy or experimental settings.

Do not call the ordinary `harness2` command and assume it is clean: use this
launcher, or explicitly export `HARNESS2_ENV_MODE=clean` yourself. The clean
policy is certified only for ecom/oc; the normal sweep rejects other cells in
clean mode. Leaving the launcher does not change the parent shell.

## What is isolated

Each child session receives private HOME, all four XDG directories, TMPDIR and
OpenCode configuration directories. It starts from an environment allowlist,
not the operator's full environment. Personal OpenCode overrides, BASH_ENV,
NODE_OPTIONS, Python injection variables, unrelated secrets and arbitrary PATH
entries are not inherited. A minimal system PATH plus the experiment's pinned
Bun and MerchantBench Python environments remains available, along with only the
explicitly provisioned and content-checked private rg executable.

Explicit HTTP(S)/SOCKS proxy and CA-certificate environment settings are retained
for connectivity. They are configuration dependencies, not a network sandbox.
Model credential values are excluded from environment audit files and cache
identity. No real credential values are written into the generated model config.

OpenCode's external `.agents` / `.claude` skill discovery, Claude global prompt,
default external plugins, automatic updates, model-catalog fetching and LSP
downloads are disabled. A fresh Git root prevents discovery from walking into
parent projects. **Project configuration is not globally disabled**: experiment
AGENTS.md, `.opencode` skills, subagents, scripts and experiment-local plugins
remain available. Fixed OpenCode built-in components also remain; this is not a
claim that the agent has zero tools or plugins. Remote / out-of-workspace plugin
specs and MCP declarations fail the audit.

Before a rollout or improver calls a model, the installed pinned OpenCode is
queried for resolved config and available skills, and its instruction service
is queried for actual instruction paths. Outside skills/instructions cause a
failure rather than silent fallback. This observes startup configuration; it
does not constrain arbitrary behavior of deliberately supplied harness code.

## Snapshot boundary

The shared clean policy appends `/.harness2-env/` to the workspace Git
`info/exclude` before audit or model startup. Existing rules are preserved,
repeated preparation is idempotent, and Git resolves the exclusion path for
linked worktrees too. No experiment `.gitignore` or personal Git configuration
is changed. Snapshots stay enabled for task, harness and output files.

This exclusion is required because OpenCode stores its snapshot Git objects
under `XDG_DATA_HOME`, inside `.harness2-env/`. Without it, each snapshot can
stage its own object store, database and other runtime files; repeated
track/patch calls grow progressively slower even with unchanged task data.
Increasing the timeout or restarting repeats that defect. This is a snapshot
performance boundary, not a filesystem or credential security boundary.

A native regression performs 12 rounds with the real pinned Snapshot service,
checks that no runtime files or snapshot objects enter its index, and verifies
that a real harness edit still appears in the diff. A positive control removes
only the exclusion and confirms self-inclusion. The full local suite passed
**57 tests with no skips** after this fix, using no paid model calls. This does
not establish a successful model-backed RSI run; use a new run root for that
verification, since the policy source hash changes experiment identity.

## Evidence and resume

- Executor: `environment.json` in its ecom episode directory.
- Improver: `environments/<session>/environment.json` under its audit directory;
  per-attempt raw CLI logs and phase/last-event diagnostics live in
  `<label>.<key>.meta.attempts/<attempt>/`. Timeouts do not delete this evidence.
  These logs can contain task data; do not publish them without reviewing/redacting.
- Each snapshot contains the non-secret model profile/policy identity, resolved
  configuration, skill locations and instruction paths. Credential values are
  redacted. Treat snapshots as experiment metadata, not a dump of the shell env.
- Environment mode, policy source, improver source, OpenCode pin and model profile
  enter cache / experiment identities. Executor snapshots also enter artifact
  integrity checks, so modifying/deleting one prevents intact-cache reuse.
- This does not fully freeze OS binaries, dependency resolution, network services
  or every runtime tool effect. Strong reproducibility still needs an immutable
  image/toolchain and recorded effective dependency versions.

## Verification and safety boundary

The clean-environment implementation and gateway credential guard were verified with
**47 tests passed, with no skips** (full local suite),
including real pinned OpenCode executor/improver HTTP requests, Bash and skill-tool
round-trips against the local fake model, pollution positive/negative controls,
a clean MerchantBench rollout, live budget-helper decisions, timeout/process-group
cleanup, external fine reconciliation and bounded paired-run resume. Ruff, shell syntax, Git whitespace checks and
all 47 existing release-asset checksums passed. The clean launcher's dry-run
selects the 24-hour task. After selecting the DeepSeek gateway, preflight stops
at the missing `ECOM_LLM_API_KEY`; a model has now been selected, but authenticated
connectivity and paid inference remain unverified.

The native tests use a fabricated personal HOME containing forbidden global
skills, instructions and MCP config. They check both a positive contamination
control and the clean case, preserve the experiment skill, capture real
OpenCode model requests against a local mock service for executor and improver,
and complete a real MerchantBench episode with a deterministic stand-in solver.
The mock model proves launch/configuration plumbing, not model intelligence or
business performance. Real paid endpoints remain unverified.

A shell running as the same OS user can still read host files, explicitly start
other tools, use the network, or inspect simulator/scoring state. Only trusted
agents are appropriate. For adversarial or formally blinded evaluation, isolate
the executor filesystem/process identity and network from the controller and
grader; do not mount the personal HOME, grader data or Docker socket.

Rollback: stop using the launcher / unset `HARNESS2_ENV_MODE`, and choose a
separate legacy run root. No personal configuration or upstream benchmark data
needs restoring. Do not reuse clean results as legacy results or vice versa.
