#!/bin/sh
# Configuration hygiene for the ecom OpenCode executor AND OpenCode improver.
# No personal files are changed; no credentials are embedded in this script.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
export HARNESS2_ENV_MODE=clean
export HARNESS2_SOLVER_MODEL="${HARNESS2_SOLVER_MODEL:-ecom_deepseek/DeepSeek-V4.1-Flash}"
export HARNESS2_IMPROVER_MODEL="${HARNESS2_IMPROVER_MODEL:-$HARNESS2_SOLVER_MODEL}"
export HARNESS2_CLEAN_MODEL_CONFIG="${HARNESS2_CLEAN_MODEL_CONFIG:-$ROOT/config/ecom-models.json}"
export HARNESS2_ECOM_TASKS_FILE="${HARNESS2_ECOM_TASKS_FILE:-$ROOT/examples/ecom-smoke-tasks.json}"
export HARNESS2_RUNS_DIR="${HARNESS2_RUNS_DIR:-$ROOT/runs/ecom-clean-deepseek-v41-flash-64k-high}"
export HARNESS2_ECOM_TIMEOUT="${HARNESS2_ECOM_TIMEOUT:-1800}"
# The configured gateway is on the LAN; do not send its credential via an outbound proxy.
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}${no_proxy:+$no_proxy,}10.52.16.108,127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
command=${1:-check}
if [ "$#" -gt 0 ]; then shift; fi
for arg in "$@"; do
    case "$arg" in --bench|--bench=*|--substrate|--substrate=*)
        echo 'This launcher fixes --bench ecom --substrate oc.' >&2; exit 2;;
    esac
done
case "$command" in
    prepare-tools) exec "$ROOT/.venv/bin/python" -m harness2.cli.ecom_tools "$@";;
    verify) exec "$ROOT/.venv/bin/python" -m unittest discover -s tests -p test_execution_env.py -v;;
    run) module=run_sweep;;
    check) module=check;;
    judge) module=judge;;
    report) module=report;;
    *) echo 'Usage: sh scripts/ecom-clean.sh {prepare-tools|verify|check|run|judge|report} [flags]' >&2; exit 2;;
esac
exec "$ROOT/.venv/bin/python" -m "harness2.cli.$module" --bench ecom --substrate oc "$@"
