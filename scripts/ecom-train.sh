#!/bin/sh
# Parameterised, detached training; HTTP monitoring is loopback and read-only.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
exec "$ROOT/.venv/bin/python" -m harness2.cli.ecom_train "$@"
