#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

exec "$PYTHON_BIN" results/run_full12.py "$@"
