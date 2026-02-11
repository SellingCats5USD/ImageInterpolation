#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d .venv ]]; then
  echo ".venv not found; run scripts/bootstrap.sh first"
  exit 1
fi

source .venv/bin/activate
python -m pytest -q tests/test_views.py
python -m src.run --help >/dev/null

echo "Smoke test passed"
