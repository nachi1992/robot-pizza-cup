#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${SMOLVLA_PYTHON:-$ROOT/../.venv-smolvla/bin/python}"
[[ -x "$PYTHON" ]] || { echo 'Run bash scripts/bootstrap_smolvla.sh local (or devlab) first.' >&2; exit 1; }
exec "$PYTHON" -u "$ROOT/scripts/train_smolvla.py" "$@"
