#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/load_config.sh"
export NORMA_CORE_ROOT STATION_HOST FOLLOWER_SERIAL TASK
PYTHON="${SMOLVLA_PYTHON:-$ROOT/../.venv-smolvla/bin/python}"
[[ -x "$PYTHON" ]] || { echo 'Run scripts/bootstrap_smolvla.sh local first.' >&2; exit 1; }
exec "$PYTHON" -u "$ROOT/scripts/infer_smolvla_station.py" "$@"
