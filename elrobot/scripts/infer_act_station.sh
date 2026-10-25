#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/load_config.sh"
require_var FOLLOWER_SERIAL
export NORMA_CORE_ROOT STATION_HOST FOLLOWER_SERIAL TASK
PYTHON="${ACT_PYTHON:-$ROOT/../so101/.venv/bin/python}"
[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }
exec "$PYTHON" -u "$ROOT/scripts/infer_act_station.py" "$@"
