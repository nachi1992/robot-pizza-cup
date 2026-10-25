#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/load_config.sh"
require_var FOLLOWER_SERIAL
export NORMA_CORE_ROOT STATION_HOST FOLLOWER_SERIAL
exec "${ACT_PYTHON:-$ROOT/../so101/.venv/bin/python}" -u "$ROOT/scripts/follower_torque.py" "$@"
