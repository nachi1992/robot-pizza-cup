#!/usr/bin/env bash
# Export and inspect three Station demos. Does not start teleop or move motors.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 2 || $# -gt 3 || ! "$1" =~ ^[0-9]+$ || ! "$2" =~ ^[0-9]+$ ]]; then
  echo 'Usage: bash scripts/milestone4.sh START_FRAME END_FRAME [OUTPUT_DIRECTORY]' >&2
  echo 'See docs/milestone4.md for the three-demo tagging procedure.' >&2
  exit 2
fi
source "$ROOT/scripts/load_config.sh"
require_var TASK
PYTHON="$ROOT/../so101/.venv/bin/python"
GENERATOR="$NORMA_CORE_ROOT/bin/dataset-generator"
PREVIEW="$NORMA_CORE_ROOT/bin/dataset-mp4"
OUTPUT="${3:-$ROOT/../data/milestone4-$(date +%Y%m%d-%H%M%S)}"
if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite existing output: $OUTPUT" >&2
  exit 1
fi
[[ -x "$GENERATOR" && -x "$PREVIEW" && -x "$PYTHON" ]] || {
  echo 'Missing Station export tools or so101 Python environment.' >&2; exit 1;
}
"$PYTHON" -c 'import sys; assert int(sys.argv[1]) < int(sys.argv[2]), "START_FRAME must precede END_FRAME"' "$1" "$2"
mkdir -p "$OUTPUT"
OUTPUT="$(cd "$OUTPUT" && pwd)"
"$GENERATOR" -no-tui -robot "$STATION_HOST" -from "$1" -to "$2" \
  -episode.duration "${EPISODE_DURATION:-45}" -task "$TASK" -output "$OUTPUT" \
  2>&1 | tee "$OUTPUT/export.log"
"$PYTHON" "$ROOT/scripts/inspect_station_dataset.py" "$OUTPUT" --episodes 3 --task "$TASK"
for episode in "$OUTPUT"/*.parquet; do
  "$PREVIEW" --input "$episode" --output "${episode%.parquet}.mp4"
done
printf '\nExport: %s\nWatch one MP4 and identify camera-0/camera-1 in inspection/.\n' "$OUTPUT"
