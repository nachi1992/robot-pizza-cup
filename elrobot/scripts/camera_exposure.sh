#!/usr/bin/env bash
# Camera controls verified on KD-USB cameras. Does not stop Station or reconnect USB.
set -euo pipefail
usage() {
  cat <<'EOF'
Usage:
  bash scripts/camera_exposure.sh status [DEVICE ...]
  bash scripts/camera_exposure.sh manual VALUE [DEVICE ...]
  bash scripts/camera_exposure.sh auto [DEVICE ...]
Defaults: /dev/video0 /dev/video2. VALUE is in 100-microsecond units (50 = 5 ms).
Stop Station first. If video nodes are missing, reattach cameras via USBIPD.
EOF
}
MODE="${1:-help}"
case "$MODE" in help|-h|--help) usage; exit 0;; status|manual|auto) shift;; *) usage >&2; exit 2;; esac
if [[ "$MODE" == manual ]]; then
  VALUE="${1:-}"
  if [[ ! "$VALUE" =~ ^[0-9]+$ || ${#VALUE} -gt 4 ]]; then
    echo 'Manual exposure must be an integer between 1 and 5000.' >&2; exit 2
  fi
  VALUE=$((10#$VALUE))
  if (( VALUE < 1 || VALUE > 5000 )); then
    echo 'Manual exposure must be between 1 and 5000.' >&2; exit 2
  fi
  shift
fi
if pgrep -x station >/dev/null; then
  echo 'Station is running. Stop it first; this script will not interrupt it.' >&2; exit 1
fi
if command -v v4l2-ctl >/dev/null; then
  TOOL="$(command -v v4l2-ctl)"
  TOOL_LIBS="${LD_LIBRARY_PATH:-}"
elif [[ -x /tmp/pizza-v4l-tools/root/usr/bin/v4l2-ctl ]]; then
  TOOL=/tmp/pizza-v4l-tools/root/usr/bin/v4l2-ctl
  TOOL_LIBS="/tmp/pizza-v4l-tools/root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
  echo 'Install the camera tool: sudo apt install v4l-utils' >&2; exit 1
fi
[[ $# -gt 0 ]] || set -- /dev/video0 /dev/video2
# Check all devices before changing any.
for device in "$@"; do
  [[ -c "$device" && -r "$device" && -w "$device" ]] || {
    echo "Cannot access $device. Check USB attachment, device mapping and video group." >&2; exit 1;
  }
done
controls=auto_exposure,exposure_time_absolute,gain,backlight_compensation
for device in "$@"; do
  echo "=== $device: current settings ==="
  LD_LIBRARY_PATH="$TOOL_LIBS" "$TOOL" -d "$device" --get-ctrl="$controls"
  if [[ "$MODE" == manual ]]; then
    LD_LIBRARY_PATH="$TOOL_LIBS" "$TOOL" -d "$device" --set-ctrl=auto_exposure=1
    LD_LIBRARY_PATH="$TOOL_LIBS" "$TOOL" -d "$device" --set-ctrl="exposure_time_absolute=$VALUE"
  elif [[ "$MODE" == auto ]]; then
    LD_LIBRARY_PATH="$TOOL_LIBS" "$TOOL" -d "$device" --set-ctrl=auto_exposure=3
  fi
  if [[ "$MODE" != status ]]; then
    echo 'Applied settings:'
    LD_LIBRARY_PATH="$TOOL_LIBS" "$TOOL" -d "$device" --get-ctrl="$controls"
  fi
done
