#!/usr/bin/env bash
# Apply the low-gain, normal-colour KD-USB profile. Stop Station first.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ $# -gt 0 ]] || set -- /dev/video0 /dev/video2
bash "$ROOT/scripts/camera_exposure.sh" manual 200 "$@"
if command -v v4l2-ctl >/dev/null; then
  TOOL="$(command -v v4l2-ctl)"
else
  TOOL=/tmp/pizza-v4l-tools/root/usr/bin/v4l2-ctl
  export LD_LIBRARY_PATH="/tmp/pizza-v4l-tools/root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
for device in "$@"; do
  "$TOOL" -d "$device" --set-ctrl=exposure_dynamic_framerate=0,backlight_compensation=0,gain=20,brightness=0,gamma=100,contrast=40,saturation=64,hue=0,white_balance_automatic=1,sharpness=3,power_line_frequency=1
  "$TOOL" -d "$device" --get-ctrl=auto_exposure,exposure_time_absolute,exposure_dynamic_framerate,backlight_compensation,gain,brightness,gamma,power_line_frequency
done
