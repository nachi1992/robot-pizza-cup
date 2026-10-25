#!/usr/bin/env bash
# Install into a separate environment. Never modifies so101/.venv or drivers.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
PROFILE="${1:-local}"
[[ "$PROFILE" == local || "$PROFILE" == devlab ]] || { echo 'Usage: bootstrap_smolvla.sh [local|devlab]' >&2; exit 2; }
if [[ "$PROFILE" == devlab ]]; then
  [[ -n "${SMOLVLA_WORKSPACE:-}" && -d "$SMOLVLA_WORKSPACE" && -w "$SMOLVLA_WORKSPACE" ]] || {
    echo 'Set SMOLVLA_WORKSPACE to the persistent Devlab workspace directory first.' >&2; exit 1; }
  case "$REPO/" in "$(realpath "$SMOLVLA_WORKSPACE")/"*) ;; *)
    echo 'Extract the bundle inside SMOLVLA_WORKSPACE so environments and outputs persist.' >&2; exit 1;; esac
fi
INSTALLER="${SMOLVLA_UV:-$(command -v uv || true)}"
if [[ -z "$INSTALLER" && -x "$REPO/so101/.venv/bin/uv" ]]; then INSTALLER="$REPO/so101/.venv/bin/uv"; fi
PYTHON="${SMOLVLA_BOOTSTRAP_PYTHON:-python3.12}"
if [[ "$PROFILE" == devlab ]]; then
  if [[ -z "$INSTALLER" ]]; then
    if [[ -x "$HOME/.local/bin/uv" ]]; then INSTALLER="$HOME/.local/bin/uv";
    else
      curl -LsSf https://astral.sh/uv/install.sh | sh
      INSTALLER="$HOME/.local/bin/uv"
    fi
  fi
  # Use managed CPython headers, following the upstream Devlab guide. This
  # pipeline keeps its validated 3.12 lock instead of the SO-101 3.13 environment.
  unset UV_PYTHON
  "$INSTALLER" python install 3.12
  PYTHON="$("$INSTALLER" python find --managed-python 3.12)"
fi
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,12), "Python 3.12 required"'
command -v nvidia-smi >/dev/null || { echo 'NVIDIA driver/GPU unavailable.' >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo 'Install FFmpeg first: sudo apt install -y ffmpeg python3.12-venv' >&2; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
VENV="${SMOLVLA_VENV:-$REPO/.venv-smolvla}"
if [[ ! -x "$VENV/bin/python" ]]; then
  if [[ -n "$INSTALLER" ]]; then "$INSTALLER" venv --python "$PYTHON" "$VENV";
  else "$PYTHON" -m venv "$VENV"; fi
fi
if [[ "$PROFILE" == devlab ]]; then
  "$VENV/bin/python" -c 'import sys; assert "/.local/share/uv/python/" in sys.base_prefix or "/uv/python/" in sys.base_prefix, "Use a new SMOLVLA_VENV path: existing environment uses image Python"'
fi
df -h "$REPO"
if [[ -n "$INSTALLER" ]]; then
  # All indexes are official and every dependency is pinned. PyTorch's wheel
  # index also mirrors old general packages; use pip-compatible selection so
  # those mirrors do not shadow the locked PyPI versions.
  UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-180}" UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-2}" \
    "$INSTALLER" pip install --index-strategy unsafe-best-match --python "$VENV/bin/python" --requirement "$ROOT/config/smolvla-requirements.lock"
  "$INSTALLER" pip check --python "$VENV/bin/python"
else
  "$VENV/bin/python" -m pip install --requirement "$ROOT/config/smolvla-requirements.lock"
  "$VENV/bin/python" -m pip check
fi
"$VENV/bin/python" - <<'PY'
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from torchcodec.decoders import VideoDecoder
assert torch.cuda.is_available(), 'PyTorch cannot use CUDA; check the driver, do not replace it automatically'
x=torch.ones((16,16),device='cuda'); y=x@x; torch.cuda.synchronize()
assert y[0,0].item()==16
print('SmolVLA imports and CUDA calculation passed:', torch.cuda.get_device_name(0))
PY
printf 'Environment ready: %s\n' "$VENV"
