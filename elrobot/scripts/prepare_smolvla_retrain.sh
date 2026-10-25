#!/usr/bin/env bash
# Prepare the upload bundle; never uploads or starts training.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${SMOLVLA_PYTHON:-$ROOT/../.venv-smolvla/bin/python}"
[[ $# == 1 || $# == 2 ]] || { echo 'Usage: bash scripts/prepare_smolvla_retrain.sh /path/to/dataset [validation-episodes=5]' >&2; exit 2; }
SOURCE="$(realpath "$1")"
NAME="$(basename "$SOURCE")-smolvla"
DEST="$ROOT/../data/$NAME"
BUNDLE="$ROOT/../outputs/deploy/$NAME-devlab.tar.gz"
"$PYTHON" "$ROOT/scripts/prepare_smolvla_dataset.py" --source "$SOURCE" --destination "$DEST" --validation-episodes "${2:-5}"
"$PYTHON" "$ROOT/scripts/package_smolvla.py" devlab --dataset-root "$DEST" --output "$BUNDLE"
cat <<EOF
Upload bundle: $BUNDLE
In the existing Devlab terminal after uploading to /home/jovyan/work:
export SMOLVLA_WORKSPACE=/home/jovyan/work
export HF_HOME=/home/jovyan/work/hf-cache
cd /home/jovyan/work
tar -xzf '$NAME-devlab.tar.gz'
cd pizza-smolvla
sha256sum -c SHA256SUMS
cd elrobot
bash scripts/train_smolvla.sh --profile devlab \\
  --dataset-root '../data/$NAME' \\
  --steps 20000 --output-dir '../outputs/train/$NAME-20k'
EOF
