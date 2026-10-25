#!/usr/bin/env bash
# Native LeRobot ACT: fresh run or resumable continuation. Never controls hardware.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
PYTHON="${ACT_PYTHON:-$REPO/so101/.venv/bin/python}"
STEPS="${STEPS:-1000}"
RESUME=""
DRY_RUN=false
DASHBOARD="${DASHBOARD:-true}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8765}"
if [[ $# -gt 0 && "$1" != --* ]]; then STEPS="$1"; shift; fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) [[ $# -ge 2 ]] || { echo '--resume needs a checkpoint directory' >&2; exit 2; }; RESUME="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) echo 'Usage: train_act_local.sh [TOTAL_STEPS] [--resume CHECKPOINT_DIR] [--dry-run]' >&2; exit 2 ;;
  esac
done
[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || { echo 'TOTAL_STEPS must be positive' >&2; exit 2; }
[[ "$DASHBOARD" == true || "$DASHBOARD" == false ]] || { echo 'DASHBOARD must be true or false' >&2; exit 2; }
[[ "$DASHBOARD_PORT" =~ ^[0-9]+$ ]] && (( DASHBOARD_PORT >= 1 && DASHBOARD_PORT <= 65535 )) || { echo 'Invalid DASHBOARD_PORT' >&2; exit 2; }
[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }

if [[ -n "$RESUME" ]]; then
  RESUME_INFO="$("$PYTHON" - "$RESUME" "$STEPS" <<'PY'
import json, sys
from pathlib import Path
checkpoint = Path(sys.argv[1]).resolve()
cfg_path = checkpoint / 'pretrained_model/train_config.json'
state = checkpoint / 'training_state'
for path in [cfg_path, checkpoint / 'pretrained_model/model.safetensors',
             state / 'training_step.json', state / 'optimizer_state.safetensors',
             state / 'optimizer_param_groups.json', state / 'rng_state.safetensors']:
    if not path.is_file():
        raise SystemExit(f'Incomplete resume checkpoint: {path}')
cfg = json.loads(cfg_path.read_text())
step = json.loads((state / 'training_step.json').read_text())['step']
if int(sys.argv[2]) <= step:
    raise SystemExit(f'TOTAL_STEPS must exceed saved step {step}; it is a cumulative target.')
if cfg.get('scheduler') is not None and not (state / 'scheduler_state.json').is_file():
    raise SystemExit('Checkpoint is missing its scheduler state')
output = Path(cfg['output_dir']).resolve()
if not (output / 'split_and_normalization.json').is_file():
    raise SystemExit(f'Missing original split/normalization record in {output}')
print(checkpoint)
print(output)
print(step)
PY
)"
  mapfile -t RESUME_VALUES <<< "$RESUME_INFO"
  RESUME="${RESUME_VALUES[0]}"
  OUTPUT_DIR="${RESUME_VALUES[1]}"
  SAVED_STEP="${RESUME_VALUES[2]}"
  LOG_FILE="$OUTPUT_DIR/train.log"
  TRAIN_ARGS=(--config_path="$RESUME/pretrained_model/train_config.json" --resume=true --steps="$STEPS")
  printf 'Resume step %s to cumulative step %s: %s\n' "$SAVED_STEP" "$STEPS" "$OUTPUT_DIR"
else
  DATASET_ROOT="${DATASET_ROOT:-$REPO/data/lerobot-cheese-takes-20261024}"
  DATASET_REPO_ID="${DATASET_REPO_ID:-n4chi/7-robopizzeria-cheese-v1}"
  BATCH_SIZE="${BATCH_SIZE:-8}"
  NUM_WORKERS="${NUM_WORKERS:-2}"
  AMP="${AMP:-true}"
  OUTPUT_DIR="${OUTPUT_DIR:-$REPO/outputs/train/act-cheese-$(date +%Y%m%d-%H%M%S)}"
  LOG_FILE="${OUTPUT_DIR}.log"
  [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ && "$NUM_WORKERS" =~ ^[0-9]+$ ]] || { echo 'Invalid batch size or workers' >&2; exit 2; }
  [[ "$AMP" == true || "$AMP" == false ]] || { echo 'AMP must be true or false' >&2; exit 2; }
  [[ ! -e "$OUTPUT_DIR" && ! -e "$LOG_FILE" ]] || { echo "Output already exists: $OUTPUT_DIR" >&2; exit 1; }
  "$PYTHON" - "$DATASET_ROOT" "$DRY_RUN" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
info = json.loads((root / 'meta/info.json').read_text())
for key in ['observation.state', 'action']:
    if info['features'][key]['shape'] != [8]:
        raise SystemExit(f'{key} must contain eight joint values')
for key in ['observation.images.top', 'observation.images.front']:
    if info['features'][key]['shape'] != [224, 224, 3]:
        raise SystemExit(f'{key} must be 224x224 RGB')
if info['fps'] != 10 or info['total_episodes'] < 5 or info['total_tasks'] != 1:
    raise SystemExit('Expected a single-task 10 Hz dataset with at least five episodes')
if sys.argv[2] != 'true':
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is unavailable; check your WSL GPU environment.')
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
print('Dataset:', root, info['total_episodes'], 'episodes,', info['total_frames'], 'frames')
PY
  TRAIN_ARGS=(
    --dataset.repo_id="$DATASET_REPO_ID" --dataset.root="$DATASET_ROOT"
    --dataset.eval_split=0.2 --dataset.image_transforms.enable=false --dataset.use_imagenet_stats=true
    --policy.type=act --policy.device=cuda --policy.use_amp="$AMP"
    --policy.chunk_size=20 --policy.n_action_steps=5 --policy.optimizer_lr=0.00001
    --policy.push_to_hub=false --steps="$STEPS" --batch_size="$BATCH_SIZE"
    --num_workers="$NUM_WORKERS" --prefetch_factor=2 --seed=42 --log_freq=20
    --eval_steps=250 --max_eval_samples=0 --env_eval_freq=0 --save_freq=250
    --save_checkpoint_to_hub=false --wandb.enable=false --job.target=local
    --output_dir="$OUTPUT_DIR" --job_name=act-cheese-local
  )
  printf 'Fresh ACT: %s updates; batch %s; output %s\n' "$STEPS" "$BATCH_SIZE" "$OUTPUT_DIR"
fi
if "$DRY_RUN"; then
  printf 'Dashboard auto-start: %s; port %s (not started in dry run).\n' "$DASHBOARD" "$DASHBOARD_PORT"
  printf 'Dry run only; trainer command: '
  printf '%q ' "$PYTHON" -u "$ROOT/scripts/train_act_entry.py" "${TRAIN_ARGS[@]}"
  printf '\n'
  exit 0
fi
mkdir -p "$(dirname "$OUTPUT_DIR")"
if "$DASHBOARD"; then
  "$PYTHON" -u "$ROOT/scripts/training_dashboard.py" --run-dir "$OUTPUT_DIR" --port "$DASHBOARD_PORT" \
    > "$OUTPUT_DIR.dashboard.log" 2>&1 &
  DASHBOARD_PID=$!
  sleep 0.2
  if kill -0 "$DASHBOARD_PID" 2>/dev/null; then
    printf 'Dashboard: http://localhost:%s (PID %s; stays available after training).\n' "$DASHBOARD_PORT" "$DASHBOARD_PID"
  else
    printf 'Dashboard did not start; training will continue. See %s.dashboard.log (the port may already be in use).\n' "$OUTPUT_DIR" >&2
  fi
fi
export PYTHONUNBUFFERED=1
set +e
if [[ -n "$RESUME" ]]; then
  "$PYTHON" -u "$ROOT/scripts/train_act_entry.py" "${TRAIN_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
else
  "$PYTHON" -u "$ROOT/scripts/train_act_entry.py" "${TRAIN_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
fi
TRAIN_STATUS=${PIPESTATUS[0]}
set -e
if [[ -z "$RESUME" && -d "$OUTPUT_DIR" ]]; then
  mv "$LOG_FILE" "$OUTPUT_DIR/train.log"
  LOG_FILE="$OUTPUT_DIR/train.log"
fi
printf '\nTraining exit status: %s\nLog: %s\n' "$TRAIN_STATUS" "$LOG_FILE"
exit "$TRAIN_STATUS"
