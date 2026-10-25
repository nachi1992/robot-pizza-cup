# Team 6 quick start: SmolVLA and ACT

Skip installation on the working presentation machine. The commands below document the existing pipeline; they do not change its code. Setup targets Linux/WSL, Python 3.12 and an NVIDIA GPU. WSL needs USBIPD attachment for the external cameras and arm adapters.

## Setup and download

From the repository root, create the ACT/data-tooling environment and separate SmolVLA environment:

```bash
cd so101
uv sync --python 3.12
cd ../elrobot
bash scripts/install.sh
cp -n elrobot.env.example elrobot.env
bash scripts/bootstrap_smolvla.sh local
cd ..
.venv-smolvla/bin/hf download n4chi/team6-robotpizzeria-dataset-cheese-v3 --repo-type dataset --local-dir data/lerobot-cheese-takes-20261025-from-015059-v3-clean-no0-smolvla
.venv-smolvla/bin/hf download n4chi/team6-robotpizzeria-smolvla-cheese-v3 --local-dir outputs/models/smolvla-cheese-v3-020000/smolvla-cheese-checkpoint/pretrained_model
cd elrobot
```

Enter your own bus serials and `TASK="place cheese on pizza"` in ignored `elrobot.env`. Keep credentials out of scripts; public assets do not need embedded tokens. Preserve all downloaded metadata, processors and `vlm_assets/`. Station needs a calibrated eight-joint follower and a fresh `inference/normvla` queue at 100 ms intervals; its auto producer requires exactly one torque-enabled bus. Do not replace the presentation machine's working configuration or calibration.

## Start Station and run the presentation

```bash
bash scripts/run.sh
```

UI: http://localhost:8889. Before policy control, stop teleop and select **Self-controlled** on the follower. Visually verify the camera map: `front` is wrist, `top` is overhead. Reconnecting cameras can change queue IDs; update the map and restart inference. The default `config/station-local-cameras.json` is specific to the current attachment; another map can be selected with `--camera-map`.

```bash
bash scripts/follower_torque.sh status --host localhost
bash scripts/follower_torque.sh on --host localhost
```

Working SmolVLA command, unchanged:

```bash
bash scripts/infer_smolvla_station.sh  --host localhost   --checkpoint ../outputs/models/smolvla-cheese-v3-020000/smolvla-cheese-checkpoint/pretrained_model   --dataset-root ../data/lerobot-cheese-takes-20261025-from-015059-v3-clean-no0-smolvla   --execute --episode-seconds 300   --max-age 3 --step-ticks 1000 --max-delta-ticks 1000
```

This explicitly sends motor goals for up to 300 seconds. Caps are ticks per commanded movement, not episode step counts. For one prediction with zero motor commands, remove both `--execute` and `--episode-seconds 300`; keep the checkpoint/dataset arguments. Run reports and initial camera images go under `outputs/inference/`. Ctrl+C stops new goals but leaves the last servo goal active. Support the arm before `bash scripts/follower_torque.sh off --host localhost`.

## Train SmolVLA or ACT

From `elrobot`, select the prepared dataset explicitly; historical script defaults use older datasets:

```bash
CHEESE_DATASET="$PWD/../data/lerobot-cheese-takes-20261025-from-015059-v3-clean-no0-smolvla"
bash scripts/train_smolvla.sh --dataset-root "$CHEESE_DATASET" --download-assets
bash scripts/train_smolvla.sh --dataset-root "$CHEESE_DATASET" --steps 20000 --output-dir ../outputs/train/smolvla-cheese-new
DATASET_ROOT="$CHEESE_DATASET" DATASET_REPO_ID=n4chi/team6-robotpizzeria-dataset-cheese-v3 OUTPUT_DIR="$PWD/../outputs/train/act-cheese-new" DASHBOARD_PORT=8758 bash scripts/train_act_local.sh 5000
```

Choose one training command at a time and a fresh output directory. Append `--dry-run` to inspect a training configuration without training. ACT starts its log dashboard automatically at the selected port. SmolVLA uses 35/7 train/validation episodes; ACT's existing 20% holdout uses 33/9 on this dataset. Their held-out losses therefore do not compare identical splits. The top crop is already applied; do not crop training images again.

Resume examples (full checkpoints including optimizer state are required):

```bash
bash scripts/train_smolvla.sh --dataset-root "$CHEESE_DATASET" --resume ../outputs/train/smolvla-cheese-new/checkpoints/010000 --steps 20000 --output-dir ../outputs/train/smolvla-cheese-resumed
bash scripts/train_act_local.sh 10000 --resume ../outputs/train/act-cheese-new/checkpoints/005000
```

Targets are cumulative. SmolVLA resume must retain the original target/schedule; the published inference-only model cannot provide optimizer resume. ACT resumes into the original run directory. For cloud training, `package_smolvla.py devlab --dataset-root ... --output ...` creates a bundle; `bootstrap_smolvla.sh devlab` and `train_smolvla.sh --profile devlab` use a persistent `SMOLVLA_WORKSPACE`.

After training ACT, predict without motor commands:

```bash
bash scripts/infer_act_station.sh --host localhost --camera-map config/station-local-cameras.json --dataset-root "$CHEESE_DATASET" --checkpoint ../outputs/train/act-cheese-new/checkpoints/last/pretrained_model
```

An ACT timed rollout additionally requires `--episode-seconds` (1–60) and `--step-ticks` (1–500); only `--execute` enables goals. Do not copy SmolVLA's 1,000-tick/300-second settings into ACT. Both bridges preserve saved normalization and preprocessing; SmolVLA also enforces the exact calibration/dataset contract.

## Record and inspect episodes

With Station running, teleoperate in one terminal and start tagging in another:

```bash
bash scripts/teleop.sh
# Separate terminal, when ready:
bash scripts/station_tag.sh
```

The helper starts immediately. `e` closes a take, waits five seconds and starts the next; `q` finishes. The exact-pair exporter accepts an inventory with `complete` entries containing `name`, `start: {name, pointer}` and `end: {name, pointer}` from actual Station tags. It preserves one pair per episode:

```bash
../so101/.venv/bin/python scripts/export_tagged_takes.py --inventory ../data/tags.json --output ../data/cheese-new.parquet --task 'place cheese on pizza'
bash scripts/station_dataset.sh --source ../data/cheese-new.parquet --name cheese-new --camera-names front,top
```

Use `front,top` only if exported previews confirm wrist first, overhead second; otherwise reverse them. Outputs must have new names. The generic `prepare_smolvla_dataset.py --source ... --destination ... --validation-episodes 7` prepares a curated copy, crop and split without modifying the source.

Load all published episodes as separate sources in one Rerun viewer:

```bash
../so101/.venv/bin/python -u scripts/view_all_episodes.py --repo-id n4chi/team6-robotpizzeria-dataset-cheese-v3 --root "$CHEESE_DATASET" --web-port 9090 --grpc-port 9876
```

Open http://localhost:9090/?url=rerun%2Bhttp%3A%2F%2Flocalhost%3A9876%2Fproxy when ready. Select an episode from the source list; timelines are separate. Generated `.rrd` files stay outside Git. Current episode numbers follow the prepared dataset; `dataset_preparation.json` records source indexes. Training loss and valid predictions do not replace repeated physical pickup/placement evaluation.
