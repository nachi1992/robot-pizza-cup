# ACT training and resume

ACT training uses native LeRobot 0.6.1 in `so101/.venv`. It does not connect to Station. Commands below run from `elrobot`; install the environment using the [setup guide](../README.md) and download the published dataset using the [asset instructions](smolvla-pipeline.md#setup-and-download).

## Train on the published dataset

```bash
DATASET_ROOT="$PWD/../data/lerobot-cheese-takes-20261025-from-015059-v3-clean-no0-smolvla" \
DATASET_REPO_ID=local/cheese-v3 \
OUTPUT_DIR="$PWD/../outputs/train/act-cheese-v3" \
DASHBOARD_PORT=8758 \
bash scripts/train_act_local.sh 5000
```

The `smolvla` suffix is a historical dataset name; its eight-dimensional state/action and named RGB features are also usable by ACT. The top crop is already applied. Supply both dataset variables explicitly because the launcher's historical defaults select an older dataset.

Append `--dry-run` to inspect the command without training or launching the dashboard. For an actual new run, select an output name that does not exist.

The existing configuration uses batch size 8, CUDA AMP, a pretrained ResNet-18, learning rate 1e-5, 20-action chunks, five actions before replanning and one observation step. It logs every 20 updates and evaluates/saves every 250 steps. State/action normalization is fitted only to training frames; images use ACT's ImageNet normalization. Training augmentations and Hub uploads are disabled.

The native 20% whole-episode holdout chooses nine validation episodes on this 42-episode dataset, leaving 33 for training. This is different from the prepared SmolVLA 35/7 split. The exact selected episodes, frame counts, normalization and dataset fingerprint are saved in `split_and_normalization.json`.

## Continue a local run

```bash
DASHBOARD_PORT=8758 bash scripts/train_act_local.sh 10000 \
  --resume ../outputs/train/act-cheese-v3/checkpoints/005000
```

The number is the cumulative target, not an additional step count. Resume requires the full checkpoint, including `training_state/`, the original run configuration and split/normalization record. It restores optimizer/RNG state and rejects changed data, splits or preprocessing. The current launcher resumes into the original output directory; it is not a general checkpoint relocation tool.

## Training dashboard

The launcher starts the existing local dashboard automatically; the example uses http://localhost:8758. It reads the training log and remains available after training ends. Use `DASHBOARD=false` to disable it or another `DASHBOARD_PORT` if occupied. Dashboard startup failure does not stop training; inspect the printed dashboard log path.

Watch training and held-out losses together, finite gradients, learning rate and throughput. Falling training loss with worsening validation loss suggests overfitting. A flat validation loss is a reason to inspect held-out demonstrations and input consistency; more updates alone are not evidence of better robot behavior.

Each checkpoint contains the policy, native processors, saved split/normalization and preprocessing contract. Use the matching dataset and an explicit checkpoint for [ACT Station inference](act-station-inference.md). No ACT weights are published as part of this submission.

## Historical preparation helpers

`prepare_act_cheese.py` contains source-specific recipes for older datasets, not a generic filter for the published 42 episodes. Do not apply its fixed episode indexes to another recording. For a fresh Station export, follow the [generic preparation workflow](station-to-lerobot.md#prepare-a-new-training-copy) into a new destination.
