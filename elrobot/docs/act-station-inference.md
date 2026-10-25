# ACT inference with Station

The existing ACT bridge loads an eight-joint checkpoint and its saved processors, reads Station observations and predicts normalized absolute goals. Use LeRobot 0.6.1 in the ACT environment. Always pass the checkpoint and dataset you actually trained with; historical defaults select an older pilot.

## Prepare the inputs

Start Station separately. Confirm the physical camera map (`front` wrist, `top` overhead), the dataset used by the model and matching saved preprocessing. Camera IDs can change on reconnect; see the [presentation setup](smolvla-pipeline.md#start-station-and-run-the-presentation).

Set the follower serial in the ignored `elrobot.env`. For physical movement, stop teleop, select **Self-controlled** in Station and enable follower torque separately using [the setup instructions](smolvla-pipeline.md#start-station-and-run-the-presentation). The runner does not enable torque.

## Predict without movement

From `elrobot`, after training the example ACT run:

```bash
bash scripts/infer_act_station.sh \
  --host localhost \
  --camera-map config/station-local-cameras.json \
  --dataset-root ../data/lerobot-cheese-takes-20261025-from-015059-v3-clean-no0-smolvla \
  --checkpoint ../outputs/train/act-cheese-v3/checkpoints/last/pretrained_model
```

Visually verify both views in `config/station-local-cameras.json` first; its saved IDs are specific to one camera attachment. An explicit map is required for this published dataset because it does not contain the historical Station export manifest that the ACT fallback expects.

This saves camera inputs and predictions under `outputs/inference/`. Add `--episode-seconds 10 --step-ticks 100` for a planning loop with no motor commands. A fresh `normvla` stream requires Station's selected torque-enabled follower bus; the bridge does not enable torque to obtain observations.

## Execute a bounded episode

```bash
bash scripts/infer_act_station.sh \
  --host localhost \
  --camera-map config/station-local-cameras.json \
  --dataset-root ../data/lerobot-cheese-takes-20261025-from-015059-v3-clean-no0-smolvla \
  --checkpoint ../outputs/train/act-cheese-v3/checkpoints/last/pretrained_model \
  --execute --episode-seconds 10 --step-ticks 100 --max-delta-ticks 200
```

`--step-ticks` bounds each submitted movement; it is not the number of episode steps. ACT accepts 1–500 ticks for either movement option and 1–60 seconds for an episode. Timed episodes require `--step-ticks`. These ACT limits differ from SmolVLA's current limits.

The bridge reads fresh observations and executes the first five actions of each 20-action prediction before replanning. It applies saved action postprocessing once, maps camera identities explicitly, and checks source motor telemetry. Model outputs use the current Station calibration to convert normalized goals to ticks. Check `calibration_matches_training` in the saved report; the ACT bridge does not enforce the same exact-calibration contract gate as SmolVLA.

Inspect `prediction.json` for raw positions, predicted goals, clipped/scaled commands, fresh frame IDs, observation/prediction ages, timing and the stop reason. Stationary predictions can reflect a learned fixed pose; increasing movement caps does not make a policy recover from unseen states.

Ctrl+C stops new submissions; the last servo goal remains active. Support the arm before switching follower torque off. Do not run teleop and policy control against the same follower at the same time.
