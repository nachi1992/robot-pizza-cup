# Team 6 — Robotpizzeria

An ElRobot cheese-placement demo built with leader/follower demonstrations, NormaCore Station and LeRobot 0.6.1. The repository includes **ACT and SmolVLA** data preparation, training and Station inference code.

```text
Teleoperate → tag Station history → LeRobotDataset → train → policy → follower
```

## Submission assets

| Asset | Link |
| --- | --- |
| Cleaned dataset | [team6-robotpizzeria-dataset-cheese-v3](https://huggingface.co/datasets/n4chi/team6-robotpizzeria-dataset-cheese-v3) |
| SmolVLA 20,000-step model | [team6-robotpizzeria-smolvla-cheese-v3](https://huggingface.co/n4chi/team6-robotpizzeria-smolvla-cheese-v3) |

The dataset contains **42 episodes / 6,357 frames at 10 Hz**, eight joint state/action dimensions and two 224 × 224 RGB cameras: wrist `front`, overhead `top`. Task: **place cheese on pizza**. Images are embedded in Parquet. The overhead crop is already applied to training images; live inference applies the checkpoint's saved crop once.

## Run it

Use the [short setup, inference, training and recording guide](elrobot/docs/smolvla-pipeline.md). It includes the working presentation command, ACT commands and the viewer that opens all episodes as separate Rerun sources.

Live SmolVLA inference works on the presentation setup. No measured task success rate is claimed. ACT is included as a pipeline; no ACT checkpoint is published with this submission. Large datasets, weights, generated outputs and filled environment files stay outside Git.

Built on [FunRobots](https://github.com/FunRobots/robot-pizza-cup), [NormaCore](https://github.com/norma-core/norma-core), [LeRobot](https://github.com/huggingface/lerobot) and [SmolVLA](https://huggingface.co/lerobot/smolvla_base), with Nebius GPU compute for training. The original [ElRobot hardware setup](elrobot/README.md) and [SO-101 starter](so101/README.md) remain available; this submission uses the eight-joint bridges in `elrobot/scripts`.
