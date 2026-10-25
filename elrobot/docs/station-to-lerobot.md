# Station recordings → LeRobotDataset → Rerun

Commands run from `elrobot`. Export/conversion reads saved Station history; it does not command motors, recalibrate, start training or upload to Hugging Face. A viewer launches only when explicitly requested.

## Record boundaries

With Station and teleoperation already running and the scene ready:

```bash
bash scripts/station_tag.sh --host localhost
```

The helper starts a take immediately and listens to stdin. Press `e` without Enter to end it, wait five seconds for a reset, and start another. Press `q` or Ctrl+C to close the last take and exit; `q` during the pause cancels the next take. Repeated `e` presses during the pause do not queue recordings.

Names follow `cheese_take_DATE_TIME_start/end`. Confirmation means the tag was saved in Station. The ignored `data/station-tag-session.json` tracks the active pair. Read it through `bash scripts/station_tag.sh status`; one-shot `start`/`end` commands are also available. Starting the interactive helper creates a real tag, so do not launch it merely to inspect status.

Set `TASK="place cheese on pizza"` in `elrobot.env`. Task text is assigned during export, not encoded in the boundary names. The five-second scene reset remains outside the adjacent episodes.

## Exact tag pairs: one episode per take

For short explicit takes, use `export_tagged_takes.py`. Create a new inventory JSON containing the exact active names and integer frame pointers shown in Station History. The structure below is illustrative: **replace the names and pointers with your real tags**.

```json
{
  "complete": [
    {
      "name": "take_001",
      "start": {"name": "take_001_start", "pointer": 100},
      "end": {"name": "take_001_end", "pointer": 200}
    }
  ],
  "incomplete": []
}
```

Save it under ignored `data/`, for example `data/cheese-tag-inventory.json`. Include only complete pairs in `complete`; keep unfinished takes in `incomplete` until an end tag exists. Resolve missing/ambiguous names instead of substituting different endpoints. Preserve the inventory as dataset provenance.

With Station running and the requested history available:

```bash
../so101/.venv/bin/python scripts/export_tagged_takes.py \
  --host localhost \
  --inventory ../data/cheese-tag-inventory.json \
  --output ../data/cheese-new-station.parquet \
  --task 'place cheese on pizza'
```

Use a fresh output name. Optional repeated `--exclude-name BASE_NAME` arguments omit selected complete takes explicitly. The exporter checks the inventory against active Station tags, uses half-open `[start, end)` ranges and preserves each selected pair as its own episode. It does not apply automatic motion/minimum-duration filters or resample frames.

Outputs include the Parquet, a sibling `.manifest.json` and `.previews/` folder. The manifest records episode boundaries, frame counts, camera ordering and source frame gaps. Camera ordering is recovered from each source snapshot and normalized across the export. A changed camera identity set causes failure rather than silently mixing cameras.

Inspect both camera previews. Only when their identities are verified, convert using their actual image-list order. If stream 0 is wrist and stream 1 overhead:

```bash
bash scripts/station_dataset.sh \
  --source ../data/cheese-new-station.parquet \
  --name cheese-new --camera-names front,top
```

For the reverse order, use `top,front`. Do not infer physical names from old device numbers or earlier Station queue IDs. The converter creates `data/lerobot-cheese-new/`, refuses an existing destination, and reloads it to confirm frame/episode counts and image decoding. Existing source task labels are retained.

## Named range with automatic episode detection

For the older Station episode detector, resolve a single outer range first:

```bash
bash scripts/station_dataset.sh \
  --start-tag ACTUAL_START_TAG --end-tag ACTUAL_END_TAG \
  --name cheese-auto --task 'place cheese on pizza' --dry-run
```

Remove `--dry-run` to export/convert. The detector may split or discard episodes according to duration and command-count thresholds; this is not equivalent to one explicit pair per episode. Read its `.export.log`. Do not silently change thresholds to hide missing takes. The pinned release merges numbered temporary exports into a final sibling Parquet file.

## LeRobot mapping

| Station value | LeRobot feature |
| --- | --- |
| Eight `position_norm` values | `observation.state` |
| Eight `goal_norm` values | `action` |
| Two JPEG streams | `observation.images.front` and `.top` after verified naming |
| Episode-relative source nanoseconds | `source.timestamp_ns` and seconds in `source.timestamp` |
| Export task string | `task` |

Each source episode group becomes one LeRobot episode. Conversion stores embedded image data in Parquet, not separate videos. It names joint slots in source order; confirm their physical motor ordering before using new data for control. The current Station producer sorts its eight motor slots by ID.

The default LeRobot timeline is uniform 10 Hz, with no frame dropping/interpolation. Original episode-relative source timestamps remain in separate features, so real elapsed time can differ from the uniform timeline. Structural validity is not evidence of a successful grasp or exact camera/motor synchronization.

The converter reads small hackathon exports into memory. Large recordings need a streaming conversion strategy. Keep raw history and source exports separately from curated copies.

## Prepare a new training copy

First curate by reviewing complete episodes and preserving an episode mapping; the generic preparation script below does not judge success or automatically delete poor takes. Supply a new curated source and destination:

```bash
../so101/.venv/bin/python scripts/prepare_smolvla_dataset.py \
  --source ../data/lerobot-cheese-new \
  --destination ../data/lerobot-cheese-new-smolvla \
  --validation-episodes 7
```

The requested validation count must leave at least one training episode. The script selects whole held-out episodes using seed 42, moves them to the end, records train/validation frame counts, preserves eight-joint calibration and applies the saved 34-pixel top crop once. An existing crop sidecar is respected. If calibration is absent, the original Station Parquet referenced by conversion notes must still be available.

It refuses an existing destination and saves preparation/preprocessing/calibration metadata for matching live inference. The published cleaned dataset is already prepared; do not run this on it merely to use the submitted model. Historical `prepare_act_cheese.py` / `prepare_smolvla_cheese.py` contain recipes for specific older sources.

## View all episodes at once in Rerun

For the published cleaned dataset:

```bash
../so101/.venv/bin/python -u scripts/view_all_episodes.py \
  --repo-id local/cheese-v3 \
  --root ../data/lerobot-cheese-takes-20261025-from-015059-v3-clean-no0-smolvla \
  --web-port 9090 --grpc-port 9876
```

This writes one `.rrd` per episode in a new timestamped sibling folder, plus `sources.json`, then opens all files in a single web viewer. Each episode is a separate source with a unique recording ID and its own timeline beginning at zero. Select episodes from the source list; timelines are not concatenated.

Open http://localhost:9090/?url=rerun%2Bhttp%3A%2F%2Flocalhost%3A9876%2Fproxy after the helper reports readiness. Choose unused ports if another viewer is running. Ctrl+C stops the viewer. `--output-dir` chooses a new explicit recording directory; recordings are generated artifacts, not Git files.

Episode numbers refer to the dataset passed to the helper. Prepared datasets can reorder/renumber takes: use `dataset_preparation.json` to trace source indexes. When no Station manifest exists, the helper's `original` display label falls back to the current episode number, not the original raw take.

For one episode, use `lerobot-dataset-viz --repo-id REPO_ID --root DATASET_ROOT --episode-index 0 --mode distant`. To convert and open that single-episode viewer explicitly, add `--view` to `station_dataset.sh`. Conversion alone does not open a viewer.
