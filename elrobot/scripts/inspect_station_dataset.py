#!/usr/bin/env python3
"""Validate Station episode Parquets and extract numbered camera previews.

Reads exported files only; never connects to or commands hardware.
"""
import argparse
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

COLUMNS = ['episode_start_ns', 'timestamp_ns_since_episode_start',
           'joints', 'images', 'task']


def inspect_dataset(root, expected_episodes, task):
    root = Path(root)
    files = sorted(root.glob('*.parquet'))
    if len(files) != expected_episodes:
        raise ValueError(f'Expected {expected_episodes} episode files; found {len(files)} in {root}')
    summaries = []
    previews = []
    episode_ids = set()
    for path in files:
        count, previous, episode_id, first = 0, None, None, None
        for batch in pq.ParquetFile(path).iter_batches(columns=COLUMNS, batch_size=128):
            for row in batch.to_pylist():
                where = f'{path.name}, row {count}'
                if row['task'] != task:
                    raise ValueError(f'{where}: unexpected task {row["task"]!r}')
                if len(row['joints'] or []) != 8:
                    raise ValueError(f'{where}: expected eight joint values')
                images = row['images'] or []
                if len(images) != 2 or any(not image['jpeg'] for image in images):
                    raise ValueError(f'{where}: expected two nonempty camera images')
                stamp = row['timestamp_ns_since_episode_start']
                if stamp is None or stamp < 0 or (previous is not None and stamp <= previous):
                    raise ValueError(f'{where}: timestamps must increase strictly')
                if previous is not None and stamp - previous > 500_000_000:
                    raise ValueError(f'{where}: gap between frames exceeds 500 ms')
                if first is None:
                    first = stamp
                    episode_id = row['episode_start_ns']
                    if episode_id in episode_ids:
                        raise ValueError(f'{where}: episode ID repeated across files')
                    episode_ids.add(episode_id)
                    for index, image in enumerate(images):
                        with Image.open(io.BytesIO(image['jpeg'])) as frame:
                            previews.append((f'{path.stem}-camera-{index}.jpg', frame.convert('RGB')))
                if row['episode_start_ns'] != episode_id:
                    raise ValueError(f'{where}: multiple episode IDs inside one file')
                previous = stamp
                count += 1
        if count < 2:
            raise ValueError(f'{path.name}: expected at least two frames')
        summaries.append(dict(file=path.name, episode_id=episode_id, frames=count,
                              duration_s=(previous - first) / 1e9, joints=8, cameras=2))
    output = root / 'inspection'
    output.mkdir(exist_ok=True)
    for name, frame in previews:
        frame.save(output / name)
    (output / 'summary.json').write_text(json.dumps(dict(
        task=task, episodes=summaries,
        camera_mapping='Unverified: identify camera-0 and camera-1 from previews; wrist=front, overhead=top.'
    ), indent=2) + '\n')
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--task', default='place pepperoni on pizza')
    args = parser.parse_args()
    try:
        summaries = inspect_dataset(args.directory, args.episodes, args.task)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f'Dataset check failed: {error}\n')
    for summary in summaries:
        print(f'{summary["file"]}: {summary["frames"]} frames, '
              f'{summary["duration_s"]:.1f}s, 8 joints, 2 cameras')
    print(f'Checks passed. Review numbered camera images and summary in {args.directory / "inspection"}')
    print('This checks structure and timing; watch the videos to judge task success and synchronization.')


if __name__ == '__main__':
    main()
