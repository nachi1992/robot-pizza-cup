#!/usr/bin/env python3
"""Prepare the agreed crop/split from local files; never connects to Station."""
import argparse
import copy
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from lerobot.datasets import LeRobotDataset

from act_image_preprocessing import dataset_fingerprint, preprocess_camera

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / 'data/lerobot-cheese-remote-20261024-144902'
DESTINATION = REPO / 'data/lerobot-cheese-remote-20261024-144902-train-v1'
TRAIN = [0, 1, 3, 4, 5, 6, 8, 9, 11, 14, 15, 16, 18, 19]
VALIDATION = [2, 7, 12, 17]
LEFT_DESTINATION = REPO / 'data/lerobot-cheese-left-crop15-v1'
LEFT_TRAIN = [0, 1, 3, 4, 5, 6, 8, 11, 14, 15, 18, 19]
LEFT_VALIDATION = [2, 7, 16, 17]
SPEC = {'version': 1, 'cameras': {'top': {
    'input_size': [224, 224], 'crop_left_pixels': 34,
    'output_size': [224, 224], 'resample': 'bilinear'}}}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def prepare(source, destination, left_only=False):
    if destination.exists():
        raise FileExistsError(f'Output already exists; refusing overwrite: {destination}')
    train = LEFT_TRAIN if left_only else TRAIN
    validation = LEFT_VALIDATION if left_only else VALIDATION
    order = train + validation
    before = dataset_fingerprint(source)
    info = json.loads((source / 'meta/info.json').read_text())
    manifest = json.loads((source / 'station_export_manifest.json').read_text())
    if info['total_episodes'] != 20 or info['total_frames'] != 3302:
        raise ValueError('Source differs from the inspected 20-episode, 3302-frame dataset')
    grouped = {}
    for path in sorted((source / 'data').rglob('*.parquet')):
        for row in pq.read_table(path).to_pylist():
            grouped.setdefault(row['episode_index'], []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: row['frame_index'])
    features = {k: v for k, v in info['features'].items()
                if k.startswith('observation.') or k in ('action', 'source.timestamp', 'source.timestamp_ns')}
    repo_id = f'local/{destination.name.removeprefix("lerobot-")}'
    dataset = LeRobotDataset.create(repo_id, fps=info['fps'], features=features,
                                   root=destination, robot_type=info['robot_type'],
                                   use_videos=False, image_writer_threads=2)
    mapping = []
    try:
        for new, original in enumerate(order):
            for row in grouped[original]:
                frame = {key: np.asarray(row[key], dtype=np.float32)
                         for key in ('observation.state', 'action')}
                frame['source.timestamp'] = np.asarray([row['source.timestamp']], dtype=np.float32)
                frame['source.timestamp_ns'] = np.asarray([row['source.timestamp_ns']], dtype=np.int64)
                frame['task'] = 'place cheese on pizza'
                for name in ('front', 'top'):
                    image = Image.open(io.BytesIO(row[f'observation.images.{name}']['bytes'])).convert('RGB')
                    frame[f'observation.images.{name}'] = np.asarray(preprocess_camera(image, name, SPEC))
                dataset.add_frame(frame)
            dataset.save_episode()
            mapping.append({'episode_index': new, 'source_episode_index': original,
                            'split': 'train' if new < len(train) else 'validation',
                            'frames': len(grouped[original])})
            print(f'Prepared episode {new} from source {original}: {len(grouped[original])} frames', flush=True)
    finally:
        dataset.finalize()
    write_json(destination / 'image_preprocessing.json', SPEC)
    new_manifest = copy.deepcopy(manifest)
    new_manifest['episodes'] = [dict(manifest['episodes'][old], episode_index=new,
                                   source_episode_index=old)
                                for new, old in enumerate(order)]
    new_manifest['frames'] = sum(item['frames'] for item in mapping)
    new_manifest['training_source'] = str(source)
    write_json(destination / 'station_export_manifest.json', new_manifest)
    notes = json.loads((source / 'conversion_notes.json').read_text())
    notes.update(repo_id=repo_id, training_source=str(source),
                 image_preprocessing='Top: crop left 34 pixels, bilinear resize to 224x224; front unchanged.')
    write_json(destination / 'conversion_notes.json', notes)
    raw = pq.read_table(Path(notes['source']), columns=['joints']).to_pylist()
    ranges = {tuple((j['range_min'], j['range_max']) for j in row['joints']) for row in raw}
    if len(ranges) != 1:
        raise ValueError('Source calibration changes; inspect before training')
    write_json(destination / 'meta/station_calibration.json', {
        'motor_ids': list(range(1, 9)), 'calibration_ranges': next(iter(ranges)),
        'source': notes['source'], 'source_frames_checked': len(raw),
        'motor_id_order_basis': 'Station normvla ST3215 producer sorts by motor ID; eight-motor IDs 1-8 verified in prior Station inspection. IDs are not embedded in source Parquet.'})
    preparation = {'source': str(source), 'source_fingerprint': before,
                   'excluded_source_episodes': sorted(set(grouped) - set(order)), 'episode_mapping': mapping,
                   'selection': 'cheese tray left in sampled top-camera views' if left_only else 'both tray positions',
                   'train_episodes': list(range(len(train))),
                   'validation_episodes': list(range(len(train), len(order))),
                   'train_frames': sum(len(grouped[i]) for i in train),
                   'validation_frames': sum(len(grouped[i]) for i in validation),
                   'image_preprocessing_applied_to_dataset': True,
                   'dataset_fingerprint': dataset_fingerprint(destination)}
    if dataset_fingerprint(source) != before:
        raise RuntimeError('Original dataset changed during preparation')
    write_json(destination / 'dataset_preparation.json', preparation)
    loaded = LeRobotDataset(repo_id, root=destination)
    expected_frames = sum(item['frames'] for item in mapping)
    if len(loaded) != expected_frames or loaded.num_episodes != len(order):
        raise RuntimeError('Prepared counts differ from the agreed split')
    loaded[0]
    print(f'Ready: {destination}\n{len(order)} episodes, {expected_frames} frames; '
          f'train {preparation["train_frames"]} / validation {preparation["validation_frames"]}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--left-only', action='store_true', help='Keep left-side tray episodes; exclude mixed/right/idle episodes')
    args = parser.parse_args()
    destination = args.destination or (LEFT_DESTINATION if args.left_only else DESTINATION)
    prepare(args.source.resolve(), destination.resolve(), args.left_only)
