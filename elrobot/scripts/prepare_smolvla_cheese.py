#!/usr/bin/env python3
"""Merge prepared cheese takes without recropping; no Station connection."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import shutil

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from lerobot.datasets import LeRobotDataset

from act_image_preprocessing import dataset_fingerprint, load_preprocessing

REPO = Path(__file__).resolve().parents[2]
SOURCES = [REPO / 'data/lerobot-cheese-remote-20261024-144902-train-v1',
           REPO / 'data/lerobot-cheese-from-144251-20261024-154424-crop15-train-v1']
DESTINATION = REPO / 'data/lerobot-cheese-combined-crop15-smolvla-v1'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def image_array(value):
    if not value.get('bytes'):
        raise ValueError('Expected embedded image bytes, not external image paths')
    with Image.open(io.BytesIO(value['bytes'])) as image:
        return np.asarray(image.convert('RGB'))


def episode_digest(rows):
    digest = hashlib.sha256()
    for row in rows:
        for key in ('observation.state', 'action', 'source.timestamp_ns'):
            dtype = np.int64 if key.endswith('_ns') else np.float32
            digest.update(np.asarray(row[key], dtype=dtype).tobytes())
        for name in ('front', 'top'):
            digest.update(image_array(row[f'observation.images.{name}']).tobytes())
    return digest.hexdigest()


def inspect_sources(sources, validation_count=5, seed=42):
    takes, fingerprints, common = {}, {}, None
    for source in sources:
        info = json.loads((source / 'meta/info.json').read_text())
        preparation = json.loads((source / 'dataset_preparation.json').read_text())
        calibration = json.loads((source / 'meta/station_calibration.json').read_text())
        spec = load_preprocessing(source)
        signature = (info['fps'], info['features'], spec,
                     calibration['motor_ids'], calibration['calibration_ranges'])
        if common is not None and signature != common:
            raise ValueError('Sources differ in features, FPS, crop or calibration')
        common = signature
        if info['fps'] != 10 or not preparation.get('image_preprocessing_applied_to_dataset'):
            raise ValueError('Sources must be prepared, already-cropped 10 Hz datasets')
        if calibration['motor_ids'] != list(range(1, 9)):
            raise ValueError('Expected motor IDs 1 through 8')
        fingerprints[str(source)] = dataset_fingerprint(source)
        manifest = json.loads((source / 'station_export_manifest.json').read_text())
        episode_meta = {e['episode_index']: e for e in manifest['episodes']}
        grouped = {}
        for file in sorted((source / 'data').rglob('*.parquet')):
            for row in pq.read_table(file).to_pylist():
                grouped.setdefault(row['episode_index'], []).append(row)
        for mapping in preparation['episode_mapping']:
            index = mapping['episode_index']
            rows = sorted(grouped[index], key=lambda row: row['frame_index'])
            if len(rows) != mapping['frames']:
                raise ValueError(f'Episode count changed: {source}:{index}')
            meta = episode_meta[index]
            tag = meta['name']
            digest = episode_digest(rows)
            origin = dict(dataset=str(source), episode_index=index,
                          prior_split=mapping['split'], content_sha256=digest)
            if tag in takes:
                if takes[tag]['digest'] != digest:
                    raise ValueError(f'Overlapping take has different content: {tag}')
                takes[tag]['origins'].append(origin)
            else:
                takes[tag] = dict(name=tag, rows=rows, meta=meta, digest=digest,
                                  origins=[origin], source_group=len(fingerprints) - 1)
    if validation_count != 5 or len(takes) != 40:
        raise ValueError(f'This v1 contract requires 40 unique takes and five validation episodes; got {len(takes)}')
    # Ten takes occur only in the first source, thirty occur in the second.
    groups = [sorted(t for t, v in takes.items() if len(v['origins']) == 1 and v['source_group'] == 0),
              sorted(t for t, v in takes.items() if any(o['dataset'] == str(sources[1]) for o in v['origins']))]
    if [len(g) for g in groups] != [10, 30]:
        raise ValueError('Unexpected source groups; inspect split instead of guessing')
    rng = random.Random(seed)
    validation = set(rng.sample(groups[0], 1) + rng.sample(groups[1], 4))
    order = sorted(set(takes) - validation) + sorted(validation)
    return takes, order, validation, fingerprints, common


def prepare(args):
    sources = [p.resolve() for p in args.sources]
    destination = args.destination.resolve()
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    takes, order, validation, fingerprints, common = inspect_sources(sources, seed=args.seed)
    summary = dict(episodes=len(order), frames=sum(len(takes[t]['rows']) for t in order),
                   train_episodes=35, validation_episodes=5,
                   train_frames=sum(len(takes[t]['rows']) for t in order if t not in validation),
                   validation_frames=sum(len(takes[t]['rows']) for t in validation),
                   duplicate_takes=sum(len(v['origins']) - 1 for v in takes.values()),
                   validation_tags=sorted(validation))
    print(json.dumps(summary, indent=2), flush=True)
    if args.dry_run:
        return
    info = json.loads((sources[0] / 'meta/info.json').read_text())
    features = {k: v for k, v in info['features'].items()
                if k.startswith('observation.') or k in ('action', 'source.timestamp', 'source.timestamp_ns')}
    repo_id = 'local/cheese-combined-crop15-smolvla-v1'
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset.create(repo_id, root=destination, fps=10, features=features,
                                   robot_type='elrobot_station', use_videos=False, image_writer_threads=2)
    mapping, episodes = [], []
    try:
        for index, tag in enumerate(order):
            take = takes[tag]
            for row in take['rows']:
                frame = {k: np.asarray(row[k], dtype=np.float32).reshape(features[k]['shape'])
                         for k in ('observation.state', 'action', 'source.timestamp')}
                frame['source.timestamp_ns'] = np.asarray(row['source.timestamp_ns'], dtype=np.int64).reshape(1)
                frame['task'] = 'place cheese on pizza'
                for camera in ('front', 'top'):
                    pixels = image_array(row[f'observation.images.{camera}'])
                    if pixels.shape != (224, 224, 3):
                        raise ValueError(f'{tag}: image shape changed')
                    frame[f'observation.images.{camera}'] = pixels
                dataset.add_frame(frame)
            dataset.save_episode()
            split = 'validation' if tag in validation else 'train'
            mapping.append(dict(episode_index=index, source_tag=tag, split=split,
                                frames=len(take['rows']), origins=take['origins']))
            episodes.append(dict(take['meta'], episode_index=index))
            print(f'Wrote episode {index}: {tag} ({split})', flush=True)
    finally:
        dataset.finalize()
    write_json(destination / 'image_preprocessing.json', common[2])
    shutil.copyfile(sources[0] / 'meta/station_calibration.json', destination / 'meta/station_calibration.json')
    notes = json.loads((sources[0] / 'conversion_notes.json').read_text())
    notes.update(repo_id=repo_id, camera_names=['front', 'top'], uploaded=False,
                 training_sources=list(fingerprints), image_preprocessing_applied_to_dataset=True)
    write_json(destination / 'conversion_notes.json', notes)
    manifest = json.loads((sources[0] / 'station_export_manifest.json').read_text())
    manifest.update(episodes=episodes, frames=summary['frames'], training_sources=list(fingerprints))
    write_json(destination / 'station_export_manifest.json', manifest)
    preparation = dict(summary, source_fingerprints=fingerprints, seed=args.seed,
                       split_rule='One first-source-only and four second-source takes held out, seed 42; all duplicates share one split.',
                       episode_mapping=mapping, train_episodes=list(range(35)), validation_episodes=list(range(35, 40)),
                       image_preprocessing_applied_to_dataset=True)
    preparation['dataset_fingerprint'] = dataset_fingerprint(destination)
    write_json(destination / 'dataset_preparation.json', preparation)
    for source in sources:
        if dataset_fingerprint(source) != fingerprints[str(source)]:
            raise RuntimeError('Source changed during preparation')
    loaded = LeRobotDataset(repo_id, root=destination)
    if len(loaded) != summary['frames'] or loaded.num_episodes != 40:
        raise RuntimeError('Merged dataset counts differ')
    loaded[0]; loaded[len(loaded) - 1]
    print(f'Verified merged dataset: {destination}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', nargs=2, type=Path, default=SOURCES)
    parser.add_argument('--destination', type=Path, default=DESTINATION)
    parser.add_argument('--seed', type=int, default=42, choices=[42])
    parser.add_argument('--dry-run', action='store_true')
    prepare(parser.parse_args())
