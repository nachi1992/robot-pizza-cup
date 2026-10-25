#!/usr/bin/env python3
"""Prepare a single Station dataset without modifying the source."""
import argparse, io, json, random
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from lerobot.datasets import LeRobotDataset
from act_image_preprocessing import dataset_fingerprint, load_preprocessing, preprocess_camera
from prepare_act_cheese import SPEC
from smolvla_common import TASK, dataset_contract, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--validation-episodes',type=int,default=5)
    a=p.parse_args();source=a.source.resolve();dest=a.destination.resolve()
    if dest.exists():raise FileExistsError(f'Refusing overwrite: {dest}')
    info=json.loads((source/'meta/info.json').read_text());n=info['total_episodes']
    if not 0<a.validation_episodes<n:raise ValueError('Validation must leave training episodes')
    if info['fps']!=10 or info['total_tasks']!=1:raise ValueError('Expected one task at 10 Hz')
    tasks=pq.read_table(source/'meta/tasks.parquet').to_pylist()
    if len(tasks)!=1 or tasks[0]['task']!=TASK:raise ValueError(f'Expected task: {TASK}')
    for key in ('observation.state','action'):
        if info['features'][key]['shape']!=[8]:raise ValueError('Expected eight joints')
    before=dataset_fingerprint(source);notes=json.loads((source/'conversion_notes.json').read_text())
    if (source/'meta/station_calibration.json').exists():
        calibration=json.loads((source/'meta/station_calibration.json').read_text())
    else:
        raw=pq.read_table(Path(notes['source']),columns=['joints'])
        ranges={tuple((j['range_min'],j['range_max']) for j in row['joints']) for row in raw.to_pylist()}
        if len(ranges)!=1 or len(next(iter(ranges)))!=8:raise ValueError('Inconsistent calibration')
        calibration=dict(motor_ids=list(range(1,9)),calibration_ranges=next(iter(ranges)),source=notes['source'],source_frames_checked=len(raw),motor_id_order_basis='Station producer sorts the eight joint slots by motor ID, as in the existing pipeline.')
    prior_crop=load_preprocessing(source);crop=prior_crop or SPEC
    grouped={}
    for path in sorted((source/'data').rglob('*.parquet')):
        for row in pq.read_table(path).to_pylist():grouped.setdefault(row['episode_index'],[]).append(row)
    if sorted(grouped)!=list(range(n)) or sum(map(len,grouped.values()))!=info['total_frames']:raise ValueError('Metadata count mismatch')
    validation=set(random.Random(42).sample(range(n),a.validation_episodes))
    order=sorted(set(range(n))-validation)+sorted(validation)
    features={k:v for k,v in info['features'].items() if k.startswith('observation.') or k in ('action','source.timestamp','source.timestamp_ns')}
    dest.parent.mkdir(parents=True,exist_ok=True)
    ds=LeRobotDataset.create('local/'+dest.name,root=dest,fps=10,features=features,robot_type=info['robot_type'],use_videos=False,image_writer_threads=2)
    mapping=[]
    try:
        for new,original in enumerate(order):
            rows=sorted(grouped[original],key=lambda r:r['frame_index'])
            for row in rows:
                frame={'task':TASK}
                for key in ('observation.state','action','source.timestamp','source.timestamp_ns'):
                    frame[key]=np.asarray(row[key],dtype=np.int64 if key.endswith('_ns') else np.float32).reshape(features[key]['shape'])
                    if not np.isfinite(frame[key]).all():raise ValueError(f'Nonfinite {key}')
                for name in ('front','top'):
                    image=Image.open(io.BytesIO(row[f'observation.images.{name}']['bytes'])).convert('RGB')
                    if image.size!=(224,224):raise ValueError('Expected 224x224 images')
                    if not prior_crop:image=preprocess_camera(image,name,crop)
                    frame[f'observation.images.{name}']=np.asarray(image)
                ds.add_frame(frame)
            ds.save_episode()
            mapping.append(dict(episode_index=new,source_episode_index=original,split='validation' if original in validation else 'train',frames=len(rows)))
            print(f'Prepared {new+1}/{n}: source episode {original}',flush=True)
    finally:ds.finalize()
    write_json(dest/'image_preprocessing.json',crop)
    write_json(dest/'meta/station_calibration.json',calibration)
    write_json(dest/'conversion_notes.json',dict(notes,training_source=str(source),repo_id='local/'+dest.name))
    nt=n-len(validation)
    prep=dict(source=str(source),source_fingerprint=before,seed=42,episode_mapping=mapping,train_episodes=list(range(nt)),validation_episodes=list(range(nt,n)),train_frames=sum(m['frames'] for m in mapping[:nt]),validation_frames=sum(m['frames'] for m in mapping[nt:]),image_preprocessing_applied_to_dataset=True,dataset_fingerprint=dataset_fingerprint(dest))
    write_json(dest/'dataset_preparation.json',prep)
    if before!=dataset_fingerprint(source):raise RuntimeError('Source changed')
    dataset_contract(dest)
    print(f'Ready: {dest}\n{nt} train / {len(validation)} validation episodes.',flush=True)


if __name__=='__main__':main()
