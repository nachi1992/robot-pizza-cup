#!/usr/bin/env python3
"""Create explicit, checksummed Devlab or checkpoint archives. Never uploads."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

from smolvla_common import DEFAULT_DATASET, REPO, dataset_contract

TRAINING_FILES = [
    'scripts/train_smolvla.sh', 'scripts/train_smolvla.py', 'scripts/train_smolvla_entry.py',
    'scripts/smolvla_common.py', 'scripts/act_image_preprocessing.py',
    'scripts/bootstrap_smolvla.sh', 'scripts/package_smolvla.py',
    'config/smolvla-requirements.lock', 'config/smolvla-base-config.json', 'docs/smolvla-pipeline.md',
]


def create_archive(output, files, prefix):
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    checksums=[]
    for path,name in files:
        digest=hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
        checksums.append(f'{digest.hexdigest()}  {name}')
    checksum_bytes=('\n'.join(checksums)+'\n').encode()
    # Exclusive creation prevents replacing an existing archive.
    with output.open('xb') as handle,tarfile.open(fileobj=handle,mode='w:gz',compresslevel=1) as archive:
        for path,name in files:
            if path.is_symlink():raise ValueError(f'Refusing symlink: {path}')
            archive.add(path,arcname=prefix+'/'+name,recursive=False)
        item=tarfile.TarInfo(prefix+'/SHA256SUMS');item.size=len(checksum_bytes)
        archive.addfile(item,io.BytesIO(checksum_bytes))
    print(f'Archive: {output} ({output.stat().st_size:,} bytes, {len(files)} files)')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind',choices=['devlab','checkpoint'])
    parser.add_argument('--dataset-root',type=Path,default=DEFAULT_DATASET)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--include-training-state',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.kind=='devlab':
        dataset_contract(args.dataset_root)
        files=[(REPO/'elrobot'/f,'elrobot/'+f) for f in TRAINING_FILES]
        for path in sorted(args.dataset_root.rglob('*')):
            if not path.is_file() or '.cache' in path.parts:continue
            relative=path.relative_to(args.dataset_root)
            if relative.parts[0] not in ('data','meta') and relative.suffix!='.json':continue
            files.append((path,'data/'+args.dataset_root.name+'/'+str(relative)))
        create_archive(args.output.resolve(),files,'pizza-smolvla')
    else:
        if args.checkpoint is None:parser.error('--checkpoint is required')
        cp=args.checkpoint.resolve()
        model=cp/'pretrained_model' if (cp/'pretrained_model').is_dir() else cp
        for name in ('config.json','model.safetensors','policy_preprocessor.json','policy_postprocessor.json',
                     'station_contract.json','split_and_normalization.json','smolvla_run.json',
                     'image_preprocessing.json','vlm_assets/tokenizer.json'):
            if not (model/name).is_file():raise FileNotFoundError(model/name)
        files=[(p,'pretrained_model/'+str(p.relative_to(model))) for p in sorted(model.rglob('*'))
               if p.is_file() and '.cache' not in p.parts]
        if args.include_training_state:
            state=model.parent/'training_state'
            for name in ('training_step.json','optimizer_state.safetensors','rng_state.safetensors','scheduler_state.json'):
                if not (state/name).is_file():raise FileNotFoundError(state/name)
            files.extend((p,'training_state/'+str(p.relative_to(state))) for p in sorted(state.rglob('*')) if p.is_file())
        create_archive(args.output.resolve(),files,'smolvla-cheese-checkpoint')


if __name__=='__main__':main()
