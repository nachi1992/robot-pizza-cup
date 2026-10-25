#!/usr/bin/env python3
"""Native LeRobot trainer with a checked split and training-only vector stats."""
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import shutil
import sys

from smolvla_common import check_version, dataset_contract, write_json, RENAME_MAP


def main():
    check_version()
    import numpy as np
    from lerobot.scripts import lerobot_train as native
    native_factory, native_save = native.make_train_eval_datasets, native.save_checkpoint
    native_processors = native.make_pre_post_processors
    run = json.loads(Path(os.environ['SMOLVLA_RUN_METADATA']).read_text())
    vlm = Path(os.environ['SMOLVLA_VLM_ASSETS']).resolve()

    def training_datasets(cfg):
        contract = dataset_contract(cfg.dataset.root)
        train, validation = native_factory(cfg)
        if (validation is None or list(train.episodes) != contract['train_episodes']
                or list(validation.episodes) != contract['validation_episodes']
                or len(train) != contract['train_frames'] or len(validation) != contract['validation_frames']):
            raise ValueError('Native split differs from the recorded episode split')
        if cfg.resume:
            saved = Path(cfg.checkpoint_path) / 'pretrained_model'
            if json.loads((saved / 'station_contract.json').read_text()) != contract:
                raise ValueError('Resume contract differs from checkpoint')
            report = json.loads((saved / 'split_and_normalization.json').read_text())
            # The native trainer loads saved processors on resume without
            # overriding normalization stats. Copy provenance, do not refit.
        else:
            vectors = train.hf_dataset.select_columns(['observation.state', 'action']).with_format('numpy')[:]
            fitted = {}
            for key in ('observation.state', 'action'):
                values = np.asarray(vectors[key], dtype=np.float32)
                if values.shape != (len(train), 8) or not np.isfinite(values).all():
                    raise ValueError(f'Invalid training vectors: {key}')
                stats = dict(min=values.min(0), max=values.max(0), mean=values.mean(0), std=values.std(0),
                             q01=np.quantile(values,.01,axis=0).astype(np.float32),
                             q99=np.quantile(values,.99,axis=0).astype(np.float32),
                             count=np.asarray([len(values)], dtype=np.int64))
                train.meta.stats[key] = stats
                validation.meta.stats[key] = stats
                fitted[key] = {k: v.tolist() for k,v in stats.items()}
            report = dict(contract, stats=fitted, vector_normalization='Training frames only; SmolVLA visual normalization is identity.')
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(cfg.output_dir / 'split_and_normalization.json', report)
        write_json(cfg.output_dir / 'station_contract.json', contract)
        write_json(cfg.output_dir / 'smolvla_run.json', dict(run, environment={
            p: importlib.metadata.version(p) for p in ('lerobot','torch','torchvision','transformers','accelerate')}))
        logging.info('Verified %s train / %s validation episodes, %s / %s frames.',
                     len(train.episodes),len(validation.episodes),len(train),len(validation))
        return train, validation

    def processors(*args, **kwargs):
        overrides = kwargs.setdefault('preprocessor_overrides', {})
        overrides['tokenizer_processor'] = {'tokenizer_name': str(vlm)}
        overrides['rename_observations_processor'] = {'rename_map': RENAME_MAP}
        return native_processors(*args, **kwargs)

    def save_checkpoint(**kwargs):
        native_save(**kwargs)
        cfg=kwargs['cfg']; destination=Path(kwargs['checkpoint_dir'])/'pretrained_model'
        for name in ('split_and_normalization.json','station_contract.json','smolvla_run.json'):
            shutil.copyfile(Path(cfg.output_dir)/name,destination/name)
        shutil.copyfile(Path(cfg.dataset.root)/'image_preprocessing.json',destination/'image_preprocessing.json')
        # These contain no weights and are small. Bundle them with every model
        # so a different machine can load tokenizer/architecture offline.
        shutil.copytree(vlm,destination/'vlm_assets',ignore=shutil.ignore_patterns('.cache'),dirs_exist_ok=True)

    native.make_train_eval_datasets=training_datasets
    native.make_pre_post_processors=processors
    native.save_checkpoint=save_checkpoint
    # Construct the complete config once, avoiding the native CLI's partial
    # ChoiceType overrides replacing the pretrained policy configuration.
    import argparse
    import draccus
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_path',type=Path,required=True)
    args=parser.parse_args()
    with draccus.config_type('json'):
        cfg=draccus.parse(TrainPipelineConfig,config_path=args.config_path,args=[])
    resume=os.environ.get('SMOLVLA_RESUME_CHECKPOINT')
    sys.argv=[sys.argv[0]]
    if resume:
        sys.argv.append(f'--config_path={Path(resume)/"pretrained_model/train_config.json"}')
    native.train(cfg)


if __name__=='__main__':main()
