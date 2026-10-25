#!/usr/bin/env python3
"""Portable SmolVLA training launcher. No hardware imports or automatic uploads."""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from smolvla_common import (BASE_REPO, BASE_REVISION, VLM_REPO, VLM_REVISION, DEFAULT_DATASET,
                           RENAME_MAP, REPO, check_version, dataset_contract, resolve_assets)


def configuration(args, contract, base, vlm):
    from lerobot.configs.train import TrainPipelineConfig
    # Merge the pinned architecture explicitly. Native CLI policy.path loading
    # replaces a policy dict and can silently discard YAML-only overrides.
    features = {'observation.state': {'type': 'STATE', 'shape': [8]}, **{
        value: {'type': 'VISUAL', 'shape': [3, 224, 224]} for value in RENAME_MAP.values()}}
    # Native splitting holds out ceil(N * fraction) episodes. Use the middle
    # of that interval to avoid floating-point rounding adding an episode.
    eval_fraction = (len(contract['validation_episodes']) - .5) / (len(contract['train_episodes']) + len(contract['validation_episodes']))
    cfg = dict(dataset=dict(repo_id='local/'+args.dataset_root.name, root=str(args.dataset_root),
                            eval_split=eval_fraction, image_transforms=dict(enable=False), use_imagenet_stats=False),
               policy=dict(type='smolvla', device=args.device, push_to_hub=False, use_amp=False,
                           input_features=features, output_features={'action': {'type': 'ACTION', 'shape': [8]}},
                           freeze_vision_encoder=True, train_expert_only=True, train_state_proj=True,
                           empty_cameras=1, adapt_to_pi_aloha=False, use_delta_joint_actions_aloha=False,
                           chunk_size=50, n_action_steps=5, num_steps=10,
                           optimizer_lr=args.learning_rate, scheduler_warmup_steps=min(1000,max(1,args.steps//20)),
                           scheduler_decay_steps=args.steps, load_vlm_weights=False, vlm_model_name=str(vlm)),
               rename_map=RENAME_MAP, steps=args.steps, batch_size=args.batch_size,
               num_workers=args.num_workers, prefetch_factor=2, seed=42, log_freq=20,
               eval_steps=min(1000,args.steps), max_eval_samples=0, env_eval_freq=0,
               save_freq=min(1000,args.steps), save_checkpoint_to_hub=False,
               wandb=dict(enable=False), job=dict(target='local'),
               output_dir=str(args.output_dir), job_name='smolvla-cheese')
    pinned=json.loads((Path(__file__).resolve().parents[1]/'config/smolvla-base-config.json').read_text())
    pinned.update(cfg['policy'],pretrained_path=str(base))
    cfg['policy']=pinned
    # Importing the class registers this choice for config decoding, without
    # constructing the model or downloading weights.
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    import draccus
    with tempfile.TemporaryDirectory(prefix='smolvla-config-') as folder:
        path = Path(folder) / 'train.json'
        path.write_text(json.dumps(cfg))
        parsed = draccus.parse(TrainPipelineConfig, config_path=path, args=[])
        assert parsed.policy.input_features['observation.state'].shape == (8,)
        assert parsed.rename_map == RENAME_MAP
        parsed.validate()
    return cfg


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['local','devlab'],default='local')
    parser.add_argument('--dataset-root',type=Path,default=DEFAULT_DATASET)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--steps',type=int,default=20000,help='Cumulative target on resume')
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--learning-rate',type=float,default=1e-4,
                        help='Peak learning rate for fresh runs; resume restores saved optimizer settings')
    parser.add_argument('--num-workers',type=int,default=2)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--asset-dir',type=Path)
    parser.add_argument('--download-assets',action='store_true',help='Fetch immutable base weights and tokenizer; do not train')
    parser.add_argument('--resume',type=Path,help='Checkpoint directory containing pretrained_model/ and training_state/')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    check_version()
    if args.steps<1 or args.batch_size<1 or args.num_workers<0:
        parser.error('Steps/batch must be positive; workers must be nonnegative')
    if not math.isfinite(args.learning_rate) or args.learning_rate<=0:
        parser.error('Learning rate must be finite and positive')
    if args.download_assets and args.dry_run:
        parser.error('--download-assets and --dry-run are mutually exclusive')
    args.dataset_root=args.dataset_root.resolve()
    args.output_dir=(args.output_dir or REPO/'outputs/train'/datetime.now().strftime('smolvla-cheese-%Y%m%d-%H%M%S')).resolve()
    if args.profile=='devlab':
        workspace=os.environ.get('SMOLVLA_WORKSPACE')
        if not workspace:
            parser.error('Set SMOLVLA_WORKSPACE to your persistent Devlab workspace')
        for path in (args.dataset_root,args.output_dir):
            if not path.is_relative_to(Path(workspace).resolve()):
                parser.error(f'{path} is outside SMOLVLA_WORKSPACE')
    contract=dataset_contract(args.dataset_root)
    if args.download_assets:
        base,vlm=resolve_assets(args.asset_dir,download=True)
        print(f'Pinned assets ready: {base}\n{vlm}');return
    if args.resume:
        cp=args.resume.resolve(); saved=cp/'pretrained_model'
        required=['training_step.json','optimizer_state.safetensors','optimizer_param_groups.json','rng_state.safetensors','scheduler_state.json']
        for name in required:
            if not (cp/'training_state'/name).is_file():raise FileNotFoundError(cp/'training_state'/name)
        old=json.loads((saved/'station_contract.json').read_text())
        if old!=contract:raise ValueError('Resume dataset/split/calibration/preprocessing differs')
        step=json.loads((cp/'training_state/training_step.json').read_text())['step']
        config=json.loads((saved/'train_config.json').read_text())
        original=json.loads((saved/'smolvla_run.json').read_text())
        if not step<args.steps or args.steps!=original['target_steps']:
            raise ValueError('Resume --steps must equal the original target and exceed the saved step; changing it rebuilds a different LR schedule')
        config.update(steps=args.steps,output_dir=str(args.output_dir),resume=True)
        config['dataset']['root']=str(args.dataset_root)
        config['dataset']['repo_id']='local/'+args.dataset_root.name
        config['policy']['device']=args.device
        config['policy']['vlm_model_name']=str(saved/'vlm_assets')
        base=saved;vlm=saved/'vlm_assets';run=original
        print(f'Resume step {step} -> {args.steps}; optimizer, scheduler and saved processors preserved.')
    else:
        from smolvla_common import asset_directory
        assets=asset_directory(args.asset_dir)
        base,vlm=assets/BASE_REVISION,assets/VLM_REVISION
        config=configuration(args,contract,base,vlm)
        run=dict(base_repo=BASE_REPO,base_revision=BASE_REVISION,vlm_repo=VLM_REPO,vlm_revision=VLM_REVISION,
                 target_steps=args.steps,batch_size=args.batch_size,learning_rate=args.learning_rate,profile=args.profile,seed=42,
                 dataset_fingerprint=contract['dataset_fingerprint'])
    if not args.resume and args.output_dir.exists():
        raise FileExistsError(f'Output exists: {args.output_dir}')
    if args.num_workers==0:
        config['persistent_workers']=False
    print(json.dumps(dict(profile=args.profile,dataset=str(args.dataset_root),output=str(args.output_dir),
                          train_episodes=len(contract['train_episodes']),validation_episodes=len(contract['validation_episodes']),train_frames=contract['train_frames'],
                          validation_frames=contract['validation_frames'],config=config),indent=2,default=str))
    if args.dry_run:
        print('DRY RUN: no assets downloaded, trainer started, or files written.');return
    import torch
    if args.device=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        print('GPU:',torch.cuda.get_device_name(0))
    if not args.resume:
        resolve_assets(args.asset_dir)
    elif not (vlm/'config.json').is_file():raise FileNotFoundError('Resume checkpoint is missing vlm_assets/')
    # Config/run metadata outside output_dir: native fresh training refuses an
    # existing output directory. The entrypoint copies these into the run later.
    args.output_dir.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='smolvla-launch-') as folder:
        folder=Path(folder)
        cfg_path=folder/'train.json';cfg_path.write_text(json.dumps(config))
        run_path=folder/'run.json';run_path.write_text(json.dumps(run))
        env=dict(os.environ,SMOLVLA_RUN_METADATA=str(run_path),SMOLVLA_VLM_ASSETS=str(vlm))
        command=[sys.executable,'-u',str(Path(__file__).with_name('train_smolvla_entry.py')),f'--config_path={cfg_path}']
        if args.resume:env['SMOLVLA_RESUME_CHECKPOINT']=str(cp)
        log=args.output_dir.with_suffix('.log')
        if log.exists():raise FileExistsError(log)
        with log.open('w') as handle:
            process=subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            try:
                for line in process.stdout:
                    print(line,end='',flush=True);handle.write(line);handle.flush()
                status=process.wait()
            except KeyboardInterrupt:
                import signal
                process.send_signal(signal.SIGINT);status=process.wait()
        print('Training exit status:',status,'Log:',log)
        raise SystemExit(status)


if __name__=='__main__':main()
