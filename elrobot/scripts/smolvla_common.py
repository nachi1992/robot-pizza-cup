"""Shared portable SmolVLA data/checkpoint contract (no hardware imports)."""
import importlib.metadata
import json
from pathlib import Path

from act_image_preprocessing import dataset_fingerprint, load_preprocessing

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO / 'data/lerobot-cheese-combined-crop15-smolvla-v1'
BASE_REPO = 'lerobot/smolvla_base'
BASE_REVISION = '5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6'
VLM_REPO = 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct'
VLM_REVISION = '7b375e1b73b11138ff12fe22c8f2822d8fe03467'
RENAME_MAP = {'observation.images.front': 'observation.images.camera1',
              'observation.images.top': 'observation.images.camera2'}
TASK = 'place cheese on pizza'
CONTRACT_FILES = ('image_preprocessing.json', 'split_and_normalization.json',
                  'smolvla_run.json', 'station_contract.json')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def check_version():
    if importlib.metadata.version('lerobot') != '0.6.1':
        raise RuntimeError('This pipeline requires LeRobot 0.6.1; use the isolated SmolVLA environment')


def dataset_contract(root, verify_fingerprint=True):
    root = Path(root).resolve()
    info = json.loads((root / 'meta/info.json').read_text())
    prep = json.loads((root / 'dataset_preparation.json').read_text())
    calibration = json.loads((root / 'meta/station_calibration.json').read_text())
    crop = load_preprocessing(root)
    for key in ('action', 'observation.state'):
        if info['features'][key]['shape'] != [8]:
            raise ValueError(f'{key}: expected eight joint slots')
    for name in ('front', 'top'):
        if info['features'][f'observation.images.{name}']['shape'] != [224, 224, 3]:
            raise ValueError(f'{name}: expected 224x224 RGB')
    train, validation = prep['train_episodes'], prep['validation_episodes']
    if (info['fps'] != 10
            or info['total_tasks'] != 1 or not prep['image_preprocessing_applied_to_dataset']
            or not train or not validation
            or train != list(range(len(train)))
            or validation != list(range(len(train), info['total_episodes']))
            or prep['train_frames'] + prep['validation_frames'] != info['total_frames']):
        raise ValueError('Dataset must have a recorded contiguous train/validation episode split, one task and 10 Hz')
    if calibration['motor_ids'] != list(range(1, 9)) or len(calibration['calibration_ranges']) != 8:
        raise ValueError('Calibration must specify eight sorted motor IDs')
    fingerprint = dataset_fingerprint(root) if verify_fingerprint else prep['dataset_fingerprint']
    if fingerprint != prep['dataset_fingerprint']:
        raise ValueError('Dataset fingerprint changed after preparation')
    return dict(dataset_fingerprint=fingerprint, train_episodes=prep['train_episodes'],
                validation_episodes=prep['validation_episodes'],
                train_frames=prep['train_frames'], validation_frames=prep['validation_frames'],
                motor_ids=calibration['motor_ids'], calibration_ranges=calibration['calibration_ranges'],
                camera_names=['front', 'top'], camera_rename_map=RENAME_MAP,
                image_preprocessing=crop, fps=10, task=TASK)


def asset_directory(cache=None):
    import os
    default = os.environ.get('HF_HOME', str(Path.home() / '.cache/huggingface'))
    return Path(cache or Path(default) / 'pizza-smolvla-assets').expanduser().resolve()


def resolve_assets(cache=None, download=False):
    """Pin both policy weights and the tokenizer/model architecture assets."""
    assets = asset_directory(cache)
    base, vlm = assets / BASE_REVISION, assets / VLM_REVISION
    if download:
        from huggingface_hub import snapshot_download
        snapshot_download(BASE_REPO, revision=BASE_REVISION, local_dir=base,
                          allow_patterns=['*.json', '*.safetensors'])
        # The policy checkpoint already includes VLM weights. Only architecture
        # and processor/tokenizer assets are needed from this second repository.
        snapshot_download(VLM_REPO, revision=VLM_REVISION, local_dir=vlm,
                          allow_patterns=['*.json', '*.txt', '*.model', '*.jinja', '*.tiktoken'])
    for path in (base / 'config.json', base / 'model.safetensors', vlm / 'config.json',
                 vlm / 'tokenizer.json', vlm / 'tokenizer_config.json'):
        if not path.is_file():
            raise FileNotFoundError(f'Missing pinned asset {path}; run training launcher --download-assets')
    return base, vlm
