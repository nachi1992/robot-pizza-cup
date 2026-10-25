"""Deterministic dataset preparation and matching live-camera preprocessing."""
import hashlib
import json
from pathlib import Path

from PIL import Image


def load_preprocessing(root):
    path = Path(root) / 'image_preprocessing.json'
    if not path.exists():
        return {}
    spec = json.loads(path.read_text())
    if (spec.get('version') != 1 or set(spec.get('cameras', {})) != {'top'}
            or spec['cameras']['top'] != {
                'input_size': [224, 224], 'crop_left_pixels': 34,
                'output_size': [224, 224], 'resample': 'bilinear'}):
        raise ValueError(f'Unsupported camera preprocessing: {path}')
    return spec


def preprocess_camera(image, name, spec):
    camera = spec.get('cameras', {}).get(name)
    if camera is None:
        return image
    if list(image.size) != camera['input_size']:
        raise ValueError(f'{name}: crop requires {camera["input_size"]}, got {image.size}')
    return image.crop((camera['crop_left_pixels'], 0, image.width, image.height)).resize(
        tuple(camera['output_size']), Image.Resampling.BILINEAR)


def dataset_fingerprint(root):
    """Hash actual dataset content and preprocessing, excluding reports/caches."""
    root = Path(root)
    paths = sorted([*root.glob('data/**/*.parquet'), *root.glob('meta/**/*')])
    paths += [root / 'image_preprocessing.json']
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            continue
        digest.update(str(path.relative_to(root)).encode())
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()
