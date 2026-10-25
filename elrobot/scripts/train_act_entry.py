#!/usr/bin/env python3
"""Use LeRobot's native trainer with normalization fitted to training episodes."""
import importlib.metadata
import json
import logging
from pathlib import Path

from act_image_preprocessing import dataset_fingerprint, load_preprocessing


def main():
    if importlib.metadata.version('lerobot') != '0.6.1':
        raise SystemExit('This launcher targets the installed LeRobot 0.6.1 training API.')

    import numpy as np
    from lerobot.scripts import lerobot_train

    native_factory = lerobot_train.make_train_eval_datasets
    native_save = lerobot_train.save_checkpoint

    def training_datasets(cfg):
        train, validation = native_factory(cfg)
        if validation is None:
            raise ValueError('This local experiment requires dataset.eval_split > 0.')
        fingerprint = dataset_fingerprint(cfg.dataset.root)
        preprocessing = load_preprocessing(cfg.dataset.root)
        if cfg.resume:
            saved = Path(cfg.checkpoint_path) / 'pretrained_model'
            saved_report = saved / 'split_and_normalization.json'
            # Older runs saved this report in the run directory only.
            if not saved_report.exists():
                saved_report = Path(cfg.output_dir) / 'split_and_normalization.json'
            report = json.loads(saved_report.read_text())
            if (report['train_episodes'] != list(train.episodes)
                    or report['validation_episodes'] != list(validation.episodes)
                    or report['train_frames'] != len(train)
                    or report['validation_frames'] != len(validation)
                    or (report.get('dataset_fingerprint') is not None
                        and report['dataset_fingerprint'] != fingerprint)
                    or load_preprocessing(saved) != preprocessing):
                raise ValueError('Resume dataset, split or image preprocessing differs from the checkpoint')
            # Saved processors and optimizer state remain authoritative on resume.
            logging.info('Resume verified: preserving saved normalization and episode split.')
            return train, validation
        # The native episode split retains the dataset-wide metadata stats.
        # Fit state/action normalization on training frames only. Camera
        # normalization remains the native ImageNet preset for ResNet-18.
        vectors = train.hf_dataset.select_columns(['observation.state', 'action']).with_format('numpy')[:]
        fitted = {}
        for key in ['observation.state', 'action']:
            values = np.asarray(vectors[key], dtype=np.float32)
            if values.ndim != 2 or values.shape[1] != 8 or not np.isfinite(values).all():
                raise ValueError(f'{key}: expected finite eight-dimensional joint vectors')
            fitted[key] = {
                'min': values.min(axis=0), 'max': values.max(axis=0),
                'mean': values.mean(axis=0), 'std': values.std(axis=0),
                'q01': np.quantile(values, .01, axis=0).astype(np.float32),
                'q99': np.quantile(values, .99, axis=0).astype(np.float32),
                'count': np.asarray([len(values)], dtype=np.int64),
            }
            train.meta.stats[key] = fitted[key]
            validation.meta.stats[key] = fitted[key]
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        report = dict(
            train_episodes=list(train.episodes), validation_episodes=list(validation.episodes),
            train_frames=len(train), validation_frames=len(validation),
            vector_normalization='Training frames only; original dataset files are not modified.',
            dataset_fingerprint=fingerprint,
            image_preprocessing=preprocessing,
            stats={key: {stat: value.tolist() for stat, value in stats.items()}
                   for key, stats in fitted.items()},
        )
        (cfg.output_dir / 'split_and_normalization.json').write_text(json.dumps(report, indent=2) + '\n')
        logging.info('Train episodes: %s; validation episodes: %s; normalization uses training frames only.',
                     report['train_episodes'], report['validation_episodes'])
        return train, validation

    def save_checkpoint(**kwargs):
        native_save(**kwargs)
        cfg = kwargs['cfg']
        folder = Path(kwargs['checkpoint_dir']) / 'pretrained_model'
        # Keep the crop contract and normalization/split record with every model.
        (folder / 'split_and_normalization.json').write_text(
            (Path(cfg.output_dir) / 'split_and_normalization.json').read_text())
        spec = load_preprocessing(cfg.dataset.root)
        if spec:
            (folder / 'image_preprocessing.json').write_text(json.dumps(spec, indent=2) + '\n')

    # Adapt the imported factory at the trainer's extension point without
    # modifying the installed LeRobot package or replacing its training loop.
    lerobot_train.make_train_eval_datasets = training_datasets
    lerobot_train.save_checkpoint = save_checkpoint
    lerobot_train.main()


if __name__ == '__main__':
    main()
