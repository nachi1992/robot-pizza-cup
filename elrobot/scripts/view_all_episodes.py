#!/usr/bin/env python3
"""Open every LeRobot episode as a separate local Rerun recording."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--web-port', type=int, default=9090)
    parser.add_argument('--grpc-port', type=int, default=9876)
    parser.add_argument('--output-dir', type=Path, help='New folder for the separate local .rrd files')
    args = parser.parse_args()

    import rerun as rr
    import rerun.blueprint as rrb
    from torch.utils.data import DataLoader
    from lerobot.datasets import LeRobotDataset
    from lerobot.scripts.lerobot_dataset_viz import get_feature_names, to_hwc_uint8_numpy

    dataset = LeRobotDataset(args.repo_id, root=args.root)
    manifest_path = args.root / 'station_export_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    episodes = {p['episode_index']: p for p in manifest.get('episodes', [])}
    views = [rrb.Spatial2DView(origin=key, name=key) for key in dataset.meta.camera_keys]
    views.append(rrb.TextDocumentView(origin='episode', name='Current episode'))
    for origin, key in [('state', 'observation.state'), ('action', 'action')]:
        if key in dataset.features:
            views.append(rrb.TimeSeriesView(origin=origin, name=origin,
                         overrides={origin: rr.SeriesLines(names=get_feature_names(dataset, key))}))
    blueprint = rrb.Blueprint(rrb.Grid(*views, grid_columns=3),
                             rrb.TimePanel(expanded=True, timeline='timestamp', play_state='paused'))
    output = args.output_dir or args.root.parent / (
        args.root.name + '-rerun-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    recordings, sources, counts = {}, [], {}
    for ep in range(dataset.num_episodes):
        meta = episodes.get(ep, {})
        original = meta.get('original_episode_index', ep)
        recording_id = str(uuid.uuid4())
        rec = rr.RecordingStream(args.repo_id, recording_id=recording_id)
        path = output / f'episode-{ep:02d}-original-{original:02d}.rrd'
        rec.save(path)
        rec.send_recording_name(f'Episode {ep:02d} (original {original:02d})')
        rec.send_blueprint(blueprint)
        rec.log('episode', rr.TextDocument(
            f'Episode: {ep}\nOriginal episode: {original}\n'
            f'Recording: {meta.get("name", "")}\nFrames: {meta.get("frames", "")}'), static=True)
        recordings[ep] = rec
        counts[ep] = 0
        sources.append(dict(episode_index=ep, original_episode_index=original,
                            recording_id=recording_id, path=str(path), name=meta.get('name')))
    print(f'Writing {dataset.num_episodes} separate recordings, {len(dataset)} frames total.', flush=True)
    count = 0
    for batch in DataLoader(dataset, batch_size=32, num_workers=0):
        for i in range(len(batch['index'])):
            ep = int(batch['episode_index'][i].item())
            rec = recordings[ep]
            rec.set_time('frame_index', sequence=counts[ep])
            rec.set_time('timestamp', duration=float(batch['timestamp'][i].item()))
            if counts[ep] == 0:
                meta = episodes.get(ep, {})
                original = meta.get('original_episode_index', ep)
                text = (f'Episode {ep}\n'
                        f'Original episode: {original}\n'
                        f'Task: {batch["task"][i]}\n'
                        f'Recording: {meta.get("name", "")}\n'
                        f'Frames: {meta.get("frames", "")}\n'
                        'This source contains this episode only.')
                rec.log('episode', rr.TextDocument(text), static=True)
                print(f'Writing episode {ep} (original {original}) to its own source.', flush=True)
            for key in dataset.meta.camera_keys:
                rec.log(key, rr.Image(to_hwc_uint8_numpy(batch[key][i])).compress())
            for origin, key in [('state', 'observation.state'), ('action', 'action')]:
                if key in batch:
                    rec.log(origin, rr.Scalars(batch[key][i].numpy()))
            counts[ep] += 1
            count += 1
    if count != len(dataset) or any(n == 0 for n in counts.values()):
        raise RuntimeError('Viewer did not log the expected frames and episodes')
    for source in sources:
        ep = source['episode_index']
        recordings[ep].flush()
        recordings[ep].disconnect()
        source['frames'] = counts[ep]
        if episodes and counts[ep] != episodes[ep]['frames']:
            raise RuntimeError(f'Episode {ep} frame count differs from the manifest')
        if not Path(source['path']).stat().st_size:
            raise RuntimeError(f'Episode {ep} recording is empty')
    (output / 'sources.json').write_text(json.dumps(sources, indent=2) + '\n')
    print(f'READY: {len(sources)} separate local recordings, {count} frames; files: {output}', flush=True)
    subprocess.run([str(Path(sys.executable).with_name('rerun')), '--serve-web',
                    '--web-viewer-port', str(args.web_port), '--port', str(args.grpc_port),
                    *[s['path'] for s in sources]], check=True)


if __name__ == '__main__':
    main()
