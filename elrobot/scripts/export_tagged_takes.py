#!/usr/bin/env python3
"""Export exact Station tag pairs as episodes without movement-based filtering."""
import argparse
import asyncio
from bisect import bisect_right
import io
import json
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

import station_tag
from station_py import new_station_client
from target.gen_python.protobuf.station import inference
from target.gen_python.protobuf.drivers.inferences import normvla


def ptr(value):
    return value.to_bytes(8, 'little')


async def read_batch(client, queue, offset, limit):
    stream = client.read_from_offset(queue, offset=ptr(offset), limit=limit, step=1, buf_size=64)
    result = []
    while True:
        entry = await asyncio.wait_for(stream.data.get(), 20)
        if entry is None:
            break
        result.append(entry)
    if stream.err:
        raise RuntimeError(f'{queue}: {stream.err}')
    return result


async def snapshot(client, pointer):
    entries = await read_batch(client, 'inference-states', pointer, 1)
    if len(entries) != 1 or int.from_bytes(entries[0].ID.ID, 'little') != pointer:
        raise ValueError(f'Inference snapshot {pointer} is unavailable')
    return inference.InferenceRxReader(memoryview(bytes(entries[0].Data)))


def schema():
    joint = pa.struct([(name, pa.float32() if name.endswith('_norm') else pa.uint32())
                       for name in ['range_min', 'range_max', 'position', 'position_norm',
                                    'goal', 'goal_norm', 'current_ma', 'velocity']])
    return pa.schema([('episode_start_ns', pa.uint64()), ('global_frame_id', pa.binary()),
                      ('timestamp_ns_since_episode_start', pa.uint64()),
                      ('joints', pa.list_(joint)),
                      ('images', pa.list_(pa.struct([('jpeg', pa.binary())]))),
                      ('task', pa.string())])


async def export(args):
    inventory = json.loads(args.inventory.read_text())
    all_pairs = inventory['complete']
    exclusions = set(args.exclude_name)
    if exclusions - {p['name'] for p in all_pairs}:
        raise ValueError('An excluded name is missing from the inventory')
    pairs = sorted([p for p in all_pairs if p['name'] not in exclusions],
                   key=lambda p: p['start']['pointer'])
    if not pairs:
        raise ValueError('No complete selected tag pairs')
    for i, pair in enumerate(pairs):
        if pair['start']['pointer'] >= pair['end']['pointer']:
            raise ValueError(f'Invalid pair: {pair["name"]}')
        if i and pairs[i - 1]['end']['pointer'] > pair['start']['pointer']:
            raise ValueError('Selected tag ranges overlap')
    manifest_path = args.output.with_suffix('.manifest.json')
    previews = args.output.with_suffix('.previews')
    if any(p.exists() for p in [args.output, manifest_path, previews]):
        raise ValueError('Output exists; choose a new output path')
    client = await new_station_client(args.host, logging.getLogger('export-takes'))
    try:
        # Confirm inventory references against current active Station tags.
        tag_entries = await read_batch(client, 'inference-tags/rx', 0, 10001)
        if len(tag_entries) > 10000:
            raise ValueError('Tag history is too large for complete lookup')
        active = {}
        for entry in tag_entries:
            event = station_tag.tag_event(entry.Data)
            key = (bytes(event.get(12, b'')).decode(), int.from_bytes(event.get(11, b''), 'little'))
            if event.get(10, 0) == 0:
                active[key] = event
            else:
                active.pop(key, None)
        for pair in pairs:
            for side in ('start', 'end'):
                tag = pair[side]
                matches = [k for k in active if k[0] == tag['name']]
                if matches != [(tag['name'], tag['pointer'])]:
                    raise ValueError(f'Tag missing, changed or ambiguous: {tag["name"]}')
        starts = await asyncio.gather(*(snapshot(client, p['start']['pointer']) for p in pairs))
        ends = await asyncio.gather(*(snapshot(client, p['end']['pointer']) for p in pairs))
        refs = [e for e in starts[0].get_entries() if e.get_queue().endswith('/inference/normvla')]
        if len(refs) != 1:
            raise ValueError('Expected exactly one normvla frame queue at the first boundary')
        queue, cursor = refs[0].get_queue(), int.from_bytes(refs[0].get_ptr(), 'little')
        start_ids = [p['start']['pointer'] for p in pairs]
        grouped = [[] for _ in pairs]
        previous = -1
        finished = False
        while not finished:
            batch = await read_batch(client, queue, cursor, 256)
            if not batch:
                break
            for entry in batch:
                frame = normvla.FrameReader(memoryview(bytes(entry.Data)))
                fid = int.from_bytes(frame.get_global_frame_id(), 'little')
                if fid <= previous:
                    raise ValueError('Global frame IDs are not strictly increasing')
                previous = fid
                if fid >= pairs[-1]['end']['pointer']:
                    finished = True
                    break
                index = bisect_right(start_ids, fid) - 1
                if index >= 0 and fid < pairs[index]['end']['pointer']:
                    if len(frame.get_joints()) != 8 or len(frame.get_images()) != 2:
                        raise ValueError(f'Frame {fid}: expected eight joints and two cameras')
                    grouped[index].append(frame)
            cursor = int.from_bytes(batch[-1].ID.ID, 'little') + 1
            if len(batch) < 256:
                break
        if any(not frames for frames in grouped):
            raise ValueError('No saved valid frames in: ' + ', '.join(p['name'] for p, f in zip(pairs, grouped) if not f))
        print('Selected frames per episode:', [len(f) for f in grouped], flush=True)

        # normvla image order follows USB camera references in its source snapshot.
        # Recover that order for every frame and normalize it to the first frame.
        frames = [f for group in grouped for f in group]
        camera_orders = {}
        for offset in range(0, len(frames), 32):
            chunk = frames[offset:offset + 32]
            states = await asyncio.gather(*(snapshot(client, int.from_bytes(f.get_global_frame_id(), 'little')) for f in chunk))
            for frame, state in zip(chunk, states):
                fid = int.from_bytes(frame.get_global_frame_id(), 'little')
                cameras = [e.get_queue() for e in state.get_entries() if e.get_type() == 21]
                if len(cameras) != 2 or len(set(cameras)) != 2:
                    raise ValueError(f'Cannot resolve camera identity for frame {fid}')
                camera_orders[fid] = cameras
        reference_order = camera_orders[int.from_bytes(frames[0].get_global_frame_id(), 'little')]
        rows, episodes = [], []
        reordered = 0
        for episode_index, (pair, group, start, end) in enumerate(zip(pairs, grouped, starts, ends)):
            times, ids = [], []
            for frame in group:
                fid = int.from_bytes(frame.get_global_frame_id(), 'little')
                order = camera_orders[fid]
                if set(order) != set(reference_order):
                    raise ValueError('Camera identities change across the selected recording')
                reordered += order != reference_order
                images = frame.get_images()
                elapsed = frame.get_monotonic_stamp_ns() - start.get_monotonic_stamp_ns()
                if elapsed < 0 or (times and elapsed <= times[-1]):
                    raise ValueError(f'Invalid timestamp in {pair["name"]}')
                times.append(elapsed)
                ids.append(fid)
                joints = [{key: getattr(j, 'get_' + key)() for key in
                           ['range_min', 'range_max', 'position', 'position_norm', 'goal',
                            'goal_norm', 'current_ma', 'velocity']} for j in frame.get_joints()]
                rows.append(dict(episode_start_ns=start.get_local_stamp_ns(),
                                 global_frame_id=bytes(frame.get_global_frame_id()),
                                 timestamp_ns_since_episode_start=elapsed, joints=joints,
                                 images=[dict(jpeg=bytes(images[order.index(cam)].get_jpeg())) for cam in reference_order],
                                 task=args.task))
            gaps = [(b - a) / 1e9 for a, b in zip(times, times[1:])]
            episodes.append(dict(episode_index=episode_index, **pair, frames=len(group),
                                 first_global_frame_id=ids[0], last_global_frame_id=ids[-1],
                                 first_frame_offset_s=times[0] / 1e9, last_frame_offset_s=times[-1] / 1e9,
                                 max_frame_gap_s=max(gaps, default=0),
                                 gaps_over_500ms=sum(g > .5 for g in gaps)))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows, schema=schema()), args.output)
        previews.mkdir()
        for i, image in enumerate(rows[0]['images']):
            with Image.open(io.BytesIO(image['jpeg'])) as im:
                im.save(previews / f'camera-{i}.jpg')
        report = dict(task=args.task, source_inventory=str(args.inventory.resolve()),
                      boundary_rule='start pointer inclusive, end pointer exclusive; one tag pair per episode',
                      automatic_episode_filtering=False, resampling=False,
                      normalized_camera_queue_order=reference_order, reordered_frames=reordered,
                      frames=len(rows), episodes=episodes,
                      excluded=[p for p in all_pairs if p['name'] in exclusions],
                      incomplete_pairs_not_exported=inventory.get('incomplete', []))
        manifest_path.write_text(json.dumps(report, indent=2) + '\n')
        print(f'Exported {len(rows)} frames in {len(episodes)} exact episodes: {args.output}', flush=True)
        print(f'Manifest: {manifest_path}; previews: {previews}', flush=True)
    finally:
        client.close_connection()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exclude-name', action='append', default=[])
    parser.add_argument('--task', default='place cheese on pizza')
    parser.add_argument('--host', default='localhost')
    asyncio.run(export(parser.parse_args()))
