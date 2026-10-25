#!/usr/bin/env python3
"""Read Station observations and predict ACT actions; motor writes require --execute."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
from functools import lru_cache
from importlib.metadata import version
import io
import json
import logging
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from act_image_preprocessing import load_preprocessing, preprocess_camera

# This existing exporter sets the Station module paths and supplies exact reads.
import station_tag  # Reuse the existing Station Python path setup.
from station_py import send_commands
from station_connection import new_station_client, close_station_client
from target.gen_python.protobuf.drivers.inferences import normvla
from target.gen_python.protobuf.drivers.st3215 import st3215
from target.gen_python.protobuf.drivers.motors_mirroring import mirroring
from target.gen_python.protobuf.station import commands, drivers, inference

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = REPO / 'outputs/train/act-cheese-20261024-023748/checkpoints/000750/pretrained_model'


@lru_cache(maxsize=4)
def dataset_config(root, camera_map=None):
    root = Path(root)
    notes = json.loads((root / 'conversion_notes.json').read_text())
    info = json.loads((root / 'meta/info.json').read_text())
    if camera_map is not None:
        cameras, names = camera_override(camera_map)
    else:
        manifest = json.loads((root / 'station_export_manifest.json').read_text())
        cameras, names = manifest['normalized_camera_queue_order'], notes['camera_names']
    if len(cameras) != 2 or len(set(cameras)) != 2 or len(names) != 2 or set(names) != {'top', 'front'}:
        raise ValueError('Dataset must identify two distinct top/front cameras')
    fps = float(info['fps'])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Dataset FPS must be finite and positive')
    calibration_path = root / 'meta/station_calibration.json'
    calibration = json.loads(calibration_path.read_text()) if calibration_path.is_file() else None
    if calibration is not None:
        ranges = np.asarray(calibration.get('calibration_ranges'))
        ids = calibration.get('motor_ids')
        if ranges.shape != (8, 2) or ids != sorted(set(ids or [])) or len(ids) != 8:
            raise ValueError('Training calibration must identify eight sorted motor IDs and ranges')
    return cameras, names, fps, calibration


@lru_cache(maxsize=4)
def camera_override(path):
    mapping = json.loads(Path(path).read_text())
    if (set(mapping) != {'top', 'front'} or any(not isinstance(v, str) for v in mapping.values())
            or len(set(mapping.values())) != 2):
        raise ValueError('--camera-map must identify distinct top/front Station queue strings')
    return [mapping['top'], mapping['front']], ['top', 'front']


async def read_exact(client, queue, pointer):
    # One valid record is sufficient. The two-slot buffer also lets the library
    # drain the stream's end marker without adding it to the control latency.
    stream = client.read_from_offset(queue, offset=pointer.to_bytes(8, 'little'), limit=1, step=1, buf_size=2)
    entry = await asyncio.wait_for(stream.data.get(), 3)
    if entry is None or int.from_bytes(entry.ID.ID, 'little') != pointer:
        raise ValueError(f'Exact Station record {queue}@{pointer} unavailable: {stream.err}')
    return bytes(entry.Data)


async def mirror_targets(client, references):
    cache = getattr(client, '_act_mirror_cache', {})
    client._act_mirror_cache = cache
    targets = []
    for ref in references:
        if ref.get_type() != 32:
            continue
        queue, pointer = ref.get_queue(), int.from_bytes(ref.get_ptr(), 'little')
        cached_pointer, cached_targets = cache.get(queue, (None, None))
        if cached_pointer != pointer:
            data = await read_exact(client, queue, pointer)
            state = mirroring.RxEnvelopeReader(memoryview(data)).get_state()
            cached_targets = [target.get_id().get_unique_id()
                              for link in state.get_mirroring() for target in link.get_targets()]
            cache[queue] = (pointer, cached_targets)
        targets.extend(cached_targets)
    return targets


async def observation(args, client, streamed_entry=None):
    started = time.perf_counter()
    timings = {}
    if streamed_entry is None:
        stream = client.read_from_tail('inference/normvla', offset=b'\x00', limit=1, step=1, buf_size=2)
        entry = await asyncio.wait_for(stream.data.get(), 3)
    else:
        entry = streamed_entry
    if entry is None:
        raise RuntimeError(f'No normvla observation: {stream.err}')
    frame = normvla.FrameReader(memoryview(bytes(entry.Data)))
    pointer = int.from_bytes(frame.get_global_frame_id(), 'little')
    timings['frame_read'] = time.perf_counter() - started
    stage = time.perf_counter()
    source_data = await read_exact(client, 'inference-states', pointer)
    source = inference.InferenceRxReader(memoryview(source_data))
    timings['source_read'] = time.perf_counter() - stage
    age = (time.time_ns() - source.get_local_stamp_ns()) / 1e9
    if not -args.max_age <= age <= args.max_age:
        raise RuntimeError(f'Station observation is stale ({age:.2f}s). Its auto normvla generator '
                           'requires exactly one torque-enabled bus; this helper does not enable torque.')
    joints, images = frame.get_joints(), frame.get_images()
    if len(joints) != 8 or len(images) != 2:
        raise ValueError(f'Expected 8 joints and 2 cameras, got {len(joints)} and {len(images)}')
    references = source.get_entries()
    cameras = [e.get_queue() for e in references if e.get_type() == 21]
    camera_map = str(args.camera_map.resolve()) if getattr(args, 'camera_map', None) is not None else None
    training_cameras, names, _, training_calibration = dataset_config(str(args.dataset_root.resolve()), camera_map)
    if getattr(args, 'camera_map', None) is not None:
        training_cameras, names = camera_override(str(args.camera_map.resolve()))
    if len(cameras) != 2 or len(set(cameras)) != 2 or set(cameras) != set(training_cameras):
        raise ValueError(f'Camera identities differ from training. Live={cameras}; training={training_cameras}')
    bus_refs = [e for e in references if e.get_type() == 13]
    if len(bus_refs) != 1:
        raise ValueError('Expected one ST3215 inference reference')
    ref = bus_refs[0]
    stage = time.perf_counter()
    bus_data, targets = await asyncio.gather(
        read_exact(client, ref.get_queue(), int.from_bytes(ref.get_ptr(), 'little')),
        mirror_targets(client, references),
    )
    timings['telemetry_and_mirror'] = time.perf_counter() - stage
    buses = st3215.InferenceStateReader(memoryview(bus_data)).get_buses()
    followers = [b for b in buses if b.get_bus().get_serial_number() == args.follower_serial]
    if len(followers) != 1:
        raise ValueError(f'Follower {args.follower_serial} is missing from source telemetry')
    motors = sorted(followers[0].get_motors(), key=lambda m: m.get_id())
    ids = [m.get_id() for m in motors]
    if len(motors) != 8 or len(set(ids)) != 8:
        raise ValueError(f'Expected eight unique follower motor IDs; got {ids}')
    torque_enabled, motor_errors = [], []
    for motor, joint in zip(motors, joints):
        state = bytes(motor.get_state())
        if len(state) < 66:
            raise ValueError(f'Motor {motor.get_id()} telemetry is incomplete')
        position = int.from_bytes(state[56:58], 'little')
        goal = int.from_bytes(state[42:44], 'little') if state[40] else position
        actual = (position, goal, motor.get_range_min(), motor.get_range_max())
        expected = (joint.get_position(), joint.get_goal(), joint.get_range_min(), joint.get_range_max())
        if actual != expected:
            raise ValueError(f'Frame does not match follower motor {motor.get_id()}: {actual} != {expected}')
        torque_enabled.append(bool(state[40]))
        motor_errors.append(int(state[65]))
    stage = time.perf_counter()
    batch = {'observation.state': torch.tensor([j.get_position_norm() for j in joints], dtype=torch.float32)}
    decoded = {}
    for name, queue in zip(names, training_cameras):
        jpeg = bytes(images[cameras.index(queue)].get_jpeg())
        with Image.open(io.BytesIO(jpeg)) as image:
            rgb = image.convert('RGB')
            if rgb.size != (224, 224):
                raise ValueError(f'{name}: expected training image size 224x224, got {rgb.size}')
            rgb = preprocess_camera(rgb, name, getattr(args, 'image_preprocessing', {}))
            decoded[name] = rgb.copy()
            # RGB bytes are finite by construction. NumPy avoids dispatching
            # several tiny image operations into PyTorch's CPU thread pool.
            pixels = np.array(rgb, dtype=np.float32)
            pixels *= 1.0 / 255.0
            batch[f'observation.images.{name}'] = torch.from_numpy(pixels.transpose(2, 0, 1).copy())
    if not torch.isfinite(batch['observation.state']).all():
        raise ValueError('Observation contains nonfinite values')
    timings['images'] = time.perf_counter() - stage
    timings['total'] = time.perf_counter() - started
    ranges = [[j.get_range_min(), j.get_range_max()] for j in joints]
    calibration_matches = (ids == training_calibration['motor_ids'] and ranges == training_calibration['calibration_ranges']
                           if training_calibration else None)
    report = dict(source_frame_id=pointer, source_local_stamp_ns=source.get_local_stamp_ns(),
                  observation_age_at_fetch_s=(time.time_ns() - source.get_local_stamp_ns()) / 1e9,
                  follower_serial=args.follower_serial, motor_ids=ids,
                  torque_enabled=torque_enabled, motor_errors=motor_errors, station_mirroring_targets=targets,
                  live_camera_queue_order=cameras, camera_mapping=dict(zip(names, training_cameras)),
                  camera_mapping_file=str(args.camera_map.resolve()) if getattr(args, 'camera_map', None) else None,
                  image_preprocessing=getattr(args, 'image_preprocessing', {}),
                  observation_state=batch['observation.state'].tolist(),
                  raw_positions=[j.get_position() for j in joints],
                  calibration_ranges=ranges, calibration_source='current_station',
                  calibration_matches_training=calibration_matches,
                  training_calibration_ranges=training_calibration['calibration_ranges'] if training_calibration else None,
                  observation_stage_seconds=timings)
    return batch, decoded, report


class ObservationStream:
    """Prefetch validated samples with bounded work and no accumulated backlog."""
    def __init__(self, args, client):
        self.args, self.client = args, client
        self.latest, self.error = None, None
        self.changed = asyncio.Event()
        self.pending = set()
        self.stream = client.read_from_tail('inference/normvla', offset=b'\x00', limit=0, step=1, buf_size=4)
        self.reader = asyncio.create_task(self._read())

    async def _read(self):
        try:
            while True:
                entry = await self.stream.data.get()
                if entry is None:
                    raise RuntimeError(f'Station frame stream ended: {self.stream.err}')
                # Overlap source/telemetry network reads for at most three frames.
                # If overloaded, skip incoming work rather than queue old frames.
                if len(self.pending) < 3:
                    task = asyncio.create_task(self._decode(entry))
                    self.pending.add(task)
                    task.add_done_callback(self.pending.discard)
        except Exception as error:
            self.error = error
            self.changed.set()

    async def _decode(self, entry):
        try:
            sample = await observation(self.args, self.client, streamed_entry=entry)
            if self.latest is None or sample[2]['source_frame_id'] > self.latest[2]['source_frame_id']:
                self.latest = sample
                self.changed.set()
        except Exception as error:
            self.error = error
            self.changed.set()

    async def next_after(self, previous_id, timeout=1.0):
        async with asyncio.timeout(timeout):
            while True:
                if self.error is not None:
                    raise self.error
                if self.latest is not None and self.latest[2]['source_frame_id'] > previous_id:
                    return self.latest
                self.changed.clear()
                await self.changed.wait()

    async def close(self):
        tasks = [self.reader, *self.pending]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def validate_control(args, current, reference, prediction_stamp):
    if current['motor_ids'] != reference['motor_ids'] or current['calibration_ranges'] != reference['calibration_ranges']:
        raise ValueError('Motor IDs or calibration changed after prediction')
    now = time.time_ns()
    for label, stamp in [('observation', current['source_local_stamp_ns']), ('prediction', prediction_stamp)]:
        age = (now - stamp) / 1e9
        if not -args.max_age <= age <= args.max_age:
            raise ValueError(f'{label.capitalize()} is stale ({age:.2f}s)')
    if args.execute:
        if not all(current['torque_enabled']):
            raise ValueError('Follower torque must already be enabled on every joint')
        if args.follower_serial in current['station_mirroring_targets']:
            raise ValueError('Stop leader/follower mirroring in Station before ACT execution')
        if any(current['motor_errors']):
            raise ValueError('Follower reports a motor error; refusing commands')


def prepare_action(args, action_goal_norm, current):
    """Map one already-postprocessed action to calibrated, bounded servo goals."""
    predicted = np.asarray(action_goal_norm, dtype=np.float64)
    if predicted.shape != (8,) or not np.isfinite(predicted).all():
        raise ValueError('Action must be a finite eight-joint goal_norm vector')
    clipped = predicted.clip(0.0, 1.0)
    goals, deltas = [], []
    for value, pos, (lo, hi) in zip(clipped, current['raw_positions'], current['calibration_ranges']):
        span = (hi - lo) % 4096
        present = (pos - lo) % 4096
        target = round(float(value) * span)
        if span == 0 or present > span:
            raise ValueError('Current position is outside the calibrated interval')
        goals.append((lo + target) % 4096)
        deltas.append(target - present)
    largest = max(map(abs, deltas))
    scale = 1.0
    if args.step_ticks is not None and largest > min(args.step_ticks, args.max_delta_ticks):
        scale = min(args.step_ticks, args.max_delta_ticks) / largest
        deltas = [round(delta * scale) for delta in deltas]
        goals = [(pos + delta) % 4096 for pos, delta in zip(current['raw_positions'], deltas)]
    if max(map(abs, deltas)) > args.max_delta_ticks:
        raise ValueError(f'Blocked: move exceeds --max-delta-ticks {args.max_delta_ticks}; use --step-ticks to constrain it')
    return dict(requested_action=predicted.tolist(), clipped_action=clipped.tolist(),
                clipped_joints=(predicted != clipped).tolist(), step_scale=scale,
                current_position_ticks=list(current['raw_positions']), goal_ticks=goals, delta_ticks=deltas)


def goal_command(args, current, plan):
    # Same ST3215 sync-write structure as Norma-core's policy runner.
    command = st3215.Command(target_bus_serial=args.follower_serial,
        sync_write=st3215.ST3215SyncWriteCommand(address=0x2A, motors=[
            st3215.ST3215SyncWriteCommand_MotorWrite(motor_id=motor, value=goal.to_bytes(2, 'little'))
            for motor, goal in zip(current['motor_ids'], plan['goal_ticks'])]))
    return commands.DriverCommand(type=drivers.StationCommandType.STC_ST3215_COMMAND, body=command.encode())


def save_report(output, report):
    (output / 'prediction.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


async def execute_once(args, report, output, client):
    if not args.execute:
        raise ValueError('Motor execution requires explicit --execute')
    report['execution_status'] = 'checking'
    try:
        _, _, current = await observation(args, client)
        validate_control(args, current, report, report['source_local_stamp_ns'])
        plan = prepare_action(args, report['first_action_goal_norm'], current)
        report['first_command'] = plan
        report['execution_delta_ticks'] = plan['delta_ticks']
        report['execution_raw_goal_ticks'] = plan['goal_ticks']
        print(' motor  current  commanded  delta_ticks')
        for motor, position, goal, delta in zip(current['motor_ids'], current['raw_positions'], plan['goal_ticks'], plan['delta_ticks']):
            print(f'{motor:6d} {position:8d} {goal:10d} {delta:+12d}')
        validate_control(args, current, report, report['source_local_stamp_ns'])
        report['execution_status'] = 'submitting; outcome unknown until enqueue returns'
        save_report(output, report)
        await asyncio.wait_for(send_commands(client, [goal_command(args, current, plan)]), 3)
        report['commands_sent'] = True
        report['execution_status'] = 'one goal sync-write queued; physical completion not verified'
    except BaseException as error:
        report['execution_error'] = str(error) or type(error).__name__
        raise
    finally:
        save_report(output, report)


async def execute_episode(args, report, output, client, policy, pre, post, initial_actions):
    """Run the same fixed-rate planning path with motor writes explicitly opted in."""
    _, _, fps, _ = dataset_config(str(args.dataset_root.resolve()))
    period = 1.0 / fps
    started = time.monotonic()
    deadline, next_tick = started + args.episode_seconds, started
    last_frame_id = report['source_frame_id']
    actions = initial_actions  # Already in Station goal_norm; never postprocess again.
    chunk_frame_id, chunk_stamp = last_frame_id, report['source_local_stamp_ns']
    records = report.setdefault('executed_actions' if args.execute else 'planned_actions', [])
    sample_times, prediction_times, enqueue_times = [], [], []
    overruns = 0
    step = 0
    verb = 'sent' if args.execute else 'planned'
    print(f'Episode mode: {verb}; target {fps:g} Hz, replan every {policy.config.n_action_steps} actions.', flush=True)
    try:
        while time.monotonic() < deadline:
            batch, _, current = await fresh_observation(args, client, last_frame_id)
            last_frame_id = current['source_frame_id']
            prediction_seconds = 0.0
            if step and step % policy.config.n_action_steps == 0:
                prediction_started = time.perf_counter()
                # Keep Station's socket reader alive while the model runs.
                actions = await asyncio.to_thread(predict_chunk, policy, pre, post, batch)
                prediction_seconds = time.perf_counter() - prediction_started
                prediction_times.append(prediction_seconds)
                chunk_frame_id, chunk_stamp = last_frame_id, current['source_local_stamp_ns']
            ready = time.monotonic()
            if step and ready > next_tick + 0.005:
                overruns += 1
            # Fetch/compute between command times, then pace to dataset FPS.
            # A slow read simply delays this step; missed slots never cause bursts.
            await asyncio.sleep(max(0.0, next_tick - ready))
            if time.monotonic() >= deadline:
                break
            validate_control(args, current, report, chunk_stamp)
            plan = prepare_action(args, actions[step % policy.config.n_action_steps].tolist(), current)
            record = dict(plan, frame_id=last_frame_id, prediction_frame_id=chunk_frame_id,
                          observation_state=current['observation_state'],
                          observation_age_s=(time.time_ns() - current['source_local_stamp_ns']) / 1e9,
                          prediction_age_s=(time.time_ns() - chunk_stamp) / 1e9,
                          observation_stage_seconds=current['observation_stage_seconds'],
                          prediction_seconds=prediction_seconds, command_status='planned')
            records.append(record)
            if args.execute:
                validate_control(args, current, report, chunk_stamp)
                record['command_status'] = 'submitting; outcome unknown until enqueue returns'
                enqueue_started = time.perf_counter()
                await asyncio.wait_for(send_commands(client, [goal_command(args, current, plan)]), 3)
                enqueue_times.append(time.perf_counter() - enqueue_started)
                record['command_status'] = 'queued'
            sample_time = time.monotonic()
            record['elapsed_s'] = sample_time - started
            sample_times.append(sample_time)
            step += 1
            next_tick = sample_time + period
            if step % 10 == 0:
                rate = (step - 1) / (sample_times[-1] - sample_times[0]) if step > 1 else 0.0
                print(f'Episode: {step} actions {verb}, {sample_time - started:.1f}s, '
                      f'{rate:.2f} Hz; last max change {max(map(abs, plan["delta_ticks"]))} ticks.', flush=True)
        report['episode_status'] = 'duration reached'
    except BaseException as error:
        report['episode_status'] = f'stopped: {str(error) or type(error).__name__}'
        raise
    finally:
        elapsed = time.monotonic() - started
        stages = [r['observation_stage_seconds'] for r in records]
        report.update(episode_elapsed_s=elapsed, commands_sent=any(r['command_status'] == 'queued' for r in records),
                      queued_command_count=sum(r['command_status'] == 'queued' for r in records),
                      control_target_fps=fps, control_deadline_overruns=overruns,
                      measured_control_hz=((len(sample_times) - 1) / (sample_times[-1] - sample_times[0])
                                           if len(sample_times) > 1 else None),
                      mean_observation_stage_seconds={k: float(np.mean([s[k] for s in stages]))
                                                      for k in stages[0]} if stages else {},
                      mean_prediction_seconds=float(np.mean(prediction_times)) if prediction_times else None,
                      mean_enqueue_seconds=float(np.mean(enqueue_times)) if enqueue_times else None)
        save_report(output, report)
        print(f'Episode stopped after {step} actions {verb}; commands queued: {report["queued_command_count"]}.', flush=True)
        if args.execute:
            print('The last servo goal remains active.', flush=True)


async def fresh_observation(args, client, previous_id):
    if getattr(client, '_act_observations', None) is not None:
        return await client._act_observations.next_after(previous_id)
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        result = await asyncio.wait_for(observation(args, client), max(0.001, deadline - time.monotonic()))
        if result[2]['source_frame_id'] > previous_id:
            return result
        await asyncio.sleep(0.01)
    raise TimeoutError('No newer Station frame within one second')


def predict_chunk(policy, pre, post, batch):
    """Return Station goal_norm actions, applying the saved postprocessor once."""
    with torch.inference_mode():
        chunk = policy.predict_action_chunk(pre(batch))
        actions = post(chunk).squeeze(0).cpu()
    if tuple(actions.shape) != (policy.config.chunk_size, 8) or not torch.isfinite(actions).all():
        raise ValueError(f'Invalid finite ACT action chunk: expected {(policy.config.chunk_size, 8)}, got {tuple(actions.shape)}')
    return actions


async def run(args):
    if version('lerobot') != '0.6.1':
        raise RuntimeError('Use the existing LeRobot 0.6.1 environment for this checkpoint')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use --device cpu explicitly if intended')
    if args.max_age <= 0 or not np.isfinite(args.max_age):
        raise ValueError('--max-age must be finite and positive')
    if not 1 <= args.max_delta_ticks <= 500:
        raise ValueError('--max-delta-ticks must be between 1 and 500')
    if args.step_ticks is not None and not 1 <= args.step_ticks <= 500:
        raise ValueError('--step-ticks must be between 1 and 500')
    if args.episode_seconds is not None and not 1 <= args.episode_seconds <= 60:
        raise ValueError('--episode-seconds must be between 1 and 60')
    if args.episode_seconds is not None and args.step_ticks is None:
        raise ValueError('--episode-seconds requires --step-ticks')
    checkpoint = args.checkpoint.resolve()
    for name in ('config.json', 'model.safetensors', 'policy_preprocessor.json', 'policy_postprocessor.json'):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)
    args.image_preprocessing = load_preprocessing(checkpoint)
    if args.image_preprocessing != load_preprocessing(args.dataset_root):
        raise ValueError('Checkpoint and --dataset-root have different image preprocessing; select the dataset used for this model')
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    config = ACTConfig.from_pretrained(checkpoint, local_files_only=True)
    if (tuple(config.input_features['observation.state'].shape) != (8,)
            or tuple(config.output_features['action'].shape) != (8,)
            or set(config.image_features) != {'observation.images.top', 'observation.images.front'}):
        raise ValueError('Checkpoint must match the eight-joint, top/front dataset')
    config.device = args.device
    # All backbone weights are already in the checkpoint: avoid another download.
    config.pretrained_backbone_weights = None
    print(f'Loading ACT checkpoint: {checkpoint}', flush=True)
    policy = ACTPolicy.from_pretrained(checkpoint, config=config, local_files_only=True, strict=True)
    policy.to(args.device).eval()
    pre, post = make_pre_post_processors(config, pretrained_path=str(checkpoint),
                                        preprocessor_overrides={'device_processor': {'device': args.device}})
    policy.reset()
    client = await new_station_client(args.host, logging.getLogger('act-station'))
    frame_stream = None
    try:
        if args.episode_seconds is not None:
            frame_stream = ObservationStream(args, client)
            client._act_observations = frame_stream
            batch, images, report = await frame_stream.next_after(-1, timeout=5)
        else:
            batch, images, report = await asyncio.wait_for(observation(args, client), 30)
        if args.execute:
            validate_control(args, report, report, report['source_local_stamp_ns'])
        if args.device == 'cuda':
            torch.cuda.synchronize()
        started = time.perf_counter()
        actions = await asyncio.to_thread(predict_chunk, policy, pre, post, batch)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        latency = time.perf_counter() - started
        if tuple(actions.shape) != (config.chunk_size, 8) or not torch.isfinite(actions).all():
            raise ValueError(f'Invalid action chunk: {actions.shape}, or nonfinite predictions')
        predicted = actions[0].tolist()
        clipped = actions[0].clamp(0, 1).tolist()
        spans = [(hi - lo) % 4096 for lo, hi in report['calibration_ranges']]
        if any(span == 0 for span in spans):
            raise ValueError('Calibration contains an empty range')
        raw = [(lo + round(value * span)) % 4096
               for value, (lo, hi), span in zip(clipped, report['calibration_ranges'], spans)]
        deltas = [(goal - pos + 2048) % 4096 - 2048 for goal, pos in zip(raw, report['raw_positions'])]
        report.update(checkpoint=str(checkpoint), device=args.device, task=args.task,
                      prediction_seconds=latency, action_chunk_goal_norm=actions.tolist(),
                      first_action_goal_norm=predicted, clipped_first_action_goal_norm=clipped,
                      clipped_first_action_raw_ticks=raw, first_action_delta_ticks=deltas,
                      commands_sent=False)
        output = args.output_dir or REPO / 'outputs/inference' / datetime.now().strftime('act-station-%Y%m%d-%H%M%S-%f')
        output.mkdir(parents=True, exist_ok=False)
        for name, image in images.items():
            image.save(output / f'{name}.png')
        (output / 'prediction.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(f'Live cameras: {report["live_camera_queue_order"]}\nMapped cameras: {report["camera_mapping"]}')
        print(' motor  position  predicted_norm  clipped_goal_ticks  delta_ticks')
        for motor, pos, value, goal, delta in zip(report['motor_ids'], report['raw_positions'], predicted, raw, deltas):
            print(f'{motor:6d} {pos:9d} {value:15.5f} {goal:19d} {delta:+12d}')
        print(f'ACT predicted {len(actions)} actions in {latency:.3f}s. Motor commands sent: 0.')
        print(f'Saved inputs and prediction: {output}')
        if args.episode_seconds is not None:
            await execute_episode(args, report, output, client, policy, pre, post, actions)
        elif args.execute:
            await asyncio.wait_for(execute_once(args, report, output, client), 30)
            print('One goal queued for the follower. Exiting; the servos may finish moving and hold that goal.')
    finally:
        if frame_stream is not None:
            await frame_stream.close()
        close_station_client(client)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path(os.environ.get('ACT_CHECKPOINT', DEFAULT_CHECKPOINT)))
    parser.add_argument('--dataset-root', type=Path, default=REPO / 'data/lerobot-cheese-takes-20261024')
    parser.add_argument('--camera-map', type=Path, help='Explicit top/front queue mapping for a different Station instance')
    parser.add_argument('--host', default=os.environ.get('STATION_HOST', 'localhost'))
    parser.add_argument('--follower-serial', default=os.environ.get('FOLLOWER_SERIAL', '5B61034794'))
    parser.add_argument('--task', default=os.environ.get('TASK', 'place cheese on pizza'))
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--max-age', type=float, default=3.0)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--execute', action='store_true', help='Enable motor writes for one action or a timed episode')
    parser.add_argument('--max-delta-ticks', type=int, default=200,
                        help='Refuse any joint move larger than this (1..500; default: 200)')
    parser.add_argument('--step-ticks', type=int,
                        help='Scale the move so no joint changes by more than this many ticks (1..500); also bounded by --max-delta-ticks')
    parser.add_argument('--episode-seconds', type=int,
                        help='Plan ACT for 1..60 seconds; requires --step-ticks. Motor writes require --execute')
    try:
        asyncio.run(run(parser.parse_args()))
    except KeyboardInterrupt:
        raise SystemExit('Stopped.')
