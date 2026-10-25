#!/usr/bin/env python3
"""Inspect follower torque, or hold its current pose and enable torque."""
import argparse
import asyncio
import logging
import os
import time

from export_tagged_takes import read_batch
from station_py import send_commands
from station_connection import new_station_client, close_station_client
from target.gen_python.protobuf.station import commands, drivers, inference
from target.gen_python.protobuf.drivers.st3215 import st3215
from target.gen_python.protobuf.drivers.motors_mirroring import mirroring


async def latest(client, queue):
    stream = client.read_from_tail(queue, offset=b'\x00', limit=1, step=1, buf_size=1)
    item = await asyncio.wait_for(stream.data.get(), 8)
    if item is None:
        raise RuntimeError(f'No data from {queue}: {stream.err}')
    return bytes(item.Data)


def follower(data, serial):
    buses = st3215.InferenceStateReader(memoryview(data)).get_buses()
    selected = [b for b in buses if b.get_bus().get_serial_number() == serial]
    if len(selected) != 1:
        raise ValueError(f'Follower {serial} is unavailable')
    bus = selected[0]
    if abs(time.time_ns() - bus.get_system_stamp_ns()) > 3_000_000_000:
        raise ValueError('Follower telemetry is stale')
    motors = sorted(bus.get_motors(), key=lambda m: m.get_id())
    if len(motors) != 8 or len({m.get_id() for m in motors}) != 8:
        raise ValueError('Expected eight unique follower motors')
    for m in motors:
        if len(m.get_state()) < 66 or abs(time.time_ns() - m.get_system_stamp_ns()) > 3_000_000_000:
            raise ValueError(f'Motor {m.get_id()} telemetry is missing or stale')
    return motors


def sync(serial, address, values):
    cmd = st3215.Command(target_bus_serial=serial, sync_write=st3215.ST3215SyncWriteCommand(
        address=address, motors=[st3215.ST3215SyncWriteCommand_MotorWrite(motor_id=mid, value=value)
                                for mid, value in values]))
    return commands.DriverCommand(type=drivers.StationCommandType.STC_ST3215_COMMAND, body=cmd.encode())


async def main(args):
    client = await new_station_client(args.host, logging.getLogger('follower-torque'))
    try:
        motors = follower(await latest(client, 'st3215/inference'), args.serial)
        print(f'Follower {args.serial}: ' + ', '.join(
            f'{m.get_id()}={"ON" if m.get_state()[40] else "OFF"}' for m in motors))
        if args.action == 'status':
            return
        source = inference.InferenceRxReader(memoryview(await latest(client, 'inference-states')))
        if abs(time.time_ns() - source.get_local_stamp_ns()) > 3_000_000_000:
            raise ValueError('Station inference snapshot is stale')
        for ref in source.get_entries():
            if ref.get_type() != 32:
                continue
            pointer = int.from_bytes(ref.get_ptr(), 'little')
            saved = await read_batch(client, ref.get_queue(), pointer, 1)
            if len(saved) != 1 or int.from_bytes(saved[0].ID.ID, 'little') != pointer:
                raise ValueError('Cannot verify Station mirroring state')
            state = mirroring.RxEnvelopeReader(memoryview(bytes(saved[0].Data))).get_state()
            for link in state.get_mirroring():
                if (link.get_source().get_id().get_unique_id() == args.serial or any(
                        target.get_id().get_unique_id() == args.serial for target in link.get_targets())):
                    raise ValueError('Select Self-controlled on the follower in Station to stop mirroring first')
        # Refresh after checking mirroring; hold the measured pose before torque-on.
        motors = follower(await latest(client, 'st3215/inference'), args.serial)
        pack = []
        if args.action == 'on':
            if any(m.get_state()[65] != 0 for m in motors):
                raise ValueError('A follower motor reports an error; refusing torque-on')
            pack.append(sync(args.serial, 0x2A, [(m.get_id(), bytes(m.get_state()[56:58])) for m in motors]))
        enabled = args.action == 'on'
        pack.append(sync(args.serial, 0x28, [(m.get_id(), bytes([int(enabled)])) for m in motors]))
        # No retry: an enqueue failure may leave command delivery uncertain.
        await asyncio.wait_for(send_commands(client, pack), 5)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            await asyncio.sleep(0.1)
            motors = follower(await latest(client, 'st3215/inference'), args.serial)
            if all(bool(m.get_state()[40]) == enabled for m in motors):
                print(f'Confirmed follower torque {"ON, holding its measured pose" if enabled else "OFF"}. '
                      'No leader commands were sent.')
                return
        raise RuntimeError('Command submitted but torque state was not confirmed; check Station before retrying')
    finally:
        close_station_client(client)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['status', 'on', 'off'], nargs='?', default='status')
    parser.add_argument('--host', default=os.environ.get('STATION_HOST', 'localhost'))
    parser.add_argument('--serial', default=os.environ.get('FOLLOWER_SERIAL', '5B61034794'))
    args = parser.parse_args()
    asyncio.run(asyncio.wait_for(main(args), 30))
