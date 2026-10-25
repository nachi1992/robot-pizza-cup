#!/usr/bin/env python3
"""Local SmolVLA/Station inference. Motor writes require explicit --execute."""
import argparse
import asyncio
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import time

import torch

import infer_act_station as station
from act_image_preprocessing import load_preprocessing
from smolvla_common import DEFAULT_DATASET, REPO, RENAME_MAP, TASK, check_version, dataset_contract, write_json


def check_calibration(report, contract):
    if (report['motor_ids'] != contract['motor_ids']
            or report['calibration_ranges'] != contract['calibration_ranges']):
        raise ValueError('Live Station calibration differs from this SmolVLA checkpoint; refusing inference/control')


def predict(policy, pre, post, batch, task):
    started=time.perf_counter()
    with torch.inference_mode():
        chunk=policy.predict_action_chunk(pre(dict(batch, task=[task])))
        actions=post(chunk).squeeze(0).cpu()
    if actions.ndim!=2 or actions.shape[1]!=8 or actions.shape[0]<5 or not torch.isfinite(actions).all():
        raise ValueError(f'Expected finite SmolVLA actions shaped [time,8], got {tuple(actions.shape)}')
    return actions[:5],time.perf_counter()-started


async def timed_loop(args, client, stream, policy, pre, post, initial, report, output, contract):
    """One worker, five committed actions, one candidate chunk; never refill with stale actions."""
    batch,_,reference=initial
    check_calibration(reference,contract)
    actions,seconds=await asyncio.to_thread(predict,policy,pre,post,batch,args.task)
    chunk_stamp=reference['source_local_stamp_ns']; chunk_id=reference['source_frame_id']
    station.validate_control(args,reference,reference,chunk_stamp)
    started=time.monotonic(); deadline=float('inf') if args.continuous else started+args.episode_seconds; next_tick=started
    previous_id=chunk_id; index=0; pending=None; records=[]; predictions=[seconds]; ticks=[]
    report.update(initial_prediction_seconds=seconds,execution_status='running',actions=records)

    async def predict_next(sample):
        batch,_,source=sample
        check_calibration(source,contract)
        result,elapsed=await asyncio.to_thread(predict,policy,pre,post,batch,args.task)
        return result,elapsed,source

    try:
        while time.monotonic()<deadline:
            try:
                sample=await stream.next_after(previous_id,timeout=min(args.max_age,max(.001,deadline-time.monotonic())))
            except TimeoutError:
                # The final frame wait is bounded by the trial deadline. Its
                # expiry is normal completion; earlier telemetry timeouts fail.
                if time.monotonic()>=deadline:break
                raise
            current=sample[2];previous_id=current['source_frame_id']
            check_calibration(current,contract)
            if index==len(actions):
                if pending is None:pending=asyncio.create_task(predict_next(sample))
                actions,seconds,source=await asyncio.wait_for(pending,timeout=args.max_age)
                pending=None;index=0;predictions.append(seconds)
                chunk_stamp=source['source_local_stamp_ns'];chunk_id=source['source_frame_id']
                # Re-read telemetry after waiting for inference. Never command
                # using the position sampled before a slow model prediction.
                sample=await stream.next_after(previous_id,timeout=args.max_age)
                current=sample[2];previous_id=current['source_frame_id']
                check_calibration(current,contract)
            ready=time.monotonic()
            await asyncio.sleep(max(0,next_tick-ready))
            if time.monotonic()>=deadline:break
            station.validate_control(args,current,reference,chunk_stamp)
            plan=station.prepare_action(args,actions[index].tolist(),current)
            record=dict(plan,frame_id=previous_id,prediction_frame_id=chunk_id,
                        observation_age_s=(time.time_ns()-current['source_local_stamp_ns'])/1e9,
                        prediction_age_s=(time.time_ns()-chunk_stamp)/1e9,status='planned')
            records.append(record)
            if args.execute:
                # Recheck immediately before sending. No torque-enable, offset,
                # speed or calibration commands are issued by this runner.
                station.validate_control(args,current,reference,chunk_stamp)
                record['status']='submitting; outcome unknown until enqueue returns'
                await asyncio.wait_for(station.send_commands(client,[station.goal_command(args,current,plan)]),3)
                record['status']='queued'
            now=time.monotonic(); ticks.append(now);record['elapsed_s']=now-started
            index+=1;next_tick=now+.1
            # Start preparing the next chunk while three current actions remain.
            # The candidate replaces actions only at a five-action boundary.
            if index==2 and pending is None:pending=asyncio.create_task(predict_next(sample))
            if len(records)%10==0:
                print(f'{len(records)} actions {"queued" if args.execute else "planned"}, {now-started:.1f}s',flush=True)
        report['execution_status']='duration reached'
    except BaseException as error:
        report['execution_status']='stopped: '+(str(error) or type(error).__name__)
        raise
    finally:
        if pending:
            pending.cancel();await asyncio.gather(pending,return_exceptions=True)
        report.update(commands_sent=any(r['status']=='queued' for r in records),
                      queued_command_count=sum(r['status']=='queued' for r in records),
                      elapsed_s=time.monotonic()-started,mean_prediction_seconds=sum(predictions)/len(predictions),
                      measured_control_hz=(len(ticks)-1)/(ticks[-1]-ticks[0]) if len(ticks)>1 else None)
        write_json(output/'prediction.json',report)
        print(f'Stopped: {len(records)} actions; {report["queued_command_count"]} commands queued.',flush=True)
        if args.execute:print('The last servo goal remains active.',flush=True)


async def run(args):
    check_version()
    if args.continuous and args.episode_seconds is not None:
        raise ValueError('Use either --continuous or --episode-seconds')
    if args.execute and args.episode_seconds is None and not args.continuous:args.episode_seconds=10
    if args.episode_seconds is not None and args.episode_seconds < 1:
        raise ValueError('--episode-seconds must be a positive number of seconds')
    if not 0<args.max_age<=3:raise ValueError('--max-age must be >0 and <=3 seconds')
    if not 1<=args.step_ticks<=4095 or not 1<=args.max_delta_ticks<=4095:
        raise ValueError('Tick caps must be between 1 and 4095')
    cp=args.checkpoint.resolve()
    contract=json.loads((cp/'station_contract.json').read_text())
    actual=dataset_contract(args.dataset_root)
    if contract!=actual:raise ValueError('Checkpoint and dataset contracts differ')
    if args.task!=contract['task']:raise ValueError('Use the task instruction recorded in this checkpoint')
    args.image_preprocessing=load_preprocessing(cp)
    if args.image_preprocessing!=contract['image_preprocessing']:
        raise ValueError('Checkpoint image preprocessing differs from its contract')
    if not args.camera_map.is_file():raise FileNotFoundError('Provide a verified local --camera-map')
    station.camera_override(str(args.camera_map.resolve()))
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    config=SmolVLAConfig.from_pretrained(cp,local_files_only=True)
    if tuple(config.output_features['action'].shape)!=(8,) or config.n_action_steps!=5:
        raise ValueError('Checkpoint must predict eight joints and use five-action replanning')
    vlm=cp/'vlm_assets'
    if not (vlm/'tokenizer.json').is_file():raise FileNotFoundError('Checkpoint missing portable vlm_assets/')
    config.vlm_model_name=str(vlm);config.load_vlm_weights=False;config.device=args.device
    print(f'Loading SmolVLA: {cp}',flush=True)
    policy=SmolVLAPolicy.from_pretrained(cp,config=config,local_files_only=True,strict=True).to(args.device).eval()
    pre,post=make_pre_post_processors(config,pretrained_path=str(cp),preprocessor_overrides={
        'device_processor':{'device':args.device},'tokenizer_processor':{'tokenizer_name':str(vlm)},
        'rename_observations_processor':{'rename_map':RENAME_MAP}})
    policy.reset()
    output=args.output_dir or REPO/'outputs/inference'/datetime.now().strftime('smolvla-station-%Y%m%d-%H%M%S-%f')
    output.mkdir(parents=True,exist_ok=False)
    client=await station.new_station_client(args.host,logging.getLogger('smolvla-station'))
    stream=None
    try:
        stream=station.ObservationStream(args,client)
        initial=await stream.next_after(-1,timeout=args.max_age)
        batch,images,source=initial
        check_calibration(source,contract)
        for name,image in images.items():image.save(output/f'{name}.png')
        report=dict(source,checkpoint=str(cp),policy='smolvla',task=args.task,commands_sent=False,continuous=args.continuous)
        print('Calibration matches; camera inputs saved:',output,flush=True)
        if args.continuous or args.episode_seconds is not None:
            await timed_loop(args,client,stream,policy,pre,post,initial,report,output,contract)
        else:
            actions,elapsed=await asyncio.to_thread(predict,policy,pre,post,batch,args.task)
            report.update(action_prefix_goal_norm=actions.tolist(),prediction_seconds=elapsed,queued_command_count=0)
            write_json(output/'prediction.json',report)
            print(f'Predicted five actions in {elapsed:.3f}s; motor commands sent: 0.',flush=True)
    finally:
        if stream is not None:await stream.close()
        station.close_station_client(client)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--dataset-root',type=Path,default=DEFAULT_DATASET)
    parser.add_argument('--camera-map',type=Path,default=Path(__file__).resolve().parents[1]/'config/station-local-cameras.json')
    parser.add_argument('--host',default=os.environ.get('STATION_HOST','localhost'))
    parser.add_argument('--follower-serial',default=os.environ.get('FOLLOWER_SERIAL','5B61034794'))
    parser.add_argument('--task',default=TASK)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--max-age',type=float,default=1.0)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--execute',action='store_true')
    duration=parser.add_mutually_exclusive_group()
    duration.add_argument('--episode-seconds',type=int)
    duration.add_argument('--continuous',action='store_true',help='Keep planning/executing until Ctrl+C or a control check fails')
    parser.add_argument('--step-ticks',type=int,default=200)
    parser.add_argument('--max-delta-ticks',type=int,default=200)
    try:asyncio.run(run(parser.parse_args()))
    except KeyboardInterrupt:raise SystemExit('Stopped; last submitted servo goal remains active.')


if __name__=='__main__':main()
