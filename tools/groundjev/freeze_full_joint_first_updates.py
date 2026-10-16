"""One blocking observer freezes actual first shared updates, without job restarts."""
from collections import Counter
import argparse
import json
import math
import os
from pathlib import Path
import random
import select

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--owner',type=Path,default=ROOT/'outputs/groundjev/training/full_joint_source_epoch1_owner_v1')
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/groundjev/training/full_joint_source_epoch1_first5_observer_v1')
    parser.add_argument('--owner-session',type=int,default=None)
    parser.add_argument('--resource-registration',type=Path,
        default=ROOT/'outputs/groundjev/resource_accounting/first5_shared_observer_v1.json')
    args=parser.parse_args()
    owner=args.owner.resolve()
    output=args.out.resolve()
    output.mkdir(parents=True,exist_ok=False)
    state=json.loads((owner/'owner_state.json').read_text())
    pid,tick=state['owner_pid'],state['owner_starttick']
    fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    assert fields[0]!='Z' and fields[19]==tick and Path(f'/proc/{pid}/cwd').resolve()==ROOT
    assert 'tools/groundjev/run_full_joint_source_training.py' in Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
    budget=json.loads(args.resource_registration.read_text())
    registration=dict(owner=[pid,tick],original_owner_session=args.owner_session,target_step=5,target_examples=80,
        output=str(output),observer_pid=os.getpid(),
        prior_registered_feature_upper_bytes=budget['prior_registered_feature_upper_bytes'],
        observer_report_cap_bytes=budget['observer_report_cap_bytes'],
        combined_registered_feature_upper_bytes=budget['combined_registered_feature_upper_bytes'],
        global_temporary_feature_budget_bytes=budget['global_temporary_feature_budget_bytes'],
        no_GPU_or_training_updates_by_observer=True,full_goal_complete=False)
    (output/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    training=Path('/data3/user/LocateMOT-checkpoints/onejev_full_joint_source_epoch1_v1')
    descriptor=os.pidfd_open(pid);waiter=select.poll();waiter.register(descriptor,select.POLLIN)
    previous=None
    try:
        while True:
            state=json.loads((owner/'owner_state.json').read_text())
            if state['status']!=previous:
                print(json.dumps(dict(status='WAITING_ACTUAL_FIRST5_SHARED_UPDATES',source_stage=state['status'])),flush=True)
                previous=state['status']
            assert state['status']!='FULL_SOURCE_JOINT_OWNER_FAILED',state.get('error')
            log=training/'log.jsonl'
            records=[]
            if log.exists():
                with log.open() as file:
                    for line in file:
                        if not line.endswith('\n'):break
                        row=json.loads(line)
                        if 'loss'in row and row['step']<=5:records.append(row)
            if records and records[-1]['step']==5:break
            if waiter.poll(20000):
                raise RuntimeError('Original full-source owner ended before observed five actual updates; inspect original terminal status')
    finally:os.close(descriptor)
    assert [r['step'] for r in records]==[1,2,3,4,5]
    assert [r['examples'] for r in records]==[16,32,48,64,80]
    assert all(math.isfinite(r[k]) for r in records for k in ('loss','ce','brier','grad_norm'))
    assert all(r['grad_norm']>0 for r in records)
    cpu=json.loads((owner/'CPU_plan_validation.json').read_text())
    assert cpu['status']=='FULL_SOURCE_ALL_NATIVE_MM_ROWS_AND_FOUR_RANK_GLOBAL16_PLAN_VALIDATED'
    assert cpu['rows']==459027 and cpu['dropped_rows']==0 and cpu['planned_world_size']==4
    settings=json.loads((training/'args.json').read_text())
    assert settings['world_size']==4 and settings['accum_rows']==16 and settings['paired_index_shuffle']
    assert settings['distributed_sft'] and settings['deterministic_execution'] and settings['seed_global_rngs']
    assert settings['full_finetune'] and not settings['train_vision'] and settings['seed']==0
    assert settings['max_steps']==cpu['planned_updates']
    initial=json.loads((training/'initial_fp32_state_receipt.json').read_text())
    assert initial['source_step']==600 and initial['initialization_max_difference']==0
    assert initial['parameters']==752393024 and initial['FP32_parameter_entries']==320
    manifest=json.loads(Path(settings['train_file']).read_text())
    order=list(range(manifest['rows']));random.Random(0).shuffle(order)
    task_counts=Counter();sources=Counter()
    for index in order[:80]:
        start=0
        for source in manifest['sources']:
            if start<=index<start+source['rows']:
                task_counts[source['bucket']]+=1;sources[source['path']]+=1;break
            start+=source['rows']
        else:raise AssertionError(index)
    assert set(task_counts)=={'mot','rmot','ovmot'} and sum(task_counts.values())==80
    workers=[];config=str(owner/'training_config.json')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
            if fields[0]=='Z':continue
            command=(proc/'cmdline').read_bytes().decode().split('\0')
            if 'tools/groundjev/run_source_manifest_sft.py' not in command or config not in command:continue
            environment={p.split('=',1)[0]:p.split('=',1)[1] for p in (proc/'environ').read_bytes().decode().split('\0') if '='in p}
            if 'RANK'not in environment:continue
            assert environment['CUDA_VISIBLE_DEVICES']=='4,5,6,7' and (proc/'cwd').resolve()==ROOT
            rank=int(environment['RANK']);assert int(environment['LOCAL_RANK'])==rank
            workers.append(dict(pid=int(proc.name),starttick=fields[19],rank=rank,physical_gpu=4+rank))
        except (FileNotFoundError,ProcessLookupError):continue
    assert sorted(w['rank'] for w in workers)==[0,1,2,3]
    report=dict(status='ACTUAL_FIRST5_FOUR_GPU_SHARED_UPDATES_AND_ALL_TASK_EXPOSURE_VERIFIED',
        CPU_plan=cpu,actual_step_in_snapshot=5,actual_rows_seen=80,actual_first5_logs=records,
        rows_per_task_derived_from_validated_actual_sampler=dict(task_counts),
        first80_source_row_counts=dict(sources),live_exact_owned_rank_workers=sorted(workers,key=lambda w:w['rank']),
        initialization=initial,training_backend='deterministic_seeded_fla052_training_only',
        inference_backend='original_reference_Torch',
        complete_epoch_or_final_tracking_metrics=False,full_goal_complete=False)
    (output/'snapshot.json').write_text(json.dumps(report,indent=2)+'\n')
    assert sum(p.stat().st_size for p in output.rglob('*') if p.is_file())<registration['observer_report_cap_bytes']
    print(json.dumps(dict(status=report['status'],actual_updates=5,actual_rows=80,task_counts=dict(task_counts))),flush=True)


if __name__=='__main__':main()
