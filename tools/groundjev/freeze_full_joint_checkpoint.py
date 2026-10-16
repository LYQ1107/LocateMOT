"""One blocking CPU observer validates an actual four-rank FP32/Adam/RNG save."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import select

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--owner',type=Path,required=True)
    parser.add_argument('--training',type=Path,required=True);parser.add_argument('--step',type=int,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    owner=json.loads((args.owner/'owner_state.json').read_text());pid=owner['owner_pid']
    proc=Path('/proc')/str(pid);fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
    assert fields[0]!='Z' and fields[19]==owner['owner_starttick'] and (proc/'cwd').resolve()==ROOT
    assert 'tools/groundjev/run_full_joint_source_training.py' in (proc/'cmdline').read_bytes().decode().split('\0')
    args.out.mkdir(parents=True,exist_ok=False)
    registration=dict(status='WAITING_ORIGINAL_FOUR_GPU_FULL_STATE_CHECKPOINT',original_owner_pid=pid,
        original_owner_starttick=owner['owner_starttick'],observer_pid=os.getpid(),target_step=args.step,
        training=str(args.training),new_GPU_or_optimizer_updates=0,
        metadata_suballocation_bytes=2**20,metadata_reservation='Existing8MiB native adapter/report reservation; no new feature allocation',
        combined_registered_feature_upper_bytes=26710582238,global_temporary_feature_budget_bytes=26843545600,
        full_goal_complete=False)
    (args.out/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    print(json.dumps(registration),flush=True)
    descriptor=os.pidfd_open(pid);waiter=select.poll();waiter.register(descriptor,select.POLLIN)
    try:
        while True:
            state=json.loads((args.owner/'owner_state.json').read_text())
            assert state['status']!='FULL_SOURCE_JOINT_OWNER_FAILED',state.get('error')
            receipt_path=args.training/'latest_training_state.json'
            if receipt_path.exists():
                receipt=json.loads(receipt_path.read_text())
                if receipt['step']>=args.step:break
            if waiter.poll(10000):raise RuntimeError('Original owner exited before the requested full state was saved')
    finally:os.close(descriptor)
    assert receipt['step']==args.step and receipt['seen']==16*args.step
    import torch
    torch.set_num_threads(4)
    checkpoint=torch.load(receipt['path'],weights_only=True,map_location='cpu',mmap=True)
    assert checkpoint['format']=='groundjev-full-fp32-adam-rng-v1'
    assert checkpoint['step']==checkpoint['next_step_in_epoch']==args.step and checkpoint['epoch']==0
    assert checkpoint['seen']==receipt['seen']
    protocol=checkpoint['protocol']
    assert protocol['world_size']==4 and protocol['accum_rows']==16 and protocol['max_rows']==1
    assert protocol['sampling_protocol']=='paired-index-shuffle-v1'
    assert protocol['distributed_training_protocol']=='native-summed-global-row-mean-allreduce-v1'
    assert protocol['full_finetune'] and not protocol['train_vision'] and protocol['seed']==0
    assert protocol['max_steps']==28690 and protocol['epochs']==1
    assert 'deterministic_execution' in protocol and 'global_rng_initialization' in protocol
    masters=checkpoint['fp32_trainable_parameters'];moments=checkpoint['optimizer']['state']
    assert len(masters)==len(moments)==320
    for tensor in masters.values():assert tensor.dtype==torch.float32 and torch.isfinite(tensor).all()
    elements=sum(v.numel() for v in masters.values());assert elements==752393024
    for values in moments.values():
        assert int(values['step'])==args.step
        for name in ('exp_avg','exp_avg_sq'):
            assert values[name].dtype==torch.float32 and torch.isfinite(values[name]).all()
    moment_elements=sum(values[name].numel() for values in moments.values() for name in ('exp_avg','exp_avg_sq'))
    assert moment_elements==2*elements and checkpoint['scheduler']['last_epoch']==args.step
    ranks=checkpoint['rank_rngs'];assert [r['rank'] for r in ranks]==[0,1,2,3]
    for rank in ranks:
        assert all(k in rank for k in ('torch_rng','cuda_rng','python_rng','numpy_rng'))
        assert rank['torch_rng'].dtype==rank['cuda_rng'].dtype==torch.uint8
        assert rank['torch_rng'].numel()>0 and rank['cuda_rng'].numel()>0
    manifest=json.loads(Path(protocol['train_file']).read_text());assert manifest['rows']==459027
    order=list(range(manifest['rows']));random.Random(0).shuffle(order);counts=Counter()
    for index in order[:checkpoint['seen']]:
        offset=0
        for source in manifest['sources']:
            if offset<=index<offset+source['rows']:counts[source['bucket']]+=1;break
            offset+=source['rows']
        else:raise AssertionError(index)
    assert sum(counts.values())==checkpoint['seen'] and set(counts)=={'mot','rmot','ovmot'}
    records=[]
    with (args.training/'log.jsonl').open() as file:
        for line in file:
            if not line.endswith('\n'):break
            row=json.loads(line)
            if 'loss' in row and row['step']<=args.step:records.append(row)
    assert records[-1]['step']==args.step and records[-1]['examples']==checkpoint['seen']
    assert all(math.isfinite(r[k]) for r in records for k in ('loss','ce','brier','grad_norm'))
    assert (args.training/f'ckpt-step{args.step}').is_dir() and not torch.cuda.is_initialized()
    result=dict(status='ACTUAL_FOUR_GPU_FULL_FP32_ADAM_ALL_RANK_RNG_CHECKPOINT_VERIFIED',
        actual_step=args.step,actual_questions=checkpoint['seen'],planned_updates=28690,planned_questions=459027,
        actual_sampler_task_questions=dict(counts),FP32_parameters_checked=elements,Adam_moment_elements_checked=moment_elements,
        all320_FP32_masters_and_both_Adam_moments_finite=True,all_Adam_steps_scheduler_and_cursor_exact=True,
        all_four_rank_Torch_CUDA_Python_NumPy_RNG_states_saved=True,protocol=protocol,full_state_receipt=receipt,
        finite_logged_update_records_checked=len(records),latest_log=records[-1],
        saved_export=str(args.training/f'ckpt-step{args.step}'),new_GPU_or_training_updates=0,
        actual_milestone_restart_or_new_native_inference_performed=False,
        epoch_and_final_tracking_controls_complete=False,ordinary_model_performance_achieved=False,full_goal_complete=False)
    (args.out/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    assert sum(p.stat().st_size for p in args.out.iterdir() if p.is_file())<registration['metadata_suballocation_bytes']
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
