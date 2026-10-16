"""Freeze a real shared-training milestone and its measured input exposure."""
import argparse
from collections import Counter,defaultdict
import json
import math
import os
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'external/OneJev')]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--step',type=int,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--training',type=Path,default=ROOT/'outputs/groundjev/training/onejev_broader_shared6000_fla_v2')
    parser.add_argument('--initial-step',type=int,default=600)
    parser.add_argument('--registered-feature-upper',type=int,default=26829071326)
    args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import torch
    torch.set_num_threads(4)
    training=args.training
    receipt=json.loads((training/'latest_training_state.json').read_text())
    assert receipt['step']==args.step and receipt['seen']==args.step*4
    checkpoint=torch.load(receipt['path'],map_location='cpu',weights_only=True,mmap=True)
    assert checkpoint['step']==args.step and checkpoint['seen']==receipt['seen']
    assert checkpoint['epoch']==0 and checkpoint['next_step_in_epoch']==args.step
    assert checkpoint['protocol']['sampling_protocol']=='paired-index-shuffle-v1'
    assert checkpoint['protocol']['max_steps']==6000 and checkpoint['protocol']['seed']==0
    assert checkpoint['protocol']['full_finetune'] and not checkpoint['protocol']['train_vision']
    masters=checkpoint['fp32_trainable_parameters'];moments=checkpoint['optimizer']['state']
    assert len(masters)==len(moments)==320
    elements=0
    for tensor in masters.values():
        assert tensor.dtype==torch.float32 and torch.isfinite(tensor).all()
        elements+=tensor.numel()
    assert elements==752393024
    moment_elements=0
    for values in moments.values():
        assert int(values['step'])==args.step
        for key in ('exp_avg','exp_avg_sq'):
            assert values[key].dtype==torch.float32 and torch.isfinite(values[key]).all()
            moment_elements+=values[key].numel()
    assert moment_elements==2*elements and checkpoint['scheduler']['last_epoch']==args.step
    assert all(k in checkpoint for k in ('torch_rng','cuda_rng','python_rng','numpy_rng'))
    initial=json.loads((training/'initial_fp32_state_receipt.json').read_text())
    assert initial['source_step']==args.initial_step and initial['initialization_max_difference']==0
    records=[]
    with (training/'log.jsonl').open() as file:
        for line in file:
            if not line.endswith('\n'):break
            row=json.loads(line)
            if 'loss' in row and row['step']<=args.step:records.append(row)
    assert records[-1]['step']==args.step and records[-1]['examples']==receipt['seen']
    assert all(math.isfinite(row[key]) for row in records for key in ('loss','ce','brier','grad_norm'))
    from train import common
    if str(checkpoint['protocol']['train_file']).endswith('.sources.json'):
        from tools.groundjev.run_source_manifest_sft import configure_source_manifest_sft
        configure_source_manifest_sft()
    else:
        from tools.groundjev.run_group_reference_sft import configure_group_reference_sft
        configure_group_reference_sft()
    rows=common.load_rows(checkpoint['protocol']['train_file'])
    assert len(rows)==24000
    order=list(range(len(rows)));random.Random(0).shuffle(order)
    counts=Counter();families=Counter();images=defaultdict(set);unique_rows=defaultdict(set);curriculum_rows=defaultdict(set)
    missing_source_IDs=defaultdict(set);missing_source_exposures=Counter()
    for index in order[:receipt['seen']]:
        row=rows[index];bucket=row['supervision']['tracking_task_bucket']
        counts[bucket]+=1;families[bucket+':'+row['task']]+=1
        if 'source_row_id' in row['supervision']:
            unique_rows[bucket].add(row['supervision']['source_row_id'])
        else:
            missing_source_IDs[bucket].add(row['id']);missing_source_exposures[bucket]+=1
        curriculum_rows[bucket].add(row['id'])
        assert row.get('media')
        images[bucket].add(str(Path(row['media'][0]['path']).resolve()))
    assert sum(counts.values())==receipt['seen'] and set(counts)=={'mot','rmot','ovmot'}
    assert not torch.cuda.is_initialized()
    result=dict(status='ACTUAL_SHARED_TRAINING_CHECKPOINT_AND_INPUT_EXPOSURE_VERIFIED',
        training_directory=str(training),expected_initial_step=args.initial_step,
        step=args.step,actual_questions_consumed=receipt['seen'],planned_updates=6000,
        saved_full_state=receipt,all320_FP32_parameters_and_both_full_Adam_moments_finite=True,
        FP32_parameters_checked=elements,Adam_moment_elements_checked=moment_elements,
        all320_Adam_steps_and_scheduler_equal_actual_step=True,epoch_and_cursor_exact=True,
        initialization=initial,protocol=checkpoint['protocol'],latest_logged_update=records[-1],
        finite_logged_update_records_checked=len(records),
        task_counts_from_actual_index_shuffle=dict(counts),question_families=dict(families),
        unique_consumed_source_row_IDs={k:len(v) for k,v in unique_rows.items()},
        consumed_questions_without_source_row_ID=dict(missing_source_exposures),
        unique_original_row_IDs_without_source_row_ID={k:len(v) for k,v in missing_source_IDs.items()},
        unique_consumed_curriculum_row_IDs={k:len(v) for k,v in curriculum_rows.items()},
        unique_consumed_first_media_source_paths={k:len(v) for k,v in images.items()},
        exposure_scope='Measured first-source-media paths among actual consumed rows; query repeats and question rows are not independent video frames or tracks.',
        full_state_retention='Atomic latest state validated at this milestone; later saves replace latest state. The numbered BF16 checkpoint is retained.',
        retained_BF16_checkpoint=str(training/f'ckpt-step{args.step}'),
        no_GPU_or_training_updates_by_validator=True,
        complete6000_training=args.step==6000,new_tracking_metrics_available=False,
        combined_registered_feature_upper_bytes=args.registered_feature_upper,global_temporary_feature_budget_bytes=26843545600,
        metadata_reservation='Small scalar proof uses existing audit/report reservation; no new image/ROI bank or model-state copy.',
        full_goal_complete=False)
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
