"""Validate both completed native plans and explain filtered RMOT exposure."""
from collections import Counter,defaultdict
import gzip
import json
import os
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[2]
RUNTIME_ROOT=Path(os.environ.get('GROUNDJEV_RUNTIME_ROOT','/data2/user/LocateMOT_GroundJEV'))
sys.path[:0]=[str(ROOT),str(ROOT/'external/OneJev')]


def resolve_runtime_path(value):
    """Resolve manifest-relative artifacts from project or registered runtime tree."""
    path=Path(value)
    if path.is_absolute():
        return path
    for base in (ROOT,RUNTIME_ROOT):
        candidate=base/path
        if candidate.exists():
            return candidate
    return ROOT/path


def read(path):
    with gzip.open(path,'rt') as file:
        for line in file:yield json.loads(line)


def main():
    import numpy as np
    from train import sft
    from tools.groundjev.group_reference_rows import REFERENCE_KEY
    out=ROOT/'outputs/groundjev/training/full_source_parallel_cpu_plan_v1'
    serial_root=ROOT/'outputs/groundjev/training/full_joint_source_epoch1_owner_v1'
    parallel=json.loads((out/'CPU_plan_validation.json').read_text())
    serial=json.loads((serial_root/'CPU_plan_validation.json').read_text())
    terminal_candidates = [
        out.with_name(out.name+'_owner.json'),
        RUNTIME_ROOT/'outputs/groundjev/training/full_source_parallel_cpu_plan_v1_owner.json',
    ]
    terminal_path = next((path for path in terminal_candidates if path.is_file()), None)
    if terminal_path is None:
        raise FileNotFoundError('full_source_parallel_cpu_plan_v1_owner.json; checked: '
                                + ', '.join(str(path) for path in terminal_candidates))
    terminal=json.loads(terminal_path.read_text())
    assert terminal['status']=='FULL_NATIVE_PARALLEL_CPU_PLAN_COMPLETE' and terminal['exit_code']==0
    for key in serial:
        if key!='CPU_plan_elapsed_s':assert serial[key]==parallel[key],key
    manifest=json.loads(Path(parallel['native_args']['train_file']).read_text())
    assert manifest==parallel['manifest'] and manifest['rows']==459027
    assert parallel['rows']==459027 and parallel['dropped_rows']==0
    for source,saved in zip(manifest['sources'],parallel['source_files']):
        assert all(source[k]==saved[k] for k in ('path','rows','families'))
        stat=resolve_runtime_path(source['path']).stat()
        assert stat.st_size==saved['bytes'] and stat.st_mtime_ns==saved['mtime_ns']
    lengths=np.load(parallel['length_array'],allow_pickle=False)
    assert lengths.dtype==np.int32 and lengths.shape==(459027,)
    assert int(lengths.min())>0 and int(lengths.max())==serial['maximum_tokens']==27481
    assert int(lengths.sum(dtype=np.int64))==serial['planned_tokens']==5639441074
    order=list(range(len(lengths)));random.Random(0).shuffle(order)
    steps=sft.plan_steps([[i] for i in order],lengths,4,16)
    assert len(steps)==serial['planned_updates']==28690
    assert sorted(i for step in steps for per in step for batch in per for i in batch)==list(range(len(lengths)))
    assert [sum(len(b) for b in rank) for rank in steps[-1]]==[1,1,1,0]
    scope=resolve_runtime_path('outputs/groundjev/training_data/rmot_allfit1814_query18_scope_v1/manifests')
    all_rmot=set();eligible_rmot=set();families_by_image=defaultdict(Counter);unit_counts={}
    for source in manifest['sources']:
        if source['bucket']!='rmot':continue
        path=resolve_runtime_path(source['path']);groups={}
        is_new='rmot_allfit1814_query18_full_v3_groups_v2_shared_gpu7' in source['path']
        if source['families'] is not None:
            for group in read(path.parent/'groups.jsonl.gz'):
                groups[group['id']]=str(Path(group['media'][0]['path']).resolve())
        selected=set();count=0
        for row in read(path):
            image=groups[row[REFERENCE_KEY]['group']] if REFERENCE_KEY in row else str(Path(row['media'][0]['path']).resolve())
            if is_new:families_by_image[image][row['task']]+=1
            if source['families'] is None or row['task'] in source['families']:
                selected.add(image);count+=1
        assert count==source['rows']
        eligible_rmot.update(selected)
        if is_new:
            name=path.parent.name
            video=json.loads((scope/(name+'.json')).read_text())
            raw={str(Path(image).resolve()) for image in video['images']}
            assert selected<=raw;all_rmot.update(raw)
            unit_counts[name]=dict(raw_physical_frames=len(raw),semantic_selected_physical_frames=len(selected))
    assert len(all_rmot)==1814 and eligible_rmot<=all_rmot
    assert len(eligible_rmot)==parallel['planned_unique_first_media_source_paths']['rmot']==1796
    missing=sorted(all_rmot-eligible_rmot)
    assert len(missing)==18
    assert all(set(families_by_image[image])=={'lifecycle'} for image in missing)
    result=dict(status='SERIAL_AND_FOUR_CPU_FULL459027_NATIVE_PLANS_AGREE_AND_FILTERED_EXPOSURE_VALIDATED',
        rows=459027,dropped_rows=0,maximum_tokens=27481,planned_multimodal_tokens=5639441074,
        planned_updates=28690,global_rows_per_update=16,world_size=4,
        all_rows_once_in_same_native_index_shuffle_plan=True,tail_rank_rows=[1,1,1,0],
        serial_and_parallel_all_plan_summaries_exact=True,complete_int32_length_array_int64_sum_valid=True,
        RMOT_raw_physical_frames=1814,RMOT_semantic_input_physical_frames=1796,
        RMOT_missing_semantic_frame_paths=missing,
        RMOT_missing_frame_original_family_counts={image:dict(families_by_image[image]) for image in missing},
        missing_exposure_reason='These18 images have only lifecycle questions in both retained native query histories. Registered semantic-only filtering excludes lifecycle proxies; no membership/presence row exists for these images. All raw frames and original rows remain preserved.',
        per_new_RMOT_unit_exposure=unit_counts,
        planned_unique_first_media_source_paths=parallel['planned_unique_first_media_source_paths'],
        coordinator_terminal=json.loads(resolve_runtime_path(
            'outputs/groundjev/training/validated_CPU_handoff_owner_v1/owner_state.json'
        ).read_text()),
        parallel_terminal_receipt=str(terminal_path),
        original_four_GPU_owner_retained=True,actual_optimizer_updates_not_claimed_by_input_validation=True,
        current_training_inputs_metrics_and_seeds_unchanged=True,full_goal_complete=False)
    (out/'completion_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<4194304
    print(json.dumps({k:v for k,v in result.items() if k not in ('RMOT_missing_semantic_frame_paths','RMOT_missing_frame_original_family_counts','per_new_RMOT_unit_exposure')}),flush=True)


if __name__=='__main__':main()
