"""Complete matched native three-task courses; CPU input plans, no GPU training."""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'external/OneJev')]


def read(path):
    with gzip.open(path,'rt') as f:return [json.loads(line) for line in f]


def main():
    from tools.groundjev.group_reference_rows import hydrate
    from tools.groundjev.run_source_manifest_sft import INDEXED_PROTOCOL
    from train import common
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    plan=json.loads(args.plan.read_text());out=ROOT/plan['output'];out.mkdir(parents=True,exist_ok=False)
    (out/'registration.json').write_text(json.dumps(plan,indent=2)+'\n')
    component=ROOT/plan['paired_component']
    proof=json.loads((component/'completion_validation.json').read_text())
    assert proof['status']=='PAIRED2882_RMOT_SUPERVISION_COMPONENT_AND_BOTH_NATIVE_MM_CPU_PLANS_COMPLETE'
    old=read(ROOT/plan['original_curriculum']);assert len(old)==24000
    bucket=lambda row:row['supervision']['tracking_task_bucket']
    assert Counter(map(bucket,old))==dict(mot=8000,rmot=8000,ovmot=8000)
    rmot=[i for i,r in enumerate(old) if bucket(r)=='rmot'];removed=set(rmot[:2882])
    kept=[i for i in range(24000) if i not in removed];assert len(kept)==21118
    shared=[old[i] for i in kept];assert Counter(map(bucket,shared))==dict(mot=8000,rmot=5118,ovmot=8000)
    physical=read(component/'physical_refs.jsonl.gz');semantic=read(component/'semantic_refs.jsonl.gz')
    assert len(physical)==len(semantic)==2882
    assert [r['id'] for r in physical]==[r['id'] for r in semantic]
    all_ids=[r['id'] for r in shared+physical];assert len(set(all_ids))==24000
    for local,(p,m) in enumerate(zip(physical,semantic)):
        assert bucket(p)==bucket(m)=='rmot'
        a,b=hydrate(p,ROOT),hydrate(m,ROOT);index=len(shared)+local
        assert a['state']==b['state'] and a['media']==b['media']
        render=lambda row:common.render_example(row,lambda state,suffix:state,index,seed=index,augment=True)
        assert render(a).prompt==render(b).prompt==a['state']
    for arm in ('physical','semantic'):
        directory=out/arm;directory.mkdir()
        manifest=dict(protocol=INDEXED_PROTOCOL,rows=24000,sources=[
            dict(path=plan['original_curriculum'],families=None,rows=21118,source_rows=24000,row_indices=kept),
            dict(path=str((component/(arm+'_refs.jsonl.gz')).relative_to(ROOT)),families=None,rows=2882)])
        path=directory/'train.sources.json';path.write_text(json.dumps(manifest,separators=(',',':'))+'\n')
        config=json.loads((ROOT/plan['base_CPU_config']).read_text())
        config.update(model=str(ROOT/plan['initial_checkpoint']),train=str(path),device='cpu',max_steps=0,
            name='rmot_'+arm+'_24000_CPU_plan',runs_dir=str(out/'CPU_plan'),save_every=1000)
        (directory/'CPU_config.json').write_text(json.dumps(config,indent=2)+'\n')
    order=list(range(24000));random.Random(0).shuffle(order)
    expected=Counter(r['task'] for r in shared+physical)
    report=dict(status='PAIRED24000_THREE_TASK_COURSES_SHARED_ROWS_AND_RENDERED_PREFIXES_VALIDATED',
        rows_each=24000,common_original_rows=21118,paired_questions_each=2882,
        physical_question_families=dict(expected),semantic_question_families=dict(Counter(r['task'] for r in shared+semantic)),
        task_rows_each=dict(mot=8000,rmot=8000,ovmot=8000),
        all_common_source_rows_order_and_IDs_retained=True,all_paired_source_questions_and_targets_retained=True,
        all2882_paired_state_media_and_native_rendered_prefixes_equal_at_actual_global_indices=True,
        same_per_update_question_position_sets=6000,
        paired_questions_per_first1200_updates=sum(i>=21118 for i in order[:4800]),
        native_microbatch_length_order_within_updates_may_differ=True,token_or_FLOP_matched=False,
        initial_checkpoint_used_for_CPU_tokenizer_only=plan['initial_checkpoint'],
        original_current_training_courses_or_source_files_modified=False,
        new_GPU_or_optimizer_updates=0,full_goal_complete=False)
    (out/'input_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    children=[]
    for arm in ('physical','semantic'):
        directory=out/arm;log=(directory/'CPU_plan.log').open('w')
        command=[sys.executable,str(ROOT/'tools/groundjev/run_broader_shared_onejev.py'),
            '--cpu-plan',str(directory/'CPU_config.json'),'--report',str(directory/'CPU_validation.json')]
        child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        children.append((arm,child,log))
    failures=[]
    for arm,child,log in children:
        code=child.wait();log.close()
        print(json.dumps(dict(arm=arm,CPU_plan_exit_code=code)),flush=True)
        if code:failures.append(dict(arm=arm,code=code,log=str(out/arm/'CPU_plan.log')))
    assert not failures,failures
    reports={a:json.loads((out/a/'CPU_validation.json').read_text()) for a in ('physical','semantic')}
    assert all(r['rows']==24000 and r['dropped_rows']==r['actual_optimizer_steps']==0 and
        r['planned_optimizer_steps']==6000 for r in reports.values())
    complete=dict(status='BOTH_COMPLETE24000_NATIVE_MM_COURSES_VALIDATED_ZERO_DROPS_NOT_TRAINED',
        CPU_plans=reports,input_validation=str(out/'input_validation.json'),
        future_training_or_GPU_queue_launched=False,new_GPU_or_optimizer_updates=0,
        combined_registered_feature_upper_bytes=plan['combined_registered_feature_upper_bytes'],
        global_temporary_feature_budget_bytes=plan['global_temporary_feature_budget_bytes'],
        new_metadata_reservation_bytes=plan['new_metadata_reservation_bytes'],full_goal_complete=False)
    (out/'completion_validation.json').write_text(json.dumps(complete,indent=2)+'\n')
    actual=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    assert actual<plan['new_metadata_reservation_bytes'],actual
    print(json.dumps(dict(status=complete['status'],bytes=actual,CPU_plans=reports)),flush=True)


if __name__=='__main__':main()
