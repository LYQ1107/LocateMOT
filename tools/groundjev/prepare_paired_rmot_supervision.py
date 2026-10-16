"""Paired actual-history RMOT physical versus membership native input component."""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'external/OneJev')]


def read(path):
    with gzip.open(path,'rt') as f:yield from map(json.loads,f)


def cpu_plan(config,report):
    import torch
    from tools.groundjev.run_group_reference_sft import configure_group_reference_sft
    from train import sft
    torch.set_num_threads(4);configure_group_reference_sft()
    def capture(bundle,rows,args,world):
        batches,dropped,lengths=sft.plan_epoch(bundle,rows,args,0,0,1)
        assert world==1 and len(rows)==2882 and dropped==0
        assert sorted(i for b in batches for i in b)==list(range(2882))
        assert not torch.cuda.is_initialized()
        result=dict(status='ALL2882_PAIRED_RMOT_NATIVE_MM_ROWS_VALIDATED_NO_TRAINING',
            rows=len(rows),dropped_rows=dropped,max_tokens=max(lengths),planned_tokens=sum(lengths),
            question_families=dict(Counter(r['task'] for r in rows)),native_multimodal_inputs=True,
            state_format='Fixed source-equivalent pretty JSON in both paired arms; native question option permutation retained.',
            actual_optimizer_updates=0,full_goal_complete=False)
        report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);return 0
    sft.plan_only=capture
    assert sft.main(['--config',str(config),'--plan-only'])==0


def prepare(plan):
    from tools.groundjev.group_reference_rows import REFERENCE_KEY
    from train import common
    from qev.prompt import render_question
    source=ROOT/plan['physical_source'];proof=json.loads((source/'validation.json').read_text())
    assert proof['status']=='KNOWN_BIRTH_RMOT_IDENTITY_AND_ASSIGNED_WRITE_NATIVE_COMPONENT_VALIDATED'
    assert proof['selected_rows']==2882 and proof['original_native_group_states_media_questions_and_targets_unchanged']
    physical=list(read(source/'train_refs.jsonl.gz'));assert len(physical)==2882
    out=ROOT/plan['output'];out.mkdir(parents=True,exist_ok=False)
    (out/'registration.json').write_text(json.dumps(plan,indent=2)+'\n')
    target_groups={};paired_rows=[];loaded_path=None;groups=None;original_rows=None
    counts=Counter();format_confounds=0
    for index,row in enumerate(physical):
        ref=row[REFERENCE_KEY]
        if ref['path']!=loaded_path:
            loaded_path=ref['path'];folder=(ROOT/loaded_path).parent
            groups={g['id']:g for g in read(ROOT/loaded_path)}
            original_rows={(r[REFERENCE_KEY]['group'],r[REFERENCE_KEY]['question']):r for r in read(folder/'train_refs.jsonl.gz')}
        group=groups[ref['group']];question=group['questions'][ref['question']]
        local=int(ref['question'].rsplit('_',1)[1]);ordinary=ref['group'].removesuffix('_assigned')
        member_key=f'membership_{local}';member=original_rows[ordinary,member_key]
        member_question=groups[ordinary]['questions'][member_key]
        assert row['target']==original_rows[ref['group'],ref['question']]['target']
        assert group['state']['task']['task']=='rmot'
        assert group['media']==groups[ordinary]['media']
        common.target_vector(row['target'],render_question('q',common.build_question(question)).labels)
        common.target_vector(member['target'],render_question('q',common.build_question(member_question)).labels)
        group_key=loaded_path+'::'+ref['group']
        # Source group IDs are already globally unique across these18rollouts.
        if ref['group'] not in target_groups:
            target_groups[ref['group']]=dict(id=ref['group'],state=common.serialize_state(group['state'],False),
                media=group['media'],questions={},source_group=group_key)
        derived=target_groups[ref['group']]
        assert derived['source_group']==group_key
        assert json.loads(derived['state'])==group['state']
        derived['questions'][ref['question']]=question;derived['questions'][member_key]=member_question
        ref_path=str((out/'groups.jsonl.gz').relative_to(ROOT))
        p=dict(row,**{REFERENCE_KEY:dict(path=ref_path,group=ref['group'],question=ref['question'])})
        m=dict(member,id=row['id'],**{REFERENCE_KEY:dict(path=ref_path,group=ref['group'],question=member_key)})
        for item,origin in [(p,row),(m,member)]:
            item['supervision']=dict(item['supervision'],source_row_id=origin['id'],paired_physical_row_id=row['id'],
                original_input_group=group_key,paired_source_question=origin[REFERENCE_KEY],
                explicit_fixed_pretty_state_format=True,GT_labels_remain_outside_model_input=True)
        # Exercise the native row renderer at the paired positions, with different question types.
        raw_p=dict(p,state=group['state'],question=question,media=group['media'])
        raw_m=dict(m,state=group['state'],question=member_question,media=group['media'])
        render=lambda item:common.render_example(item,lambda state,suffix:state,index,seed=index,augment=True).prompt
        format_confounds+=render(raw_p)!=render(raw_m)
        p_input=dict(raw_p,state=derived['state']);m_input=dict(raw_m,state=derived['state'])
        assert render(p_input)==render(m_input)==derived['state']
        assert p_input['media']==m_input['media']==group['media']
        paired_rows.append((p,m));counts['physical:'+row['task']]+=1
        counts['physical_target:'+next(iter(row['target'])) if row['task']!='identity' else 'physical_identity_existing']+=1
        counts['semantic_target:'+next(iter(member['target']))]+=1
    assert len(target_groups)>0 and len({p['id'] for p,_ in paired_rows})==2882
    with gzip.open(out/'groups.jsonl.gz','wt') as f:
        for group in target_groups.values():f.write(json.dumps(group,ensure_ascii=False)+'\n')
    for arm,column in [('physical',0),('semantic',1)]:
        with gzip.open(out/(arm+'_refs.jsonl.gz'),'wt') as f:
            for pair in paired_rows:f.write(json.dumps(pair[column],ensure_ascii=False)+'\n')
        config=json.loads((source/'CPU_plan_config.json').read_text())
        config.update(train=str(out/(arm+'_refs.jsonl.gz')),name=arm+'_CPU_plan',runs_dir=str(out/'CPU_plan'),device='cpu',max_steps=0)
        (out/(arm+'_CPU_config.json')).write_text(json.dumps(config,indent=2)+'\n')
    report=dict(status='PAIRED2882_RMOT_STATE_MEDIA_SOURCE_QUESTIONS_AND_RENDERED_PREFIXES_VALIDATED',
        pairs=2882,request_groups=len(target_groups),counts=dict(counts),
        native_original_dynamic_serialization_would_differ_pairs=format_confounds,
        all_original_physical_questions_and_targets_exact=True,all_original_membership_questions_and_targets_exact=True,
        input_state_JSON_trees_and_media_equal_original_physical_source=True,
        all_native_rendered_state_prefixes_equal_in_fixed_format=True,original_pair_row_order_and_IDs_shared=True,
        actor_histories_or_assignments_corrected_using_GT=False,new_GPU_or_optimizer_updates=0,
        original_pixel_validation_reused=proof['source_all18_actor_state_output_and_pixel_validation_reused'],
        scope='TRAIN selected known-birth subset from18actual query rollouts. Assigned control uses the physical assigned-state context with the original membership question/target. Original physical and semantic question labels unchanged; fixed pretty state strings are an explicit new paired-component formatting rule. Registered IoU/birth-label contract is not exhaustive physical truth. Not a new training experiment or native decoder evaluation.',
        combined_registered_feature_upper_bytes=plan['combined_registered_feature_upper_bytes'],
        global_temporary_feature_budget_bytes=plan['global_temporary_feature_budget_bytes'],full_goal_complete=False)
    (out/'input_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<plan['component_cap_bytes']
    print(json.dumps(report),flush=True)
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path)
    parser.add_argument('--cpu-config',type=Path);parser.add_argument('--report',type=Path)
    args=parser.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    if args.cpu_config:return cpu_plan(args.cpu_config,args.report)
    plan=json.loads(args.plan.read_text());out=prepare(plan)
    for arm in ('physical','semantic'):
        print('VALIDATING_ALL2882_NATIVE_MM_INPUTS_'+arm,flush=True)
        command=[sys.executable,str(Path(__file__).resolve()),'--cpu-config',str(out/(arm+'_CPU_config.json')),'--report',str(out/(arm+'_CPU_validation.json'))]
        with (out/(arm+'_CPU_plan.log')).open('w') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    reports={arm:json.loads((out/(arm+'_CPU_validation.json')).read_text()) for arm in ('physical','semantic')}
    assert all(r['rows']==2882 and r['dropped_rows']==r['actual_optimizer_updates']==0 for r in reports.values())
    complete=dict(status='PAIRED2882_RMOT_SUPERVISION_COMPONENT_AND_BOTH_NATIVE_MM_CPU_PLANS_COMPLETE',
        input_validation=str(out/'input_validation.json'),CPU_plans=reports,source_labels_or_running_courses_modified=False,
        new_GPU_or_optimizer_updates=0,new_ROI_or_image_bank_bytes=0,
        combined_registered_feature_upper_bytes=plan['combined_registered_feature_upper_bytes'],full_goal_complete=False)
    assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<plan['component_cap_bytes']
    (out/'completion_validation.json').write_text(json.dumps(complete,indent=2)+'\n');print(json.dumps(complete),flush=True)


if __name__=='__main__':main()
