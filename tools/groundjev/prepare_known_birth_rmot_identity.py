"""Select source-identical RMOT identity/write labels with explicit birth evidence."""
import argparse
from collections import Counter,defaultdict
import gzip
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'external/OneJev')]


def read(path):
    with gzip.open(path,'rt') as file:
        for line in file:yield json.loads(line)


def main():
    import torch
    from tools.groundjev.group_reference_rows import REFERENCE_KEY,request_groups
    from tools.groundjev.streamed_tracking_media import HISTORY_KEY
    from tools.groundjev.run_group_reference_sft import configure_group_reference_sft
    from train import sft
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    assert plan['combined_registered_feature_upper_bytes']<plan['global_temporary_feature_budget_bytes']
    out=ROOT/plan['output'];out.mkdir(parents=True,exist_ok=False)
    (out/'registration.json').write_text(json.dumps(plan,indent=2)+'\n')
    source=ROOT/plan['source'];export=ROOT/plan['export']
    produced=json.loads((source/'owner_state.json').read_text());exported=json.loads((export/'owner_state.json').read_text())
    assert produced['status']=='ALL1814_FIT_PHYSICAL_FRAMES_18_NATIVE_QUERY_ROLLOUTS_COMPLETE'
    assert exported['status']=='ALL18_NATIVE_RMOT_QUERY_ROLLOUTS_FULL_V3_GROUPS_AND_CPU_PLANS_VALIDATED'
    assert len(produced['completed'])==len(exported['completed'])==18
    counts=Counter();age_buckets=Counter();images=set();unit_counts={};kept=[]
    for name,unit in exported['completed'].items():
        directory=Path(unit['row_references']).parent
        rows=defaultdict(dict)
        for row in read(directory/'train_refs.jsonl.gz'):
            ref=row[REFERENCE_KEY];rows[ref['group']][ref['question']]=row
        groups=request_groups(str(directory/'groups.jsonl.gz'))
        registry=defaultdict(lambda:defaultdict(list))
        for record in read(directory/'candidate_registry.jsonl.gz'):
            registry[record['frame']][record['ordinary_group']].append(record)
        manifest=json.loads((ROOT/plan['scope']/'manifests'/f'{name}.json').read_text())
        birth,birth_frame,active={}, {},set();local_counts=Counter()
        with (Path(produced['completed'][name]['run'])/'frame_decisions.jsonl').open() as trace:
            for line in trace:
                frame_row=json.loads(line);frame=frame_row['frame']
                events={e['candidate']:e for e in frame_row['state']['events'] if 'candidate'in e}
                before={tid:birth[tid] for tid in active}
                for group_id,records in registry[frame].items():
                    records.sort(key=lambda r:r['candidate']);group=groups[group_id]
                    actual=events[records[0]['candidate']]['decision']['shortlisted_ids']
                    aliases=[str(t['id']) for t in group['state']['historical_tracks']]
                    assert len(actual)==len(aliases) and set(actual)<=active
                    owners={alias:before[tid] for alias,tid in zip(aliases,actual)}
                    aliased={alias:tid for alias,tid in zip(aliases,actual)}
                    crop_history=next((m[HISTORY_KEY] for m in group['media'] if HISTORY_KEY in m),[])
                    for item in crop_history:
                        assert item['anchor']['frame']==birth_frame[aliased[str(item['id'])]]
                    for index,record in enumerate(records):
                        assert events[record['candidate']]['decision']['shortlisted_ids']==actual
                        gid=record['posthoc_GT_identity']
                        row=rows[group_id][f'match_{index}'];answer=next(iter(row['target']))
                        correct=[alias for alias in aliases if gid is not None and owners[alias]==gid]
                        eligible=False;target_tid=None
                        if answer.isdigit():
                            assert gid is not None and answer in correct
                            if len(correct)==1:eligible=True;target_tid=aliased[answer]
                            else:local_counts['multiple_correct_input_anchors_identity_rows_not_selected']+=1
                        elif answer=='NO_MATCH':
                            surviving=[tid for tid,owner in before.items() if gid is not None and owner==gid]
                            assert surviving and not correct
                            eligible=True;target_tid=min(surviving)
                        if eligible:
                            span=(frame-birth_frame[target_tid])/manifest['fps'];assert span>=0
                            bucket='under1s' if span<1 else '1to3s' if span<=3 else '3to10s' if span<=10 else '10to30s' if span<=30 else 'over30s'
                            age_buckets['identity:'+bucket]+=1
                            selected=dict(row,supervision=dict(row['supervision'],
                                immutable_birth_contract_certified=True,unique_correct_input_anchor=answer!='NO_MATCH',
                                source_row_id=row['id'],causal_correct_anchor_age_seconds=span,
                                GT_labels_remain_outside_native_state=True))
                            kept.append(selected);local_counts['identity:'+('EXISTING' if answer.isdigit() else answer)]+=1
                            images.add(frame_row['image_path'])
                        assigned_id=group_id+'_assigned';question=f'assigned_update_{index}'
                        assigned_row=rows.get(assigned_id,{}).get(question)
                        if assigned_row is not None:
                            alias=str(record['selected_input_alias']);assert gid is not None and owners[alias] is not None
                            state=groups[assigned_id]['state']
                            assert str(state['final_assignment']['selected_existing_identity_by_candidate'][str(index)])==alias
                            quality=int(next(iter(rows[group_id][f'reliability_{index}']['target'])))
                            expected='yes' if owners[alias]==gid and quality>=2 else 'no'
                            assert assigned_row['target']=={expected:1.} and record['assigned_update_target']==expected
                            span=(frame-birth_frame[aliased[alias]])/manifest['fps'];assert span>=0
                            bucket='under1s' if span<1 else '1to3s' if span<=3 else '3to10s' if span<=10 else '10to30s' if span<=30 else 'over30s'
                            age_buckets['assigned_update:'+bucket]+=1
                            kept.append(dict(assigned_row,supervision=dict(assigned_row['supervision'],
                                immutable_birth_contract_certified=True,known_current_and_assigned_birth_identity=True,
                                source_row_id=assigned_row['id'],causal_assigned_anchor_age_seconds=span,
                                reliability_is_original_IoU_proxy_not_visibility=True)))
                            local_counts['assigned_identity_memory_update:'+expected]+=1;images.add(frame_row['image_path'])
                    local_counts['native_candidate_registry_rows_checked']+=len(records)
                for records in registry[frame].values():
                    for record in records:
                        if record['actual_actor_action']=='NEW':
                            tid=record['actual_actor_identity'];assert tid not in birth
                            birth[tid]=record['posthoc_GT_identity'];birth_frame[tid]=frame
                active=set(frame_row['state']['persistent_ids'])
                assert active<=birth.keys()
        counts.update(local_counts);unit_counts[name]=dict(local_counts)
    assert len({r['id'] for r in kept})==len(kept) and kept
    train=out/'train_refs.jsonl.gz'
    with gzip.open(train,'wt') as target:
        for row in kept:target.write(json.dumps(row)+'\n')
    torch.set_num_threads(4);stateful=configure_group_reference_sft()
    config=json.loads((ROOT/'configs/groundjev/onejev_unified600_native_v2.json').read_text())
    config.update(model=str(ROOT/'outputs/groundjev/training/onejev_unified600_native_v2/ckpt-step600'),
        train=str(train),device='cpu',max_steps=0,name='native_CPU_plan',runs_dir=str(out/'CPU_plan'))
    config_path=out/'CPU_plan_config.json';config_path.write_text(json.dumps(config,indent=2)+'\n')
    planned={}
    def capture(bundle,rows,args,world):
        batches,dropped,lengths=sft.plan_epoch(bundle,rows,args,0,0,1)
        assert world==1 and len(rows)==len(kept) and dropped==0
        assert sorted(i for b in batches for i in b)==list(range(len(rows)))
        assert not torch.cuda.is_initialized()
        planned.update(rows=len(rows),dropped_rows=0,max_tokens=max(lengths),planned_tokens=sum(lengths))
        return 0
    sft.plan_only=capture
    assert sft.main(['--config',str(config_path),'--plan-only'])==0
    report=dict(status='KNOWN_BIRTH_RMOT_IDENTITY_AND_ASSIGNED_WRITE_NATIVE_COMPONENT_VALIDATED',
        source_rollouts=18,source_physical_frames=1814,actual_filtered_unique_media_paths=len(images),
        selected_rows=len(kept),counts=dict(counts),unit_counts=unit_counts,
        correct_or_assigned_anchor_age_seconds_buckets=dict(age_buckets),CPU_plan=planned,
        anchor_age_definition='Registered source time since immutable actor birth; not an occlusion gap or time since last observation.',
        original_native_group_states_media_questions_and_targets_unchanged=True,
        source_all18_actor_state_output_and_pixel_validation_reused=True,
        new_Ground_or_ROI_or_GPU_or_optimizer_calls=0,
        unknown_birth_and_absent_correspondence_not_fabricated_as_NEGATIVE=True,
        dataset_is_registered_IoU_and_birth_contract_not_exhaustive_physical_truth=True,
        current459027_or6000_inputs_changed=False,full_goal_complete=False)
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<plan['component_cap_bytes']
    print(json.dumps({k:v for k,v in report.items() if k!='unit_counts'}),flush=True)


if __name__=='__main__':main()
