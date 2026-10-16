"""Versioned whole-frame entity/action labels over all43 immutable MOT TRAIN units."""
import argparse
from collections import Counter,defaultdict
import csv
import gzip
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def frame_support(candidates,truth):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from locatemot.groundjev.decision import overlap
    # Reconstruct the original exporter's float64/Python overlap arithmetic.
    # Float32 geometry can change a Hungarian tie and its immutable birth label.
    matrix=np.asarray([[overlap(c['box'],g['box']) for g in truth] for c in candidates],
        dtype=np.float64).reshape(len(candidates),len(truth))
    matched={}
    if len(candidates) and len(truth):
        rr,cc=linear_sum_assignment(-matrix)
        matched={int(i):int(j) for i,j in zip(rr,cc) if matrix[i,j]>=.5}
    support=[np.flatnonzero(row>=.5).tolist() for row in matrix]
    return matrix,matched,support


def main():
    import numpy as np
    from locatemot.groundjev.global_competition import sparse_competition
    from tools.groundjev.analyze_tracking_events import load_truth
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text());source=Path(plan['immutable_source_root'])
    out=ROOT/plan['output'];out.mkdir(parents=True,exist_ok=True)
    storage=json.loads((out/'storage_validation.json').read_text())
    assert storage['conservative_temporary_total']+2097152<26843545600
    index=json.loads((source/plan['source_index']).read_text());prior=json.loads((source/plan['source_competition_audit']).read_text())
    destination=out/'labels_v1.csv.gz';assert not destination.exists()
    units={};total=Counter()
    columns=['unit','frame','candidate','support_GT_ids','unique_GT','original_global_winner','original_IGNORE',
        'admission','entity_support','winner_for_unique_GT','full_competitor_indices','all_competitors_in_sparse_neighborhood',
        'confirmed_live_GT_identity_exists','correct_live_track_ids','source_topK_contains_correct_ID',
        'action','assigned_actor_ID','assigned_birth_GT','assigned_write_label','unknown_reason']
    with gzip.open(destination,'wt',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=columns);writer.writeheader()
        for name,unit in index['units'].items():
            if unit['task']!='mot':continue
            manifest=json.loads((source/unit['manifest']).read_text())
            assert manifest['scope']=='registered_train_only_native_source_not_benchmark_evaluation'
            manifest['gt_path_for_evaluation_only']=manifest.get('gt_path_for_evaluation_only',manifest.get('gt_path'))
            truth,_=load_truth(manifest);folder=Path(unit['full_source_rows']).parent
            with gzip.open(folder/'candidate_registry.jsonl.gz','rt') as f:registry={(r['frame'],r['candidate']):r for r in map(json.loads,f)}
            anchors={};prior_ids=set();seen_GT=set();counts=Counter();observed=0
            with (source/unit['original_run']/'frame_decisions.jsonl').open() as traces:
                for row in map(json.loads,traces):
                    frame=row['frame'];candidates=row['candidates'];gt=truth[frame]
                    assert frame==manifest['frame_numbers'][observed] and row['image_path']==manifest['images'][observed]
                    matrix,matched,support=frame_support(candidates,gt);winners={j:i for i,j in matched.items()}
                    features_path=source/unit['frames'][str(frame)]['feature_path']
                    with np.load(features_path,allow_pickle=False) as f:features=f['regions']
                    assert features.shape==(len(candidates),2560) and features.dtype==np.float32
                    edges,_=sparse_competition([c['box'] for c in candidates],features,
                        plan['competition_neighbors_each_metric'],plan['competition_score_block_rows'])
                    neighbors=defaultdict(set)
                    for a,b in edges:neighbors[int(a)].add(int(b))
                    before=dict(anchors);events={e['candidate']:e for e in row['state']['events'] if 'candidate' in e}
                    assert len(events)==len(candidates)
                    for i,c in enumerate(candidates):
                        r=registry[frame,i];supported=support[i];unique=gt[supported[0]]['identity'] if len(supported)==1 else None
                        old_gid=gt[matched[i]]['identity'] if i in matched else None
                        assert r['posthoc_GT_identity']==old_gid
                        rivals={int(j) for g in supported for j in np.flatnonzero(matrix[:,g]>=.5) if j!=i}
                        complete=rivals<=neighbors[i];winner=bool(len(supported)==1 and winners.get(supported[0])==i)
                        invalid=c['box'][2]<=c['box'][0] or c['box'][3]<=c['box'][1]
                        admission='BACKGROUND_OR_INVALID' if invalid else 'UNKNOWN_SUPERVISION'
                        if unique is not None:admission='VALID_OBJECT' if winner else 'DUPLICATE_COMPETITOR' if supported[0] in winners else 'UNKNOWN_SUPERVISION'
                        existing=[tid for tid in prior_ids if unique is not None and before.get(tid)==unique]
                        top=events[i]['decision'].get('shortlisted_ids',[]);covered=bool(set(existing)&set(top))
                        unknown='';action='UNKNOWN'
                        if admission=='BACKGROUND_OR_INVALID':action='REJECT_INVALID'
                        elif admission=='DUPLICATE_COMPETITOR':action='REJECT_DUPLICATE'
                        elif admission=='UNKNOWN_SUPERVISION':unknown='multiple_GT_support' if len(supported)>1 else 'uncertified_unmatched_or_unassigned_support'
                        elif existing:
                            if covered:action='MATCH_OR_REACTIVATE'
                            else:unknown='correct_live_identity_omitted_from_TopK'
                        elif unique in seen_GT:unknown='previous_GT_observed_without_confirmed_live_birth_anchor'
                        elif any(before.get(tid) is None for tid in prior_ids):unknown='unbound_live_birth_may_conflict_with_NEW'
                        else:action='NEW'
                        write=r['assigned_update_target']
                        if unique is None:write=None
                        counts['candidates']+=1;counts['admission:'+admission]+=1;counts['action:'+action]+=1
                        counts['original_IGNORE']+=i not in matched
                        counts['GT_supported_original_IGNORE']+=i not in matched and bool(supported)
                        counts['confirmed_existing_candidates']+=bool(existing)
                        counts['confirmed_existing_TopK_covered']+=covered
                        counts['certified_assigned_write_labels']+=write is not None
                        counts['GT_supported_competitor_sets']+=bool(rivals)
                        counts['complete_sparse_competitor_sets']+=bool(rivals) and complete
                        outside=[winners[g] for g in supported if g in winners and registry[frame,winners[g]]['ordinary_group']!=r['ordinary_group']]
                        inside=[winners[g] for g in supported if g in winners and registry[frame,winners[g]]['ordinary_group']==r['ordinary_group']]
                        if i not in matched and outside and not inside:
                            counts['original_4804_cross_group_cases']+=1
                            counts['cross_group_any_winner_retrieved']+=bool(set(outside)&neighbors[i])
                            counts['cross_group_complete_competitor_set_retrieved']+=complete
                        writer.writerow(dict(unit=name,frame=frame,candidate=i,support_GT_ids='|'.join(str(gt[j]['identity']) for j in supported),
                            unique_GT=unique,original_global_winner=i in matched,original_IGNORE=i not in matched,admission=admission,
                            entity_support=bool(supported),winner_for_unique_GT=winner,full_competitor_indices='|'.join(map(str,sorted(rivals))),
                            all_competitors_in_sparse_neighborhood=complete,confirmed_live_GT_identity_exists=bool(existing),
                            correct_live_track_ids='|'.join(map(str,existing)),source_topK_contains_correct_ID=covered,action=action,
                            assigned_actor_ID=r['actual_actor_identity'],assigned_birth_GT=r['posthoc_immutable_birth_GT_identity'],
                            assigned_write_label=write,unknown_reason=unknown))
                    for i,event in events.items():
                        if event['action']=='NEW':
                            anchors[event['track_id']]=gt[support[i][0]]['identity'] if len(support[i])==1 and i in matched else None
                    prior_ids=set(row['state']['persistent_ids'])
                    seen_GT.update(g['identity'] for g in gt);observed+=1;counts['frames']+=1;counts['sparse_edges']+=len(edges)
            assert observed==len(manifest['images']) and counts['candidates']==len(registry)
            units[name]=dict(counts);total.update(counts)
            print(json.dumps(dict(status='WHOLE_FRAME_LABEL_AUDIT_UNIT_COMPLETE',unit=name,frames=counts['frames'],candidates=counts['candidates'])),flush=True)
    assert len(units)==43 and total['frames']==1056 and total['candidates']==48051
    assert total['original_IGNORE']==7992 and total['GT_supported_original_IGNORE']==5832
    assert total['original_4804_cross_group_cases']==prior['train_totals']['GT_supported_IGNORE_with_only_outside_winners']==4804
    report=dict(status='GLOBAL_IDENTITY_V1_ALL43_WHOLE_FRAME_TRAIN_LABELS_AND_COMPETITION_AUDITED',
        label_version=plan['label_version'],totals=dict(total),units=units,labels=str(destination),
        unknown_supervision_not_invented_background=True,TopK_misses_never_relabelled_NEW=True,
        retirement_supervision='No unknown birth becomes permanent-exit target; RETIRED is a fixed registered timeout, annotation units retained.',
        sparse_competition='Up to96 directed neighbors per candidate,32xN score blocks. Counts distinguish winner access from complete GT-supported competitor-set access; no full-coverage claim if either count is below4804.',
        original_data_or_actor_histories_modified=False,new_feature_bank_bytes=0,new_GPU_processes=0,full_goal_complete=False)
    (out/'label_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    assert destination.stat().st_size<1024*1024
    print(json.dumps(dict(status=report['status'],totals=report['totals'],compressed_label_bytes=destination.stat().st_size)),flush=True)


if __name__=='__main__':main()
