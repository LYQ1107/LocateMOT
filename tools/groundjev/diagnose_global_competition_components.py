"""Separate real sparse-graph coverage from learned component admission failures."""
import json
from collections import Counter,defaultdict
from itertools import combinations,zip_longest
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def main():
    import numpy as np
    from tools.groundjev.run_video import ReplayFrames
    from tools.groundjev.analyze_tracking_events import load_truth
    from tools.groundjev.audit_global_identity_labels import frame_support
    from locatemot.groundjev.global_competition import sparse_competition
    root=ROOT/'outputs/groundjev/global_identity_v1';plan=json.loads((ROOT/'configs/groundjev/global_identity_v1.json').read_text());source=Path(plan['immutable_source_root'])
    scope=json.loads((source/plan['evaluation_plan']).read_text())['scopes']['mot_full1203'];manifest=json.loads((source/scope['manifest']).read_text());truth,_=load_truth(manifest)
    runs={'A2':root/'controls/A2_mot_full1203','A4':root/'own_state_controls/A4_own_state_v2_mot_full1203'}
    streams={k:(v/'frame_decisions.jsonl').open() for k,v in runs.items()};replay=ReplayFrames(source/scope['candidates'])
    counts=Counter();arms={k:Counter() for k in runs};examples=[]
    try:
        for number,lines in zip_longest(manifest['frame_numbers'],zip_longest(*streams.values())):
            assert number is not None and lines is not None and all(lines)
            rows={k:json.loads(line) for k,line in zip(streams,lines)};assert all(row['frame']==number for row in rows.values())
            raw=replay.frame(number);candidates=raw['candidates'];_,_,support=frame_support(candidates,truth[number])
            groups=defaultdict(list);gt={}
            for i,s in enumerate(support):
                if len(s)==1:gt[i]=truth[number][s[0]]['identity'];groups[gt[i]].append(i)
            features=np.asarray([c['native_region_feature'] for c in candidates],np.float32).reshape(-1,2560)
            edges,_=sparse_competition(np.asarray([c['box'] for c in candidates],np.float32).reshape(-1,4),features,32)
            neighbors=defaultdict(set)
            for i,j in edges.tolist():neighbors[i].add(j);neighbors[j].add(i)
            duplicate_groups=[g for g in groups.values() if len(g)>1]
            counts['frames']+=1;counts['candidates']+=len(candidates);counts['same_GT_competitor_sets']+=len(duplicate_groups)
            for group in duplicate_groups:
                target=set(group);reached={group[0]};pending=list(reached)
                while pending:
                    for j in neighbors[pending.pop()]&target-reached:reached.add(j);pending.append(j)
                counts['sparse_graph_same_GT_induced_set_connected']+=reached==target
                counts['same_GT_proposal_pairs']+=len(group)*(len(group)-1)//2
                counts['same_GT_proposal_pairs_in_sparse_graph']+=sum(j in neighbors[i] for i,j in combinations(group,2))
            for arm,row in rows.items():
                c=arms[arm];components=row['state']['decision_metadata']['competition_components'];by_candidate={i:j for j,g in enumerate(components) for i in g['candidates']}
                c['predicted_duplicate_components']+=len(components)
                c['explicitly_suppressed_proposals']+=sum(len(g['candidates'])-1 for g in components)
                for component in components:
                    for i,j in combinations(component['candidates'],2):
                        if i not in gt or j not in gt:c['component_pair_annotation_unknown']+=1
                        elif gt[i]==gt[j]:c['component_pair_same_GT']+=1
                        else:c['component_pair_different_GT']+=1
                for group in duplicate_groups:
                    complete=all(i in by_candidate for i in group) and len({by_candidate[i] for i in group})==1
                    c['complete_GT_competitor_sets_in_one_predicted_component']+=complete
                    if arm=='A4' and not complete and len(examples)<6:
                        examples.append(dict(frame=number,GT=gt[group[0]],candidates=group,
                            predicted_components=[by_candidate.get(i) for i in group],
                            actions=[next(e['action'] for e in row['state']['events'] if e.get('candidate')==i) for i in group]))
    finally:
        replay.close()
        for f in streams.values():f.close()
    assert counts['frames']==1203 and counts['candidates']==13664
    result=dict(status='REAL_DT_SPARSE_RETRIEVAL_VERSUS_LEARNED_COMPONENT_ADMISSION_SEPARATED',scope='Original fixed DT1203 proposals, official GT used posthoc only, no inference/training/threshold or source mutation.',graph=dict(counts),arms={k:dict(v) for k,v in arms.items()},failure_examples=examples,
        interpretation='Predicted component pair counts are transitive component closure relations, not logged raw edge-classifier TPR. Unknown annotations are not negatives.',new_GPU_processes=0,new_feature_bank_bytes=0,full_goal_complete=False)
    (root/'DT_sparse_graph_vs_learned_components.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
