"""Audit legal Full fit rollout Q3 labels against the final assigned identity."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    original_source = ROOT/'outputs/groundjev/training_data/native_ground_own_history_v1'
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=original_source)
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/groundjev/diagnostics/full_fit_update_supervision_v1')
    parser.add_argument('--eligible-mixture',type=Path)
    args = parser.parse_args()
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from locatemot.groundjev.decision import overlap
    from tools.groundjev.prepare_tracking_decisions import PROTOCOL, annotations
    source = args.source.resolve()
    rollout_summary = json.loads((source/'summary.json').read_text())
    mixture = args.eligible_mixture
    if mixture is None and source == original_source.resolve():
        mixture = ROOT/'outputs/groundjev/training_data/onejev_native_mixture200_v2_family_matched/train.jsonl'
    updates, grouped = {}, defaultdict(dict)
    for line in (source/'train.jsonl').open():
        row = json.loads(line)
        if row['task'] != 'memory_update':
            continue
        match = re.fullmatch(r'(.+)_(\d+)_batch(\d+)_update_(\d+)',row['id'])
        name,frame,batch,local = match.groups()
        key = name,int(frame),int(batch),int(local)
        updates[key] = row
        grouped[key[:2]][key[2]] = row['state']['candidates']
    counts, examples, anchors, prior_ids, previous = Counter(), [], {}, set(), None
    specs = [('mot17_half.json','MOT17-02-FRCNN'),('dancetrack.json','dancetrack0001'),('mot20_half.json','MOT20-01')]
    protocols = {}
    for file,name in specs:
        protocol = json.loads((PROTOCOL/file).read_text())
        seq = next(s for s in protocol['sequences'] if s['sequence']==name)
        allowed = {int(Path(f).stem) for f in seq['train_frames']}
        protocols[name] = allowed,annotations(seq,allowed)
    checked = set()
    for line in (source/'native_rollout.jsonl').open():
        row = json.loads(line)
        name,frame = row['sequence'],row['frame']
        allowed,truth = protocols[name]
        assert frame in allowed
        if previous != name:
            previous,anchors,prior_ids = name,{},set()
        before = dict(anchors)
        gt,candidates = truth[frame],row['candidates']
        matched = {}
        if gt and candidates:
            pairwise = np.asarray([[overlap(c['box'],g['box']) for g in gt] for c in candidates])
            rr,cc = linear_sum_assignment(-pairwise)
            matched = {int(i):gt[int(j)] for i,j in zip(rr,cc) if pairwise[i,j]>=.5}
        events = {e['candidate']:e for e in row['state']['events'] if 'candidate' in e}
        offset = 0
        for batch,group in sorted(grouped[name,frame].items()):
            assert [c['box'] for c in group] == [c['box'] for c in candidates[offset:offset+len(group)]]
            for local in range(len(group)):
                key = name,frame,batch,local
                training,event,target = updates[key],events[offset+local],matched.get(offset+local)
                checked.add(key)
                gid = target['gt_id'] if target else None
                available = any(gid is not None and before.get(t)==gid for t in event['decision']['shortlisted_ids'])
                known = any(gid is not None and before.get(t)==gid for t in prior_ids)
                proxy = bool(target and target['visibility']>=.6 and not (known and not available))
                assert training['target'] == {'yes' if proxy else 'no':1.}
                counts['native_candidate_Q3_labels_reproduced'] += 1
                assigned = event.get('track_id')
                if assigned is None:
                    category = 'no_final_identity'
                elif event['action']=='NEW':
                    category = 'NEW_initialization'
                elif gid is None:
                    category = 'unmatched_not_universal_negative_truth'
                elif before.get(assigned) is None:
                    category = 'unknown_birth_anchor'
                elif before[assigned] == gid:
                    category = 'known_consistent_existing_identity'
                else:
                    category = 'known_conflicting_existing_identity'
                counts[category] += 1
                counts[category+'_Q3_yes'] += proxy
                if assigned is not None and event['action'] != 'NEW':
                    counts[category+'_actual_UPDATE'] += event.get('memory_action') == 'UPDATE'
                if category=='known_conflicting_existing_identity' and proxy and len(examples)<12:
                    examples.append({'sequence':name,'fit_frame':frame,'candidate':offset+local,
                        'training_row_id':training['id'],'candidate_GT_identity':gid,
                        'assigned_ID':assigned,'immutable_birth_GT_identity':before[assigned],
                        'original_Q3_target':training['target'],'actual_memory_action':event.get('memory_action'),
                        'question':training['question']['instructions']})
            offset += len(group)
        assert offset == len(candidates)
        for i,event in events.items():
            if event['action']=='NEW':
                anchors[event['track_id']] = matched[i]['gt_id'] if i in matched else None
        prior_ids = set(row['state']['persistent_ids'])
        counts['frames'] += 1
    assert checked == set(updates) and counts['frames']==48 and len(checked)==1184
    original = {r['id']:r for r in updates.values()}
    retained = set()
    if mixture is not None:
        for line in mixture.open():
            row = json.loads(line)
            if row['task']!='memory_update' or not row['id'].startswith('native_'):
                continue
            match = re.fullmatch(r'native_(.+)_idweight\d+_familycycle\d+',row['id'])
            base = original[match[1]]
            assert row['target']==base['target'] and row['question']==base['question'] and row['state']==base['state']
            retained.add(base['id'])
            counts['actual_family_matched_eligible_native_Q3_rows'] += 1
        assert retained == set(original)
        if source == original_source.resolve():
            assert counts['actual_family_matched_eligible_native_Q3_rows']==2460
    report = {'status':'FULL_NATIVE_FIT_OBSERVATION_Q3_VERSUS_ASSIGNED_IDENTITY_AUDITED',
        'source':str(source),'checkpoint':rollout_summary['checkpoint'],
        'eligible_mixture':str(mixture) if mixture else None,
        'counts':dict(counts),'first_known_conflicting_positive_examples':examples,
        'actual_existing_identity_assignments':sum(counts[k] for k in
            ('known_consistent_existing_identity','known_conflicting_existing_identity',
             'unknown_birth_anchor','unmatched_not_universal_negative_truth')),
        'all_original_Q3_labels_reproduced':True,
        'all_original_Q3_labels_and_actual_family_matched_copies_reproduced':True if mixture else None,
        'original_data_checkpoints_or_predictions_modified':False,'new_model_calls':False,
        'GT_access':'Registered legal fit frames only; no validation/test supervision or actor state correction.',
        'interpretation':'Original Q3 asks whether the candidate is a reliable observation. Its labels '
            'use visibility and correct-option availability, not the identity actually assigned after '
            'global matching. Counts explicitly report actual existing-ID assignments, known '
            'consistent/conflicting writes and unknown events in the selected actor trajectory. '
            'Identity/lifecycle target rows still exist; absence of actual assignments '
            'does not mean the training corpus contains no association targets. Unknown birth '
            'anchors remain unknown. This eligible '
            'corpus audit does not prove actual600-prefix exposure or a causal source of validation failure. '
            'A repair needs an explicit selected-identity question and matched training/inference, '
            'rather than silently replacing labels under the old question.'}
    if source == original_source.resolve():
        report['actual_existing_identity_assignments_in_original_native_actor_rollout'] = report['actual_existing_identity_assignments']
    out = args.out
    out.mkdir(parents=True,exist_ok=False)
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'counts':report['counts']}))


if __name__ == '__main__':
    main()
