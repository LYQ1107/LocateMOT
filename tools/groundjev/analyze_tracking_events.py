"""Post-hoc association/memory diagnostics; GT never changes inference state."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def analyze(manifest, rows):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from locatemot.groundjev.decision import overlap
    truth = defaultdict(list)
    allowed = set(manifest['frame_numbers'])
    with open(manifest['gt_path_for_evaluation_only']) as source:
        for fields in csv.reader(source):
            frame = int(fields[0])
            if frame not in allowed or int(float(fields[6]))!=1 or int(float(fields[7]))!=1:
                continue
            x,y,w,h = map(float,fields[2:6])
            truth[frame].append({'identity':int(fields[1]),'box':[x,y,x+w,y+h],'visibility':float(fields[8])})
    anchors, prior_ids, last_correct = {}, set(), {}
    counts, diagnostics, correct_masses = Counter(), [], []
    for row in rows:
        gt, candidates = truth[row['frame']], row['candidates']
        pairwise = np.array([[overlap(c['box'],g['box']) for g in gt] for c in candidates])
        matched = {}
        if len(candidates) and len(gt):
            rr,cc = linear_sum_assignment(-pairwise)
            matched = {int(i):gt[int(j)] for i,j in zip(rr,cc) if pairwise[i,j]>=.5}
        counts['gt_observations'] += len(gt)
        counts['candidates'] += len(candidates)
        counts['gt_matched_candidates'] += len(matched)
        before = dict(anchors)
        for event in row['state']['events']:
            if 'candidate' not in event:
                continue
            index = event['candidate']
            target = matched.get(index)
            identity = target['identity'] if target else None
            existing = [tid for tid in prior_ids if identity is not None and before.get(tid)==identity]
            d = event['decision']
            shortlist = d.get('shortlisted_ids',list(prior_ids))
            option_present = bool(set(shortlist)&set(existing))
            tid = event.get('track_id')
            if target:
                if existing:
                    counts['existing_identity_events'] += 1
                    counts['correct_identity_option_present'] += option_present
                    correct = tid in existing
                    counts['correct_existing_associations'] += correct
                    if event['action']=='NEW': counts['existing_identity_recreated_as_NEW'] += 1
                    if row['state']['decision_metadata']['native_probabilities']:
                        correct_masses.append(sum(d['probabilities'].get(str(t),0.) for t in existing))
                else:
                    counts['no_surviving_correct_anchor_events'] += 1
                    correct = event['action']=='NEW'
                if identity in last_correct:
                    counts['correct_to_wrong_events'] += last_correct[identity] and not correct
                    counts['wrong_to_correct_events'] += not last_correct[identity] and correct
                last_correct[identity] = correct
            else:
                correct = event['action']=='IGNORE'
                counts['unmatched_candidate_events'] += 1
                counts['unmatched_candidates_ignored'] += correct
            if event['action']=='NEW':
                anchors[tid] = identity
                counts['memory_initializations_matched_GT' if identity is not None else 'memory_initializations_unmatched_GT'] += 1
            elif event.get('memory_action')=='UPDATE':
                identity_correct = identity is not None and before.get(tid)==identity
                counts['memory_updates_identity_consistent' if identity_correct else 'memory_updates_wrong_identity_or_unmatched'] += 1
                if target and target['visibility']<.6:
                    counts['memory_updates_low_visibility'] += 1
            elif event.get('memory_action')=='HOLD':
                counts['memory_holds'] += 1
            diagnostics.append({'frame':row['frame'],'candidate':index,'matched_gt_identity':identity,
                'action':event['action'],'assigned_track_id':tid,'past_anchor_gt_identity':before.get(tid),
                'correct_surviving_anchor_ids':existing,'correct_option_present':option_present,
                'association_correct_under_fixed_anchor_diagnostic':correct,
                'memory_action':event.get('memory_action')})
        prior_ids = set(row['state']['persistent_ids'])
    n = counts['existing_identity_events']
    return {'scope':manifest['scope'],'frames':len(rows),'counts':dict(counts),
        'correct_option_coverage':counts['correct_identity_option_present']/n if n else None,
        'existing_association_accuracy':counts['correct_existing_associations']/n if n else None,
        'mean_native_correct_identity_option_mass':sum(correct_masses)/len(correct_masses) if correct_masses else None,
        'candidate_GT_recall_at_iou05':counts['gt_matched_candidates']/counts['gt_observations'],
        'diagnostic_policy':'marked pedestrian GT, one-to-one candidate IoU>=.5; immutable track-anchor ownership from birth, not GT correction',
        'unmatched_candidates_note':'includes evaluator-ignore regions; these are unmatched diagnostics, not official FP/NoMatch precision',
        'explicit_NO_MATCH_decision_currently_supported':any(r['state']['decision_metadata'].get('explicit_no_match',False) for r in rows),
        'gt_used_posthoc_only':True,'official_metric_replacement':False,'events':diagnostics}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads(args.video.read_text())
    rows=list(map(json.loads,(args.run/'frame_decisions.jsonl').read_text().splitlines()))
    if [r['frame'] for r in rows]!=manifest['frame_numbers']:
        parser.error('diagnostic requires every registered frame')
    report=analyze(manifest,rows)
    (args.run/'association_events.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='events'},indent=2))


if __name__=='__main__':
    main()
