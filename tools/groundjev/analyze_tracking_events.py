"""Post-hoc association/memory diagnostics; GT never changes inference state."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def load_truth(manifest):
    truth = defaultdict(list)
    allowed = set(manifest['frame_numbers'])
    query_targets = None
    if manifest['task'] == 'mot':
        with open(manifest['gt_path_for_evaluation_only']) as source:
            for fields in csv.reader(source):
                frame = int(fields[0])
                if frame not in allowed or int(float(fields[6])) != 1 or int(float(fields[7])) != 1:
                    continue
                x, y, w, h = map(float, fields[2:6])
                truth[frame].append({'identity': int(fields[1]), 'box': [x,y,x+w,y+h], 'visibility': float(fields[8])})
    elif manifest['task'] == 'rmot':
        from locatemot.unified.data.legal_scope import assert_legal_video, LEGAL_SPLITS
        video = assert_legal_video(manifest['sequence'], dataset=manifest['dataset'])
        assert video in LEGAL_SPLITS[manifest['dataset']]['validation']
        query = json.loads(Path(manifest['expression_file_for_evaluation_only']).read_text())
        query_targets = query['label']
        width, height = manifest['image_size']
        for frame in manifest['frame_numbers']:
            path = Path(manifest['label_dir_for_evaluation_only'])/f'{frame:06d}.txt'
            for line in path.read_text().splitlines() if path.exists() else []:
                fields = line.split(); x, y, w, h = map(float, fields[2:6])
                truth[frame].append({'identity': int(fields[1]),
                    'box': [x*width,y*height,(x+w)*width,(y+h)*height], 'visibility': None})
    else:
        assert manifest['task'] == 'ovmot'
        annotation = json.loads(Path(manifest['annotation_path_for_evaluation_only']).read_text())
        image_frames = dict(zip(manifest['image_ids'], manifest['frame_numbers']))
        for obj in annotation['annotations']:
            if obj['image_id'] in image_frames:
                x, y, w, h = obj['bbox']
                truth[image_frames[obj['image_id']]].append({'identity': obj['track_id'],
                    'box': [x,y,x+w,y+h], 'visibility': None})
    return truth, query_targets


def analyze(manifest, rows):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from locatemot.groundjev.decision import overlap
    truth, query_targets = load_truth(manifest)
    physical_clock = manifest.get('physical_video_fps_verified', True)
    buckets = ('gap_le1s','gap_1to3s','gap_gt3s') if physical_clock else ('gap_le1step','gap_1to3steps','gap_gt3steps')
    anchors, prior_ids, last_correct = {}, set(), {}
    previous_observation, last_emitted, previous_gt_frame = {}, {}, {}
    no_match, recovery, semantic = Counter(), Counter(), Counter()
    counts, diagnostics, correct_masses = Counter(), [], []
    for observation_index, row in enumerate(rows):
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
                if query_targets is not None and 'query_membership_probability' in d:
                    positive = identity in set(query_targets.get(str(row['frame']), []))
                    predicted = d['query_membership_probability'] >= .5
                    semantic['TP' if predicted and positive else 'FP' if predicted else 'FN' if positive else 'TN'] += 1
                available=set(shortlist)&set(existing)
                expected_no_match=bool(existing) and not available
                predicted_no_match=event['action']=='NO_MATCH'
                no_match['TP' if predicted_no_match and expected_no_match else
                         'FP' if predicted_no_match else 'FN' if expected_no_match else 'TN'] += 1
                if existing:
                    model_choice=max(d['probabilities'],key=d['probabilities'].get)
                    model_correct=model_choice in {str(t) for t in existing}
                    assigned_correct=tid in existing
                    counts['model_correct_before_global_matching']+=model_correct
                    counts['model_correct_to_wrong_after_global_matching']+=model_correct and not assigned_correct
                    counts['model_wrong_to_correct_after_global_matching']+=not model_correct and assigned_correct
                previous=previous_observation.get(identity)
                if previous and observation_index-previous['observation_index']>1:
                    recovery['candidate_gap_return_events']+=1
                    gap=row['state']['timestamp']-previous['timestamp']
                    bucket=buckets[0] if gap<=1 else buckets[1] if gap<=3 else buckets[2]
                    old=last_emitted.get(identity)
                    if old is not None:
                        recovery[bucket+'_eligible']+=1
                        recovered=tid==old and before.get(tid)==identity
                        recovery[bucket+'_same_correct_birth_anchor_ID']+=recovered
                        counts['recovery_identity_options_present']+=old in shortlist
                        counts['recovery_global_assignment_same_ID']+=tid==old
                    else:
                        recovery['candidate_gap_returns_without_prior_emitted_identity']+=1
                if identity in previous_gt_frame and observation_index-previous_gt_frame[identity]>1:
                    recovery['GT_gap_returns_with_matched_candidate']+=1
                previous_observation[identity]={'frame':row['frame'],'timestamp':row['state']['timestamp'],
                                              'observation_index':observation_index}
                if tid is not None:last_emitted[identity]=tid
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
                if target and target['visibility'] is not None and target['visibility']<.6:
                    counts['memory_updates_low_visibility'] += 1
            elif event.get('memory_action')=='HOLD':
                counts['memory_holds'] += 1
            diagnostics.append({'frame':row['frame'],'candidate':index,'matched_gt_identity':identity,
                'action':event['action'],'assigned_track_id':tid,'past_anchor_gt_identity':before.get(tid),
                'correct_surviving_anchor_ids':existing,'correct_option_present':option_present,
                'association_correct_under_fixed_anchor_diagnostic':correct,
                'memory_action':event.get('memory_action')})
        prior_ids = set(row['state']['persistent_ids'])
        previous_gt_frame.update({g['identity']:observation_index for g in gt})
    n = counts['existing_identity_events']
    return {'scope':manifest['scope'],'frames':len(rows),'counts':dict(counts),
        'correct_option_coverage':counts['correct_identity_option_present']/n if n else None,
        'existing_association_accuracy':counts['correct_existing_associations']/n if n else None,
        'mean_native_correct_identity_option_mass':sum(correct_masses)/len(correct_masses) if correct_masses else None,
        'candidate_GT_recall_at_iou05':counts['gt_matched_candidates']/counts['gt_observations'] if counts['gt_observations'] else None,
        'diagnostic_policy':'all registered physical GT (MOT marked pedestrians; RMOT all release physical objects; TAO annotated objects), one-to-one candidate IoU>=.5; immutable track-anchor ownership from birth, no GT correction',
        'physical_identity_state_diagnostic_separate_from_query_output_filter': True,
        'GT_annotation_scope': 'nonexhaustive TAO; unmatched objects are not universal negative truth' if manifest['task']=='ovmot' else 'registered release GT',
        'physical_seconds_verified': physical_clock,
        'posthoc_query_membership_matched_physical_GT_only': {'counts':dict(semantic),
            'precision':semantic['TP']/(semantic['TP']+semantic['FP']) if semantic['TP']+semantic['FP'] else None,
            'recall':semantic['TP']/(semantic['TP']+semantic['FN']) if semantic['TP']+semantic['FN'] else None,
            'head_was_present':bool(semantic), 'threshold':.5,
            'legacy_missing_head_not_interpreted_as_negative_predictions':True},
        'unmatched_candidates_note':'includes evaluator-ignore regions; these are unmatched diagnostics, not official FP/NoMatch precision',
        'explicit_NO_MATCH_decision_currently_supported':any(r['state']['decision_metadata'].get('explicit_no_match',False) for r in rows),
        'posthoc_NO_MATCH_matched_GT_only':{'counts':dict(no_match),
            'precision':no_match['TP']/(no_match['TP']+no_match['FP']) if no_match['TP']+no_match['FP'] else None,
            'recall':no_match['TP']/(no_match['TP']+no_match['FN']) if no_match['TP']+no_match['FN'] else None,
            'gold_rule':'GT-matched valid object has a surviving correct birth anchor, but no correct identity in its actual shortlist',
            'unmatched_or_ignore_regions_excluded':True},
        'posthoc_identity_recovery':{'counts':dict(recovery),
            'success_rates':{b:recovery[b+'_same_correct_birth_anchor_ID']/recovery[b+'_eligible'] if recovery[b+'_eligible'] else None
                             for b in buckets},
            'rule':'first GT-aligned candidate after at least one missing candidate observation; same previous emitted ID with matching immutable birth anchor',
            'scope':'natural candidate/GT gaps only; no artificial occlusion, future GT correction or recovery metric replacing TrackEval'},
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
