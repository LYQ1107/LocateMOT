"""Read saved Full decisions: is the memory gate reliable after global assignment?"""
from collections import Counter, defaultdict
from itertools import groupby
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def auc(pairs):
    positive = sum(label for _,label in pairs)
    negative = len(pairs)-positive
    if not positive or not negative:
        return None
    rank_sum, offset = 0., 0
    for _, group in groupby(sorted(pairs),key=lambda row:row[0]):
        group = list(group)
        mean_rank = offset+(len(group)+1)/2
        rank_sum += mean_rank*sum(label for _,label in group)
        offset += len(group)
    return (rank_sum-positive*(positive+1)/2)/(positive*negative)


def diagnose(run):
    diagnostic = json.loads((run/'association_events.json').read_text())
    indexed = {(e['frame'],e['candidate']):e for e in diagnostic['events']}
    assert len(indexed) == len(diagnostic['events']) == 13664
    counts, values, pairs, seen = Counter(), defaultdict(list), [], set()
    for line in (run/'frame_decisions.jsonl').open():
        row = json.loads(line)
        for event in row['state']['events']:
            if 'candidate' not in event:
                continue
            key = row['frame'],event['candidate']
            source = indexed[key]
            assert key not in seen
            seen.add(key)
            assert source['action'] == event['action']
            assert source['assigned_track_id'] == event.get('track_id')
            assert source['memory_action'] == event.get('memory_action')
            if event['action'] in ('NEW','IGNORE','NO_MATCH'):
                continue
            gid, anchor = source['matched_gt_identity'],source['past_anchor_gt_identity']
            category = ('unmatched' if gid is None else 'unknown_birth' if anchor is None else
                'known_consistent' if anchor == gid else 'known_conflicting')
            decision = event['decision']
            assigned = str(event['track_id'])
            probabilities = decision['probabilities']
            preferred = max(probabilities,key=probabilities.get)
            agree = assigned == preferred
            update = decision['update_probability']
            counts[category+'_assigned'] += 1
            counts[category+'_assigned_argmax_agrees'] += agree
            values[category+'_update_probability'].append(update)
            values[category+'_assigned_identity_probability'].append(probabilities[assigned])
            values[category+'_top_choice_probability'].append(probabilities[preferred])
            if category.startswith('known_'):
                pairs.append((update,int(category=='known_consistent')))
            if event['memory_action'] == 'UPDATE':
                counts[category+'_UPDATE'] += 1
                counts[category+'_UPDATE_argmax_agrees'] += agree
                counts[category+'_UPDATE_argmax_disagrees'] += not agree
    assert seen == set(indexed)
    original = diagnostic['counts']
    assert counts['known_consistent_UPDATE'] == original['memory_updates_identity_consistent']
    assert sum(counts[k+'_UPDATE'] for k in ('known_consistent','known_conflicting','unknown_birth','unmatched')) == original['memory_updates_identity_consistent']+original['memory_updates_wrong_identity_or_unmatched']
    summary = {key:{'count':len(v),'mean':statistics.mean(v),'median':statistics.median(v)} for key,v in values.items()}
    return {'run':str(run),'counts':dict(counts),'saved_probability_summaries':summary,
        'native_Q3_AUROC_known_consistent_vs_known_conflicting_assigned_events':auc(pairs),
        'hypothetical_argmax_agreement_gate_on_fixed_existing_writes':{
            'known_consistent_writes_held':counts['known_consistent_UPDATE_argmax_disagrees'],
            'known_conflicting_writes_held':counts['known_conflicting_UPDATE_argmax_disagrees'],
            'known_consistent_writes_retained':counts['known_consistent_UPDATE_argmax_agrees'],
            'known_conflicting_writes_retained':counts['known_conflicting_UPDATE_argmax_agrees']}}


def main():
    directory = ROOT/'outputs/groundjev/training/onejev_full1203_controls_v2_wait_fix'
    validated = json.loads((directory/'completed_units_validated_summary.json').read_text())
    rows = {name:diagnose(Path(entry['run'])) for name,entry in validated['results'].items()}
    report = {'status':'SAVED_FULL_MOT_GLOBAL_ASSIGNMENT_MEMORY_GATE_DIAGNOSED',
        'results':rows,'new_model_calls_or_modified_predictions_or_states':False,
        'GT_posthoc_only':True,'new_feature_cache_bytes':0,
        'limits':'Two completed1203frame trajectories, one legal validation video/seed. '
            'Known birth anchors are required for consistent/conflicting labels; unknown birth '
            'and unmatched candidates remain separate. Argmax agreement only describes which '
            'already-observed writes would be held on frozen histories, not a causal rollout, '
            'trained repair, counterfactual HOTA or an accepted policy change. '
            'Global matching legitimately uses non-argmax alternatives under competition; '
            'disagreement alone does not prove the chosen identity incorrect.'}
    output = ROOT/'outputs/groundjev/diagnostics/full_mot_assignment_memory_v1'
    output.mkdir(parents=True,exist_ok=False)
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'results':{name:row['hypothetical_argmax_agreement_gate_on_fixed_existing_writes'] for name,row in rows.items()}}))


if __name__ == '__main__':
    main()
