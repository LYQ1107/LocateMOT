"""Paired full-video metrics and unmodified J1 action-probability diagnostics."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.groundjev.eval_no_mot20.common import RUN, load, save


def summarize(job):
    baseline = load(job/'simple/frozen_receipt.json')['input']
    results = {}
    for arm in ('simple','j1','j1_competition_xyxy_v2',
                'j1_current_crops_v3','j1_current_crops_memory_v4'):
        directory = job/arm
        if not (directory/'frozen_receipt.json').exists():
            results[arm] = {'status':'PENDING_FULL_VIDEO_RECEIPT'}
            continue
        frozen = load(directory/'frozen_receipt.json')
        if frozen['input'] != baseline:
            raise ValueError(f'{arm}: different native full-video candidates')
        actions, raw_best, feasible_best = Counter(), Counter(), Counter()
        posterior = Counter()
        outputs = candidates = 0
        native_probabilities = True
        seen_ids = set()
        frames = 0
        for line in (directory/'frame_decisions.jsonl').open():
            row = json.loads(line)
            frames += 1
            outputs += len(row['outputs'])
            seen_ids.update(output['track_id'] for output in row['outputs'])
            metadata = row['state']['decision_metadata']
            native_probabilities = native_probabilities and metadata['native_probabilities']
            masked = set(metadata.get('no_match_infeasible_candidate_indices') or [])
            for event in row['state']['events']:
                actions[event['action']] += 1
                if 'candidate' not in event:
                    continue
                p = event['decision']['probabilities']
                candidates += 1
                for label in ('NEW','NO_MATCH','IGNORE'):
                    posterior[label] += p.get(label, 0.)
                posterior['maximum_existing'] += max((value for key,value in p.items() if key.isdigit()), default=0.)
                category = lambda key: 'existing' if key.isdigit() else key
                raw_best[category(max(p, key=p.get))] += 1
                feasible = {key:value for key,value in p.items()
                            if key != 'NO_MATCH' or event['candidate'] not in masked}
                feasible_best[category(max(feasible, key=feasible.get))] += 1
        if frames != baseline['frames']:
            raise ValueError(f'{arm}: incomplete prediction-frame scope')
        timing = load(directory/'result.json')
        official = load(directory/'official_metrics.json')
        results[arm] = {
            'status':'FROZEN_FULL_VIDEO_SAME_CANDIDATES','frames':frames,
            'input':frozen['input'],'prediction_sha256':frozen['prediction_sha256'],
            'metrics':official['metrics'],'action_counts':dict(actions),
            'candidate_decisions':candidates,'actual_output_boxes':outputs,'distinct_output_ids':len(seen_ids),
            'native_probabilities':native_probabilities,
            'mean_raw_probability':({key:value/candidates for key,value in posterior.items()}
                                    if candidates and native_probabilities else None),
            'mean_raw_reference_score':({key:value/candidates for key,value in posterior.items()}
                                        if candidates and not native_probabilities else None),
            'raw_decision_argmax_counts':dict(raw_best),
            'legal_NO_MATCH_mask_only_argmax_counts':dict(feasible_best),
            'timing':{key:timing.get(key) for key in ('association_only_seconds','association_replay_fps',
                'complete_video_wall_seconds','end_to_end_fps','peak_decision_gpu_memory_bytes')},
            'candidate_evidence':frozen.get('candidate_evidence','full_frame'),
            'memory_transport':frozen.get('memory_transport','legacy'),
        }
    return {'status':'PAIRED_J1_INPUT_CONTROLS_FULL_VIDEO' if all(
            result['status']=='FROZEN_FULL_VIDEO_SAME_CANDIDATES' for result in results.values())
            else 'INPUT_CONTROLS_PARTIAL','job':str(job),'arms':results,
            'probabilities_modified_for_analysis':False,'new_training':False,
            'argmax_diagnostic_scope':'Candidate-wise diagnostics only; actual outputs use the unchanged global one-to-one assignment.',
            'scope_note':'Full video diagnostic, not a full-dataset result. '
                         'Low IDSW must be interpreted with FN and actual output counts. '
                         'GPU sharing prevents a controlled speed claim.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=RUN/'engineering_checks/j1_input_contract/paired_summary.json')
    args = parser.parse_args()
    result = summarize(args.job)
    save(args.out,result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
