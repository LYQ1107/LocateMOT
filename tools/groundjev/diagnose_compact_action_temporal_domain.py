"""Frozen fit-input predictions and actual long-run temporal input-domain audit."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def distribution(values):
    import numpy as np
    data = np.asarray(values, dtype=np.float32).reshape(-1, 4)
    return {'observations': len(data), 'min': data.min(0).tolist(), 'max': data.max(0).tolist(),
        'p50': np.quantile(data, .5, axis=0).tolist(), 'p95': np.quantile(data, .95, axis=0).tolist(),
        'p99': np.quantile(data, .99, axis=0).tolist()}


def action(index, history_count):
    return 'EXISTING' if index < history_count else ('NEW', 'IGNORE', 'NO_MATCH')[index-history_count]


def main():
    import numpy as np
    import torch
    import torch.nn.functional as F
    from locatemot.groundjev.compact import CompactCore

    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(); plan = json.loads(args.plan.read_text())
    assert plan['project_root'] == '/data2/user/LocateMOT'
    assert plan['combined_registered_upper_bytes'] < plan['global_temporary_feature_budget_bytes']
    out = ROOT/plan['output']; out.mkdir(parents=True, exist_ok=False)
    owner = {'status': 'COMPACT_ACTION_AND_TEMPORAL_DOMAIN_CPU_AUDIT_RUNNING', 'full_goal_complete': False}

    def save():
        (out/'owner_state.json').write_text(json.dumps(owner, indent=2)+'\n')
        print(json.dumps(owner), flush=True)

    save()
    try:
        torch.set_num_threads(plan['torch_threads']); torch.manual_seed(0)
        trained = json.loads((ROOT/plan['training_owner']).read_text())
        proof = json.loads((ROOT/plan['completed_natural_validation']).read_text())
        assert proof['frames'] == 1657 and proof['native_candidates'] == 235232
        pack = torch.load(Path(trained['source_pack'])/'examples.pt', map_location='cpu', weights_only=True)
        saved = torch.load(trained['checkpoint'], map_location='cpu', weights_only=True)
        model = CompactCore(saved['architecture'], semantic=saved['semantic'],
            input_evidence=saved.get('input_evidence', 'geometry_appearance_language'))
        model.load_state_dict(saved['state_dict']); model.eval()
        eligible, confusion, exact, errors, groups = defaultdict(Counter), defaultdict(Counter), Counter(), defaultdict(list), Counter()
        train_values, per_task_values = [], defaultdict(list)
        with torch.no_grad(), (out/'frozen_training_input_predictions.jsonl').open('w') as record:
            for example in pack['examples']:
                task = ('mot', 'rmot', 'ovmot')[example['task_index']]
                m = len(example['history']); target = example['target']
                inputs = dict(example, class_text=pack['class_text'])
                result = model(inputs); predicted = result['match'].argmax(-1)
                assert bool(torch.isfinite(result['match']).all())
                gold_counts, prediction_counts = Counter(), Counter()
                for truth, prediction in zip(target.tolist(), predicted.tolist()):
                    gold, chosen = action(truth, m), action(prediction, m)
                    gold_counts[gold] += 1; prediction_counts[chosen] += 1
                    confusion[task][gold+'->'+chosen] += 1
                    exact[task] += truth == prediction
                eligible[task].update(gold_counts)
                groups[task] += 1
                loss = float(F.cross_entropy(result['match'], target)) if len(target) else None
                if loss is not None:
                    errors[task].append(loss)
                scalars = example['history_geo'][:, 4:8].tolist()
                train_values.extend(scalars); per_task_values[task].extend(scalars)
                record.write(json.dumps({'id': example['id'], 'task': task, 'history_count': m,
                    'candidate_count': len(target), 'target_actions': dict(gold_counts),
                    'predicted_actions': dict(prediction_counts), 'exact_identity_or_action_correct': int((predicted == target).sum()),
                    'mean_candidate_CE': loss, 'derived_withholding': bool(example.get('derived_withholding'))})+'\n')
        train_domain = distribution(train_values)
        lower, upper = np.asarray(train_domain['min']), np.asarray(train_domain['max'])
        run = ROOT/plan['completed_natural_run']
        live, deployment_values = {}, []
        outside, below, total, frames, candidates, number_groups = np.zeros(4, dtype=int), np.zeros(4, dtype=int), 0, 0, 0, 0
        with (run/'frame_decisions.jsonl').open() as source, (out/'deployment_group_domains.jsonl').open('w') as records:
            for line in source:
                row = json.loads(line); timestamp = row['state']['timestamp']
                events = {e['candidate']: e for e in row['state']['events'] if 'candidate' in e}
                assert set(events) == set(range(len(row['candidates'])))
                for offset in range(0, len(row['candidates']), plan['compact_group_size']):
                    group = [events[i] for i in range(offset, min(offset+plan['compact_group_size'], len(events)))]
                    ids = group[0]['decision']['shortlisted_ids']
                    assert all(e['decision']['shortlisted_ids'] == ids for e in group)
                    values = [[live[t]['age']/100, live[t]['lost_age']/3,
                        (timestamp-live[t]['last_reliable_timestamp'])/3, live[t]['confidence'] or 0.] for t in ids]
                    if values:
                        data = np.asarray(values, dtype=np.float32)
                        high, low = (data > upper).sum(0), (data < lower).sum(0)
                        outside += high; below += low; total += len(data)
                        deployment_values.extend(values)
                        records.write(json.dumps({'frame': row['frame'], 'candidate_offset': offset,
                            'history_count': len(ids), 'min': data.min(0).tolist(), 'max': data.max(0).tolist(),
                            'above_final_stage_eligible_max_counts': high.tolist(),
                            'below_final_stage_eligible_min_counts': low.tolist(),
                            'actual_candidate_actions': dict(Counter(e['action'] for e in group))})+'\n')
                    number_groups += 1
                seen = set()
                for i, event in events.items():
                    tid = event.get('track_id')
                    if tid is None:
                        continue
                    if event['action'] == 'NEW':
                        assert tid not in live
                        live[tid] = {'age': 1, 'last_reliable_timestamp': timestamp}
                    else:
                        live[tid]['age'] += 1
                        if event['decision']['update_probability'] >= .5:
                            live[tid]['last_reliable_timestamp'] = timestamp
                    probabilities = event['decision']['probabilities']
                    live[tid].update(timestamp=timestamp, lost_age=0.,
                        confidence=probabilities.get(str(tid), probabilities['NEW']))
                    seen.add(tid)
                for tid in live.keys()-seen:
                    live[tid]['lost_age'] = timestamp-live[tid]['timestamp']
                live = {tid: value for tid, value in live.items() if tid in row['state']['persistent_ids']}
                assert set(live) == set(row['state']['persistent_ids'])
                frames += 1; candidates += len(events)
        assert frames == 1657 and candidates == 235232
        memory = torch.load(run/'final_identity_memory.pt', map_location='cpu', weights_only=True)
        assert set(live) == {r['track_id'] for r in memory['tracks']}
        for row in memory['tracks']:
            assert all(live[row['track_id']][key] == row[key] for key in
                       ('age', 'lost_age', 'last_reliable_timestamp', 'timestamp', 'confidence'))
        assert not torch.cuda.is_initialized()
        result = {'status': 'ELIGIBLE_ACTION_SUPPORT_FROZEN_FIT_PROBES_AND_ACTUAL_TEMPORAL_DOMAIN_VALIDATED',
            'checkpoint': trained['checkpoint'], 'final_stage_eligible_pack': trained['source_pack'],
            'eligible_groups': dict(groups), 'eligible_action_counts': {k: dict(v) for k, v in eligible.items()},
            'frozen_fit_input_predicted_action_confusion': {k: dict(v) for k, v in confusion.items()},
            'frozen_fit_exact_identity_or_action_accuracy': {k: exact[k]/sum(eligible[k].values()) for k in groups},
            'frozen_fit_mean_group_CE': {k: float(np.mean(v)) for k, v in errors.items()},
            'actual_sampled_example_IDs_saved': 'sampled_example_ids' in trained,
            'exact_historical_optimizer_sample_exposure_claim': False,
            'scalar_fields': plan['fields'], 'final_stage_eligible_scalar_domain': train_domain,
            'per_task_eligible_scalar_domain': {k: distribution(v) for k, v in per_task_values.items()},
            'deployment_preframe_scalar_domain': distribution(deployment_values),
            'deployment_candidate_groups': number_groups, 'deployment_nonempty_history_rows': total,
            'deployment_above_eligible_max_counts': outside.tolist(), 'deployment_below_eligible_min_counts': below.tolist(),
            'deployment_above_eligible_max_fractions': (outside/total).tolist(),
            'all1657_frames_and235232_candidates_reconstructed': True,
            'all_surviving_final_temporal_and_confidence_fields_match_saved_memory_exactly': True,
            'zero_candidate_groups_not_in_scalar_domain': 'Empty-candidate groups have no saved shortlist; audited candidate groups only. All frames still update reconstructed state.',
            'GT_or_input_or_model_intervention': False, 'new_GPU_or_optimizer_or_feature_archive': False,
            'temporal_distribution_shift_not_causal_failure_or_repair_proof': True,
            'fit_probe_is_not_held_out_accuracy': True,
            'combined_registered_upper_bytes': plan['combined_registered_upper_bytes'], 'full_goal_complete': False}
        (out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
        assert sum(p.stat().st_size for p in out.iterdir() if p.is_file()) < plan['new_result_budget_bytes']
        owner.update(status=result['status'], reconstructed_frames=frames, candidate_groups=number_groups)
        save(); print(json.dumps(result, indent=2), flush=True)
    except Exception as error:
        owner.update(status='COMPACT_ACTION_AND_TEMPORAL_DOMAIN_AUDIT_FAILED', error=repr(error)); save()
        raise


if __name__ == '__main__':
    main()
