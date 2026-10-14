"""CPU query interventions on real recorded candidates and causal Compact histories."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from PIL import Image
    from locatemot.groundjev.compact import CompactDecision
    from locatemot.groundjev.decision import overlap
    from locatemot.groundjev.tracker import OnlineTracker
    from locatemot.unified.data.legal_scope import assert_legal_video

    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(0)
    scopes = [('rmot_single8', 'rmot_single_empty'), ('rmot_multi12', 'rmot_multi_empty')]
    manifests = {name: json.loads((ROOT/'outputs/groundjev/multitask_pilot_data'/
                 (file+'.json')).read_text()) for name, file in scopes}
    controls = ROOT/'outputs/groundjev/compact/rmot_native72_stage_v1'
    variants = [('joint', ROOT/'outputs/groundjev/compact/joint1200_v1',
                 ROOT/'outputs/groundjev/compact/joint1200_controls_v1',
                 ROOT/'outputs/groundjev/compact/joint_pack_v3/examples.pt'),
                ('broad_teacher', controls/'broad_teacher_training', controls/'broad_teacher_controls',
                 ROOT/'outputs/groundjev/compact/rmot_native72_v1/broad_teacher/examples.pt'),
                ('broad_native', controls/'broad_native_training', controls/'broad_native_controls',
                 ROOT/'outputs/groundjev/compact/rmot_native72_v1/broad_native/examples.pt')]
    report = {'status': 'RUNNING_CPU_QUERY_RESPONSE_DIAGNOSTIC', 'variants': {},
              'runtime_GT_read': False, 'no_threshold_tuning': True,
              'official_metric_replacement': False, 'extra_GPU_or_native_requests': False,
              'scope': '20 registered short validation frames and each existing fit pack; not full datasets',
              'intervention': 'same captured primary-query causal core input; change spec_text only',
              'counter_expression': 'other registered validation expression, fixed before model outputs',
              'feature_source': 'retained original DLM native2560 ROI features',
              'permutation_probability_tolerance': 1e-4}
    for name, training, result_root, pack_path in variants:
        decision = CompactDecision(training/'checkpoint.pt', device='cpu')
        owner = json.loads((result_root/'owner_state.json').read_text())
        result = {'checkpoint': str(training/'checkpoint.pt'), 'validation': {}}
        for scope, _ in scopes:
            manifest = manifests[scope]
            assert_legal_video(manifest['sequence'], dataset=manifest['dataset'])
            before = Path(owner['completed'][scope+'_before']['run'])
            before = before if before.is_absolute() else ROOT/before
            rows = [json.loads(x) for x in (before/'frame_decisions.jsonl').open()]
            assert [r['frame'] for r in rows] == manifest['frame_numbers']
            reference = [json.loads(x) for x in (result_root/(scope+'_after')/'frame_decisions.jsonl').open()]
            spec = {'task': 'rmot', 'vocabulary': manifest['vocabulary'], 'expression': manifest['expression']}
            other = manifests[scopes[1][0] if scope == scopes[0][0] else scopes[0][0]]['expression']
            texts = {mode: decision.text_features.specification(dict(spec, expression=expression))
                     for mode, expression in [('original', spec['expression']), ('no_expression', None), ('counter', other)]}
            tracker = OnlineTracker(decision)
            captured_rows, differences, changed_frames = [], [], []
            permutation_delta = {k: 0. for k in ('match', 'update', 'reliability', 'life', 'membership', 'category')}
            for row, old in zip(rows, reference):
                assert row['candidates'] == old['candidates']
                assert all(len(c['native_region_feature']) == 2560 for c in row['candidates'])
                captured = []
                hook = decision.model.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0]))
                with Image.open(row['image_path']) as raw:
                    outputs, state = tracker.step(raw.convert('RGB'), row['candidates'], row['frame']/manifest['fps'], spec)
                hook.remove()
                assert state['decision_metadata']['GT_read'] is False
                geometry = lambda values: [{k: v for k, v in p.items()
                                            if k not in ('association_probability', 'query_membership_probability')}
                                           for p in values]
                if geometry(outputs) != geometry(old['outputs']):
                    changed_frames.append(row['frame'])
                old_decisions = {e['candidate']: e['decision'] for e in old['state']['events'] if 'candidate' in e}
                current_decisions = {e['candidate']: e['decision'] for e in state['events'] if 'candidate' in e}
                differences.extend(abs(current_decisions[i]['query_membership_probability']-
                                       old_decisions[i]['query_membership_probability']) for i in current_decisions)
                scores = {mode: [] for mode in texts}
                with torch.no_grad():
                    for e in captured:
                        baseline = decision.model(e)
                        for mode, text in texts.items():
                            y = decision.model(dict(e, spec_text=text))
                            scores[mode].extend(y['membership'].sigmoid().tolist())
                        n, m = len(e['candidates']), len(e['history'])
                        cp, hp = torch.randperm(n, generator=generator), torch.randperm(m, generator=generator)
                        shuffled = dict(e, candidates=e['candidates'][cp], candidate_geo=e['candidate_geo'][cp],
                                        history=e['history'][hp], history_geo=e['history_geo'][hp],
                                        pair_geometry=e['pair_geometry'][cp][:, hp])
                        y = decision.model(shuffled)
                        ci, hi = cp.argsort(), hp.argsort()
                        match_columns = torch.cat([hi, torch.arange(m, m+3)])
                        aligned = {'match': y['match'][ci][:, match_columns].softmax(-1),
                                   'update': y['update'][ci].sigmoid(), 'membership': y['membership'][ci].sigmoid(),
                                   'reliability': y['reliability'][ci].softmax(-1),
                                   'life': y['life'][hi].softmax(-1), 'category': y['category'][ci].softmax(-1)}
                        for key, actual in aligned.items():
                            expected = baseline[key].sigmoid() if key in ('update', 'membership') else baseline[key].softmax(-1)
                            assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
                            if actual.numel():
                                permutation_delta[key] = max(permutation_delta[key], float((actual-expected).abs().max()))
                assert all(len(v) == len(row['candidates']) for v in scores.values())
                captured_rows.append({'frame': row['frame'], 'boxes': [c['box'] for c in row['candidates']], 'scores': scores})
            # Read release supervision only after the causal actor has finished.
            query = json.loads(Path(manifest['expression_file_for_evaluation_only']).read_text())
            all_scores = {mode: [] for mode in texts}
            positive_scores = {mode: [] for mode in texts}
            for row in captured_rows:
                ids = set(query['label'].get(str(row['frame']), []))
                gt = []
                if ids:
                    width, height = manifest['image_size']
                    label = Path(manifest['label_dir_for_evaluation_only'])/f"{row['frame']:06d}.txt"
                    for line in label.read_text().splitlines():
                        f = line.split()
                        if int(f[1]) in ids:
                            x, y, w, h = map(float, f[2:6])
                            gt.append([x*width, y*height, (x+w)*width, (y+h)*height])
                positive = [any(overlap(box, g) >= .5 for g in gt) for box in row['boxes']]
                row['posthoc_primary_query_GT_covered'] = positive
                for mode in texts:
                    all_scores[mode].extend(row['scores'][mode])
                    positive_scores[mode].extend(s for s, matched in zip(row['scores'][mode], positive) if matched)
            mean = lambda values: sum(values)/len(values) if values else None
            result['validation'][scope] = {'frames': len(rows), 'candidate_observations': len(all_scores['original']),
                'GT_covered_candidate_observations': len(positive_scores['original']),
                'all_candidate_mean_membership': {k: mean(v) for k, v in all_scores.items()},
                'GT_covered_mean_membership': {k: mean(v) for k, v in positive_scores.items()},
                'GT_covered_above_fixed_half': {k: sum(x >= .5 for x in v) for k, v in positive_scores.items()},
                'mean_absolute_query_intervention_change': {k: mean([abs(a-b) for a, b in zip(all_scores['original'], v)])
                                                            for k, v in all_scores.items() if k != 'original'},
                'CPU_vs_recorded_GPU_changed_output_frames': changed_frames,
                'CPU_vs_recorded_GPU_max_membership_difference': max(differences, default=0.),
                'inverse_aligned_permutation_max_probability_difference': permutation_delta,
                'inverse_aligned_permutation_within_declared_tolerance': max(permutation_delta.values()) <= 1e-4,
                'candidate_grouping_unchanged': True,
                'permutation_scope': 'within captured groups and identity shortlists; no cross-group/full-pipeline invariance claim',
                'counter': other, 'rows': captured_rows}
        pack = torch.load(pack_path, map_location='cpu', weights_only=True)
        fit = {'groups': 0, 'positive': 0, 'positive_above_half': 0, 'negative': 0, 'negative_below_half': 0}
        with torch.no_grad():
            for e in pack['examples']:
                if e['task_index'] != 1:
                    continue
                p = decision.model(dict(e, class_text=pack['class_text']))['membership'].sigmoid()
                y = e['membership_target'].bool()
                fit['groups'] += 1
                fit['positive'] += int(y.sum()); fit['negative'] += int((~y).sum())
                fit['positive_above_half'] += int(((p >= .5) & y).sum())
                fit['negative_below_half'] += int(((p < .5) & ~y).sum())
        result['fit_membership_at_unchanged_half'] = fit
        report['variants'][name] = result
        (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({'variant': name, 'fit': fit, 'validation': {
            k: {a: b for a, b in v.items() if a != 'rows'} for k, v in result['validation'].items()}}), flush=True)
    report['status'] = 'REAL_NATIVE_CPU_QUERY_RESPONSE_DIAGNOSTIC_COMPLETE'
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
