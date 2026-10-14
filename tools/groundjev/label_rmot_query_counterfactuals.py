"""Post-hoc labels for query interventions; no actor or official metric changes."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from locatemot.groundjev.decision import overlap
    from locatemot.unified.data.legal_scope import assert_legal_video
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.report.read_text())
    assert source['status'] == 'REAL_NATIVE_CPU_QUERY_RESPONSE_DIAGNOSTIC_COMPLETE'
    names = {'rmot_single8': 'rmot_single_empty', 'rmot_multi12': 'rmot_multi_empty'}
    manifests = {k: json.loads((ROOT/'outputs/groundjev/multitask_pilot_data'/(v+'.json')).read_text())
                 for k, v in names.items()}
    queries = {}
    for name, manifest in manifests.items():
        assert_legal_video(manifest['sequence'], dataset=manifest['dataset'])
        queries[name] = json.loads(Path(manifest['expression_file_for_evaluation_only']).read_text())
    assert len({m['sequence'] for m in manifests.values()}) == 1
    report = {'status': 'POSTHOC_REAL_QUERY_COUNTERFACTUAL_LABELS_COMPLETE', 'source': str(args.report),
              'GT_used_during_actor_or_interventions': False, 'official_metric_replacement': False,
              'diagnostic': 'candidate IoU>=.5 with primary target and not counter target, conditional membership only',
              'unknown_or_unmatched_candidates_used_as_counter_negatives': False,
              'counter_expressions_chosen_before_model_outputs': True, 'threshold': .5, 'variants': {}}
    for variant, result in source['variants'].items():
        report['variants'][variant] = {}
        for scope, validation in result['validation'].items():
            other = next(n for n in names if n != scope)
            manifest = manifests[scope]
            assert validation['counter'] == manifests[other]['expression']
            width, height = manifest['image_size']
            records = []
            for row in validation['rows']:
                target_ids = [set(queries[n]['label'].get(str(row['frame']), [])) for n in (scope, other)]
                labels = {}
                if set.union(*target_ids):
                    label = Path(manifest['label_dir_for_evaluation_only'])/f"{row['frame']:06d}.txt"
                    for line in label.read_text().splitlines():
                        f = line.split(); x, y, w, h = map(float, f[2:6])
                        labels[int(f[1])] = [x*width, y*height, (x+w)*width, (y+h)*height]
                    assert set.union(*target_ids) <= labels.keys()
                boxes = [[labels[i] for i in ids] for ids in target_ids]
                for i, candidate in enumerate(row['boxes']):
                    primary = any(overlap(candidate, b) >= .5 for b in boxes[0])
                    counter = any(overlap(candidate, b) >= .5 for b in boxes[1])
                    if primary and not counter:
                        records.append({'frame': row['frame'], 'candidate': i,
                                        'scores': {k: v[i] for k, v in row['scores'].items()}})
            n = len(records)
            report['variants'][variant][scope] = {'primary_only_candidate_observations': n,
                'original_positive_at_fixed_half': sum(r['scores']['original'] >= .5 for r in records),
                'counter_false_positive_at_fixed_half': sum(r['scores']['counter'] >= .5 for r in records),
                'counter_score_exceeds_correct_query': sum(r['scores']['counter'] > r['scores']['original'] for r in records),
                'mean_original_score': sum(r['scores']['original'] for r in records)/n if n else None,
                'mean_counter_score': sum(r['scores']['counter'] for r in records)/n if n else None,
                'records': records}
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({**report, 'variants': {k: {s: {a: b for a, b in v.items() if a != 'records'}
           for s, v in d.items()} for k, d in report['variants'].items()}}, indent=2))


if __name__ == '__main__':
    main()
