"""Aggregate actual saved native-engine timings without running a model."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def distribution(values):
    ordered = sorted(values)
    return {'count': len(values), 'sum': sum(values), 'mean': statistics.mean(values) if values else None,
            'p50': statistics.median(values) if values else None,
            'p95': ordered[min(len(ordered)-1, int(.95*len(ordered)))] if values else None,
            'max': max(values) if values else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    reports = {}
    for run in args.runs:
        result = json.loads((run/'result.json').read_text())
        assert result['all_input_frames_processed'] and result['backend'] == 'full'
        timings = {k: [] for k in ('render_s', 'forward_s', 'total_s', 'media_load_s')}
        counts = Counter()
        tokens, branches, questions = [], [], []
        frames = 0
        with (run/'frame_decisions.jsonl').open() as source:
            for line in source:
                row = json.loads(line)
                frames += 1
                for batch in row['state']['decision_metadata'].get('batches', []):
                    meta = batch['metadata']
                    counts[meta['fork_mode']] += 1
                    for k in ('render_s', 'forward_s', 'total_s'):
                        timings[k].append(meta['timing'][k])
                    timings['media_load_s'].append(meta['media']['load_s'])
                    tokens.append(meta['prefix_tokens'])
                    branches.append(meta['branches'])
                    questions.append(batch['questions'])
        assert frames == result['frames'] == len(result['latencies_seconds'])
        reports[str(run)] = {'frames': frames, 'fork_mode_counts': dict(counts),
            'native_requests': sum(counts.values()),
            'request_timing_seconds': {k: distribution(v) for k,v in timings.items()},
            'prefix_tokens': distribution(tokens), 'branches': distribution(branches),
            'typed_questions_per_request': distribution(questions),
            'actual_frame_latency_seconds': distribution(result['latencies_seconds']),
            'perception_included_E2E': result['end_to_end_fps'] is not None,
            'limits': 'Saved engine timing labels; nested scopes can overlap. Media loading is not an '
                'additional disjoint cost. Concurrent work and video density differ; not an optimization benchmark.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'status': 'ACTUAL_SAVED_NATIVE_ENGINE_TIMINGS_AGGREGATED',
        'results': reports, 'new_model_invocations': 0, 'full_goal_complete': False}, indent=2)+'\n')
    for run, report in reports.items():
        print(Path(run).name, report['native_requests'], report['fork_mode_counts'],
              {k: round(v['sum'], 3) for k,v in report['request_timing_seconds'].items()})


if __name__ == '__main__':
    main()
