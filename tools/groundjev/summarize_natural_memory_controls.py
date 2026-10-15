"""Freeze all eight completed memory controls with their exact scored predictions."""
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    root = ROOT/'outputs/groundjev/natural_memory/mot20_05_full_validation_gpu5_v1'
    owner = json.loads((root/'owner_state.json').read_text())
    controls = json.loads((root/'controls/owner_state.json').read_text())
    assert owner['status'] == 'NATURAL_MOT20_SCOPE_AND_FROZEN_MEMORY_CONTROLS_COMPLETE'
    assert controls['status'] == 'ALL_FIXED_AFFINITY_CONTROLS_COMPLETE' and len(controls['completed']) == 8
    destination = root/'completed_memory_results'; destination.mkdir(exist_ok=True)
    results = {}
    for name, record in controls['completed'].items():
        run = Path(record['run'])
        official = json.loads((run/'official_metrics.json').read_text())
        assert official['metrics'] == record['metrics']
        frames = candidates = predictions = 0
        with (run/'frame_decisions.jsonl').open() as events, (run/'predictions.txt').open() as raw:
            for line in events:
                event = json.loads(line); frames += 1; candidates += len(event['candidates'])
                for row in event['outputs']:
                    x1, y1, x2, y2 = row['box']
                    expected = f'{event["frame"]},{row["track_id"]},{x1:.3f},{y1:.3f},{x2-x1:.3f},{y2-y1:.3f},1,-1,-1,-1\n'
                    assert next(raw, None) == expected; predictions += 1
            assert next(raw, None) is None
        assert frames == record['decoded_frames'] == 1657 and candidates == record['native_candidates'] == 235232
        assert len(record['native_rejected_frames']) == 117
        target = destination/name; target.mkdir(exist_ok=True)
        for filename in ('official_metrics.json', 'result.json'):
            (target/filename).write_bytes((run/filename).read_bytes())
        diagnosis = json.loads((run/'association_events.json').read_text())
        summary = {k: v for k, v in diagnosis.items() if k != 'events'}
        summary['per_candidate_event_rows_retained_on_server'] = len(diagnosis['events'])
        (target/'association_events_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        with (run/'predictions.txt').open('rb') as source, (target/'predictions.txt.gz').open('wb') as binary:
            with gzip.GzipFile(filename='', mode='wb', fileobj=binary, mtime=0) as packed:
                for block in iter(lambda: source.read(1024*1024), b''): packed.write(block)
        results[name] = dict(record, raw_prediction_rows=predictions, raw_prediction_text_exact=True)
    report = {'status': 'ALL8_MEMORY_CONTROLS1657_FRAMES_RAW_PREDICTIONS_AND_OFFICIAL_RESULTS_VALIDATED',
              'original_gpu5_followup_session': 36847, 'actual_followup_exit_code': 0,
              'frames_each': 1657, 'candidates_each': 235232, 'rejected_frames_each': 117,
              'results': results, 'fresh_GPU_metrics': owner['completed']['fresh_source']['metrics'],
              'CPU_baseline': 'mot20_challenge_reference_gated3',
              'CPU_GPU_closed_loop_outputs_not_equal': True, 'checkpoint_or_policy_promoted': False,
              'interpretation': 'Same CPU reference for all CPU variants. Simple association is substantially stronger than the current learned action policy; changing memory retention alone does not repair the learned policy. Simple30 does not improve HOTA over Simple3. These are inference policy controls on one registered held-out scope, not learned models, full-benchmark parity or unique JEV evidence.',
              'full_goal_complete': False}
    (root/'completed_eight_memory_controls_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': report['status'], 'units': len(results),
                      'raw_prediction_gzip_bytes': sum(p.stat().st_size for p in destination.rglob('*.gz'))}), flush=True)


if __name__ == '__main__':
    main()
