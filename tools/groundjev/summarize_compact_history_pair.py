"""Freeze completed five-scope controls and verify their scored prediction text."""
import argparse
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', choices=['history_pair', 'original_recipe'], default='history_pair')
    args = parser.parse_args()
    original = args.study == 'original_recipe'
    out = ROOT/'outputs/groundjev/compact'/('original_recipe_continuation1200_v1' if original else 'own_history_refresh1200_pair_v1')
    owner = json.loads((out/'owner_state.json').read_text())
    expected_status = ('ORIGINAL_RECIPE_CONTINUATION1200_AND_FIVE_SCOPE_CONTROLS_VALIDATED' if original else
                       'PAIRED_COMPACT_HISTORY1200_AND_ALL_FIVE_SCOPE_OFFICIAL_CONTROLS_VALIDATED')
    assert owner['status'] == expected_status
    if not original: assert owner['exact_per_step_group_ID_sampling_matched']
    controls = owner['completed']['controls']
    assert len(controls) == (10 if original else 15)
    publication = out/'completed_results'; publication.mkdir(exist_ok=True)
    results = {}
    for name, row in controls.items():
        run = Path(row['run'])
        official = json.loads((run/'official_metrics.json').read_text())
        runtime = json.loads((run/'result.json').read_text())
        assert official['metrics'] == row['metrics']
        assert official.get('base_novel') == row['base_novel']
        assert runtime['frames'] == row['decoded_frames'] and runtime['decision_device'] == 'cpu'
        assert runtime['all_input_frames_processed'] and not runtime['gt_read_by_runtime']
        frames = candidates = predictions = 0
        with (run/'predictions.txt').open() as raw, (run/'frame_decisions.jsonl').open() as traces:
            for line in traces:
                frame = json.loads(line); frames += 1; candidates += len(frame['candidates'])
                for output in frame['outputs']:
                    x1, y1, x2, y2 = output['box']
                    expected = f'{frame["frame"]},{output["track_id"]},{x1:.3f},{y1:.3f},{x2-x1:.3f},{y2-y1:.3f},1,-1,-1,-1\n'
                    assert next(raw, None) == expected
                    predictions += 1
            assert next(raw, None) is None
        assert frames == row['decoded_frames'] and candidates == row['native_candidates']
        checkpoint = Path(runtime['checkpoint']).resolve()
        arm = name.rsplit('_', 1)[1]
        expected_checkpoint = (ROOT/'outputs/groundjev/compact/rmot_native72_stage_v1/broad_native_training/checkpoint.pt'
                               if arm == 'before' else out/('training' if original else arm+'_training')/'checkpoint.pt')
        assert checkpoint == expected_checkpoint.resolve()
        results[name] = dict(row, actual_prediction_rows=predictions, frames_and_raw_prediction_text_validated=True)
        if (original and arm == 'continuation') or name in ('tao_raw_full40_refreshed', 'mot20_challenge_old', 'mot20_challenge_refreshed'):
            target = publication/name; target.mkdir(exist_ok=True)
            for filename in ('official_metrics.json', 'result.json'):
                (target/filename).write_bytes((run/filename).read_bytes())
            events = json.loads((run/'association_events.json').read_text())
            scalar_summary = {k: v for k, v in events.items() if k != 'events'}
            scalar_summary['per_candidate_event_rows_retained_on_server'] = len(events['events'])
            (target/'association_events_summary.json').write_text(json.dumps(scalar_summary, indent=2)+'\n')
            with (run/'predictions.txt').open('rb') as source, (target/'predictions.txt.gz').open('wb') as binary:
                with gzip.GzipFile(filename='', fileobj=binary, mode='wb', mtime=0) as packed:
                    for block in iter(lambda: source.read(1024*1024), b''): packed.write(block)
    report = {'status': ('ORIGINAL_RECIPE_FIVE_SCOPE_OFFICIAL_RESULTS_AND_RAW_PREDICTIONS_VALIDATED' if original else
                         'ALL15_FIVE_SCOPE_SAME_CHECKPOINT_OFFICIAL_RESULTS_AND_RAW_PREDICTIONS_VALIDATED'),
              'source_owner_session': 76373 if original else 70970, 'source_owner_exit_code': 0,
              'new_optimization_steps_each': 1200, 'actual_groups_each_task': 400,
              'actual_sample_ID_streams_paired': 'RMOT/OVMOT only; original MOT pool149 versus80' if original else True,
              'results': results,
              'checkpoint_promoted': False, 'full_goal_complete': False,
              'interpretation': 'MOT20 and RMOT improve but DanceTrack and TAO regress against before. Both new arms remove withholding augmentation and MOT distillation; old versus refreshed measures combined history/conditional-label refresh, not an isolated age/confidence intervention. These are registered development scopes, not full-dataset ordinary-model success.'}
    if original:
        prior = json.loads((ROOT/'outputs/groundjev/compact/own_history_refresh1200_pair_v1/completed_five_scope_validation.json').read_text())
        comparison = {}
        for name in results:
            if not name.endswith('_continuation'): continue
            scope = name.removesuffix('_continuation')
            assert results[scope+'_before']['metrics'] == prior['results'][scope+'_before']['metrics']
            assert results[scope+'_before']['base_novel'] == prior['results'][scope+'_before']['base_novel']
            arms = {arm: prior['results'][scope+'_'+arm]['metrics'] for arm in ('before', 'old', 'refreshed')}
            arms['original_recipe_continuation'] = results[name]['metrics']
            metric = 'HOTA' if 'HOTA' in arms['before'] else 'TETA'
            scale = 100 if metric == 'HOTA' else 1
            comparison[scope] = {'reported_metric': metric, 'unit': 'percent',
                                 'scores': {arm: values[metric]*scale for arm, values in arms.items()}}
        assert len(comparison) == 5
        report.update(comparison_with_completed_history_pair=comparison,
            interpretation='Original recipe retains withholding augmentation and MOT teacher distillation. Compare its fixed final checkpoint with before and the completed history pair on every registered scope. This contrast does not separately isolate withholding versus distillation. No per-task checkpoint selection or full-dataset parity claim.')
    (out/'completed_five_scope_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': report['status'], 'units': len(results),
                      'new_prediction_gzip_bytes': sum(p.stat().st_size for p in publication.rglob('*.gz'))}), flush=True)


if __name__ == '__main__':
    main()
