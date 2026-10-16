"""Freeze completed Full600 controls, checking native inputs and official scores."""
import argparse
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def recompute(run, manifest):
    import numpy as np
    from locatemot.unified.runtime.u2_ordinary_evaluation import verify_trackeval_source
    verify_trackeval_source()
    import trackeval
    directory = run/'official_evaluation'
    if manifest['task'] == 'mot':
        sequence = manifest['sequence']
    else:
        sequence = 'query_'+manifest['sequence']
    config = dict(GT_FOLDER=str(directory/'gt'), TRACKERS_FOLDER=str(directory/'trackers'),
        TRACKERS_TO_EVAL=['groundjev'], TRACKER_SUB_FOLDER='data', SKIP_SPLIT_FOL=True,
        BENCHMARK='MOT17', PRINT_CONFIG=False, DO_PREPROC=True,
        SEQ_INFO={sequence: len(manifest['images'])})
    if manifest['task'] == 'rmot':
        config['GT_LOC_FORMAT'] = '{gt_folder}/{seq}/gt.txt'
    adapter = trackeval.datasets.MotChallenge2DBox(config)
    evaluator = trackeval.Evaluator(dict(PRINT_CONFIG=False, PRINT_RESULTS=False, TIME_PROGRESS=False,
        OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False, PLOT_CURVES=False, BREAK_ON_ERROR=True))
    raw, messages = evaluator.evaluate([adapter], [trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR({'PRINT_CONFIG': False}), trackeval.metrics.Identity({'PRINT_CONFIG': False})])
    assert messages[adapter.get_name()]['groundjev'] == 'Success'
    values = raw[adapter.get_name()]['groundjev'][sequence]['pedestrian']
    metrics = {k: float(np.mean(values['HOTA'][k])) for k in ('HOTA', 'DetA', 'AssA')}
    metrics.update(IDF1=float(values['Identity']['IDF1']), IDSW=int(values['CLEAR']['IDSW']),
        FP=int(values['CLEAR']['CLR_FP']), FN=int(values['CLEAR']['CLR_FN']))
    return metrics


def main():
    from tools.groundjev.run_history_refresh_controls import validate_run
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); args.out = args.out.resolve(); args.out.mkdir(parents=True, exist_ok=False)
    plan = json.loads((ROOT/'configs/groundjev/full_history_refresh600_controls_v1.json').read_text())
    report = dict(status='COMPLETED_OLD_FULL600_THREE_SCOPE_NATIVE_OUTPUT_AND_OFFICIAL_RECHECK',
        completed_scopes={}, still_pending=['old_TAO40', 'refreshed_all_four_scopes'],
        same_checkpoint_all_tasks=True, full_goal_complete=False,
        ordinary_model_performance_achieved=False, global_all_task_improvement_proven=False,
        additional_diagnostic_reserve_bytes=1048576,
        combined_registered_feature_upper_bytes=26839845584,
        global_temporary_feature_budget_bytes=26843545600)
    for scope in ('mot_full1203', 'rmot_single800', 'rmot_multi800'):
        job = plan['scopes'][scope]
        run = ROOT/plan['output_root']/('old_'+scope)
        manifest = json.loads((ROOT/job['manifest']).read_text())
        evidence = validate_run(run, job, ROOT/plan['checkpoints']['old'])
        official_predictions, raw_predictions = [], []
        with (run/'frame_decisions.jsonl').open() as stream:
            for index, line in enumerate(stream, 1):
                row = json.loads(line)
                for p in row['outputs']:
                    x, y, x2, y2 = p['box']; identity = p['track_id']
                    raw_predictions.append(f'{row["frame"]},{identity},{x:.3f},{y:.3f},{x2-x:.3f},{y2-y:.3f},1,-1,-1,-1\n')
                    precision = 3 if manifest['task'] == 'mot' else 6
                    official_predictions.append(f'{index},{identity},{x:.{precision}f},{y:.{precision}f},{x2-x:.{precision}f},{y2-y:.{precision}f},1,-1,-1,-1\n')
        raw = ''.join(raw_predictions).encode()
        assert raw == (run/'predictions.txt').read_bytes()
        sequence = manifest['sequence'] if manifest['task'] == 'mot' else 'query_'+manifest['sequence']
        target = run/'official_evaluation/trackers/groundjev/data'/f'{sequence}.txt'
        assert ''.join(official_predictions) == target.read_text()
        rescored = recompute(run, manifest)
        assert rescored == evidence['official_metrics']
        frozen = args.out/scope; frozen.mkdir()
        compressed = gzip.compress(raw, mtime=0)
        assert gzip.decompress(compressed) == raw
        (frozen/'predictions.txt.gz').write_bytes(compressed)
        for name in ('result.json', 'official_metrics.json'):
            (frozen/name).write_bytes((run/name).read_bytes())
        baseline = json.loads((ROOT/job['baseline_run']/'official_metrics.json').read_text())
        evidence.update(baseline_run=job['baseline_run'], baseline_metrics=baseline['metrics'],
            raw_prediction_bytes_exactly_reconstructed_from_every_frame_output=True,
            raw_predictions_gzip_bytes=len(compressed), official_input_exactly_reconstructed=True,
            pinned_original_TrackEval_recomputed_metrics_exact=True,
            HOTA_change_percentage_points=100*(rescored['HOTA']-baseline['metrics']['HOTA']))
        report['completed_scopes'][scope] = evidence
    (args.out/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    actual = sum(p.stat().st_size for p in args.out.rglob('*') if p.is_file())
    assert actual < report['additional_diagnostic_reserve_bytes']
    print(json.dumps(dict(status=report['status'], actual_bytes=actual,
        HOTA_percent={k: 100*v['official_metrics']['HOTA'] for k, v in report['completed_scopes'].items()})), flush=True)


if __name__ == '__main__':
    main()
