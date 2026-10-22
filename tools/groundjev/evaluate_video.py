"""Evaluate a complete legal MOT video scope with unmodified official TrackEval."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    from locatemot.unified.runtime.u2_ordinary_evaluation import digest, prepare_gt_view, verify_trackeval_source
    import trackeval
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--fixture-only', action='store_true', help='explicit evaluator format fixture; never a model score')
    args = parser.parse_args()
    manifest = json.loads(args.video.read_text())
    result = json.loads((args.run/'result.json').read_text())
    rows = [json.loads(line) for line in (args.run/'frame_decisions.jsonl').read_text().splitlines()]
    if result['status'] not in ('VIDEO_INFERENCE_COMPLETE','VIDEO_INFERENCE_COMPLETE_WITH_REJECTED_PERCEPTION') or result['frames'] != len(manifest['images']):
        parser.error('official scope evaluation requires every registered frame')
    if result['scope'] != manifest['scope'] or [r['frame'] for r in rows] != manifest['frame_numbers']:
        parser.error('short/altered/missing frame scope cannot be presented as the complete video')
    if [r['image_path'] for r in rows] != manifest['images']:
        parser.error('prediction log belongs to different physical frames')
    seq = manifest['sequence']
    protocol = json.loads(Path(manifest['protocol']).read_text())
    dataset = protocol['dataset']
    if dataset not in ('mot17','mot20','dancetrack','animaltrack') or result['task'] != 'mot':
        parser.error('this entrypoint is ordinary MOT; RMOT/OVMOT require their own official evaluator')
    source = verify_trackeval_source()
    key = f'{dataset}/dev/{seq}'
    group = {'video':seq,'key':key,'frames':[Path(p).name for p in manifest['images']]}
    directory = args.run/'official_evaluation'
    gt = directory/'gt'
    view = prepare_gt_view(dataset,[group],{key:{'source_path':manifest['gt_path_for_evaluation_only'],
                           'sha256':digest(Path(manifest['gt_path_for_evaluation_only']))}},gt)
    prediction_root = directory/'trackers'
    target = prediction_root/'groundjev/data'/f'{seq}.txt'
    target.parent.mkdir(parents=True,exist_ok=True)
    mapping = {f:i+1 for i,f in enumerate(manifest['frame_numbers'])}
    with target.open('w') as output:
        for line in (args.run/'predictions.txt').read_text().splitlines():
            fields = line.split(',')
            frame = int(float(fields[0]))
            if frame not in mapping:
                raise ValueError('Prediction contains a frame outside the legal scope')
            fields[0] = str(mapping[frame])
            output.write(','.join(fields)+'\n')
    config = {'GT_FOLDER':str(gt),'TRACKERS_FOLDER':str(prediction_root),'TRACKERS_TO_EVAL':['groundjev'],
              'TRACKER_SUB_FOLDER':'data','PRINT_CONFIG':False,'SKIP_SPLIT_FOL':True,
              'BENCHMARK':'MOT20' if dataset=='mot20' else 'MOT17',
              'SEQ_INFO':view['sequence_lengths'],'DO_PREPROC':True}
    adapter = trackeval.datasets.MotChallenge2DBox(config)
    metrics = [trackeval.metrics.HOTA(),trackeval.metrics.CLEAR({'PRINT_CONFIG':False}),
               trackeval.metrics.Identity({'PRINT_CONFIG':False})]
    evaluator = trackeval.Evaluator({'PRINT_CONFIG':False,'PRINT_RESULTS':False,'TIME_PROGRESS':False,
        'OUTPUT_SUMMARY':False,'OUTPUT_DETAILED':False,'PLOT_CURVES':False,'BREAK_ON_ERROR':True})
    raw,messages = evaluator.evaluate([adapter],metrics)
    if messages[adapter.get_name()]['groundjev'] != 'Success':
        raise RuntimeError('Official evaluation did not complete')
    values = raw[adapter.get_name()]['groundjev']['COMBINED_SEQ']['pedestrian']
    measured = {k:float(np.mean(values['HOTA'][k])) for k in ('HOTA','DetA','AssA')}
    measured.update(IDF1=float(values['Identity']['IDF1']),IDSW=int(values['CLEAR']['IDSW']),
                    FP=int(values['CLEAR']['CLR_FP']),FN=int(values['CLEAR']['CLR_FN']))
    report = {'status':'OFFICIAL_SCOPE_EVALUATION_COMPLETE','dataset':dataset,'sequence':seq,
              'scope':manifest['scope'],'frames':len(rows),'metrics':measured,
              'metric_scale':'fraction, counts for IDSW/FP/FN','trackeval_commit':source['commit'],
              'gt_view':view,'gt_access':'evaluation only','official_test_labels_read':False,
              'fixture_only':args.fixture_only,'model_score_claim':not args.fixture_only,
              'perception_rejected_frames':result.get('perception_rejected_frames',[]),
              'perception_rejection_rate':result.get('perception_rejection_rate',0.),
              'rejected_frames_in_official_denominator':True,
              'end_to_end_fps':result.get('end_to_end_fps'),'association_replay_fps':result.get('association_replay_fps')}
    (args.run/'official_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    result['official_metrics'] = {'status': 'COMPLETE', 'path': str(args.run/'official_metrics.json'),
                                  'scope': manifest['scope'], 'frames': len(rows)}
    (args.run/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='gt_view'}),flush=True)


if __name__ == '__main__':
    main()
