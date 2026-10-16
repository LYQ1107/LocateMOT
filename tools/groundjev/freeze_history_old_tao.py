"""Preserve the newly completed old-history native TAO unit and exact scorer input."""
import argparse
import gzip
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    from tools.groundjev.evaluate_multitask import normalize_label
    from tools.groundjev.run_history_refresh_controls import validate_run
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--first3',type=Path,required=True);args=parser.parse_args()
    plan=json.loads((ROOT/'configs/groundjev/full_history_refresh600_controls_v1.json').read_text())
    root=ROOT/plan['output_root'];owner=json.loads((root/'old_worker/owner_state.json').read_text())
    assert owner['status']=='HISTORY_REFRESH_WORKER_ALL_REGISTERED_UNITS_VALIDATED'
    job=plan['scopes']['tao_raw_full40'];run=root/'old_tao_raw_full40'
    record=validate_run(run,job,ROOT/plan['checkpoints']['old'])
    assert record==owner['completed']['old_tao_raw_full40']
    manifest=json.loads((ROOT/job['manifest']).read_text())
    annotation=json.loads(Path(manifest['annotation_path_for_evaluation_only']).read_text())
    categories=annotation['categories'];by_synset={c['synset']:c['id'] for c in categories}
    aliases={normalize_label(a):c['id'] for c in categories for a in [c['name']]+c.get('synonyms',[])}
    for c in json.loads(Path(manifest['vocabulary_path']).read_text())['categories']:
        if c['synset'] in by_synset:
            for alias in [c['name']]+c['synonyms']:aliases[normalize_label(alias)]=by_synset[c['synset']]
    raw=[];official=[];rows=0
    with (run/'frame_decisions.jsonl').open() as f:
        for index,line in enumerate(f):
            row=json.loads(line);rows+=1
            assert row['frame']==manifest['frame_numbers'][index]
            for p in row['outputs']:
                x,y,x2,y2=p['box'];tid=p['track_id']
                raw.append(f'{row["frame"]},{tid},{x:.3f},{y:.3f},{x2-x:.3f},{y2-y:.3f},1,-1,-1,-1\n')
                official.append(dict(image_id=manifest['image_ids'][index],video_id=manifest['video_id'],
                    track_id=tid,category_id=aliases.get(normalize_label(p['label']),-1),
                    bbox=[x,y,x2-x,y2-y],score=1.))
    raw_bytes=''.join(raw).encode();official_bytes=json.dumps(official).encode()
    assert rows==40 and raw_bytes==(run/'predictions.txt').read_bytes()
    assert official_bytes==(run/'official_evaluation/trackers/groundjev/data/predictions.json').read_bytes()
    before=json.loads((root/'refreshed_worker/owner_state.json').read_text())['completed']['before_tao_raw_full40']
    assert Path(before['checkpoint']).resolve()==(ROOT/plan['checkpoints']['before']).resolve()
    baseline=json.loads((Path(before['run'])/'official_metrics.json').read_text())
    assert baseline['metrics']==before['official_metrics'] and baseline['base_novel']==before['base_novel']
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'predictions.txt.gz').write_bytes(gzip.compress(raw_bytes,mtime=0))
    (args.out/'official_predictions.json.gz').write_bytes(gzip.compress(official_bytes,mtime=0))
    for name in ('result.json','official_metrics.json'):(args.out/name).write_bytes((run/name).read_bytes())
    report=dict(status='OLD_HISTORY_NATIVE_TAO40_COMPLETE_EXACT_RAW_AND_OFFICIAL_PREDICTIONS_FROZEN',
        completed=record,before=before,official_prediction_count=len(official),
        raw_and_official_inputs_exactly_reconstructed_from_all_outputs=True,
        unmapped_category_minus1_predictions_retained=True,
        official_evaluated_categories=baseline['evaluated_categories'],
        original_completed_scorer_result_retained_not_recomputed=True,
        annotation_categories_read_posthoc_only_not_new_runtime_inference=True,
        refreshed_TAO_complete=False,common_model_promoted=False,new_GPU_or_optimizer_updates=0,
        combined_registered_feature_upper_bytes=26834314206,global_temporary_feature_budget_bytes=26843545600,
        report_reservation='Shares the existing aggregate1MiB native history first3/report reservation.',
        scope='One complete40annotation-step video, not full TAO. All1203 declared input classes; NovelClsA remains zero. No end-to-end FPS or full-dataset claim.',
        full_goal_complete=False)
    (args.out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    total=sum(p.stat().st_size for directory in (args.first3,args.out) for p in directory.rglob('*') if p.is_file())
    assert total<1048576,total
    print(json.dumps(dict(status=report['status'],combined_history_report_bytes=total,
        before_TETA=before['official_metrics']['TETA'],old_TETA=record['official_metrics']['TETA'],
        official_predictions=len(official))),flush=True)


if __name__=='__main__':main()
