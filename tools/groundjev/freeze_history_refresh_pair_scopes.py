"""Freeze complete before/old/refreshed native units without repeated inference."""
import argparse
import gzip
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    from tools.groundjev.run_history_refresh_controls import validate_run
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    plan=json.loads((ROOT/'configs/groundjev/full_history_refresh600_controls_v1.json').read_text())
    training=json.loads((ROOT/plan['training']['directory']/'owner_state.json').read_text())
    assert training['status']=='PAIRED_HISTORY_REFRESH600_TRAINING_AND_MM_RELOAD_COMPLETE'
    root=ROOT/plan['output_root'];owners={arm:json.loads((root/(arm+'_worker')/'owner_state.json').read_text()) for arm in ('old','refreshed')}
    trained={}
    for arm in owners:
        t=training['completed'][arm];assert t['done']['steps']==600 and t['done']['examples']==2400
        assert t['initial_FP32']['source_step']==600 and t['initial_FP32']['initialization_max_difference']==0
        assert t['native_MM_reload']['repeated_reload_probability_max_difference']==0
        trained[arm]=dict(training=t['training'],done=t['done'],initial_FP32=t['initial_FP32'],
            native_MM_reload_status=t['native_MM_reload']['status'],full_state_receipt=t['latest_training_state'])
    completed={}
    for scope in ('mot_full1203','rmot_single800','rmot_multi800'):
        job=plan['scopes'][scope];manifest=json.loads((ROOT/job['manifest']).read_text())
        for arm in ('before','old','refreshed'):
            run=ROOT/job['baseline_run'] if arm=='before' else root/(arm+'_'+scope)
            record=validate_run(run,job,ROOT/plan['checkpoints'][arm])
            if arm!='before':assert owners[arm]['completed'][arm+'_'+scope]['official_metrics']==record['official_metrics']
            raw=[];official=[]
            with (run/'frame_decisions.jsonl').open() as f:
                for index,row in enumerate(map(json.loads,f),1):
                    precision=3 if manifest['task']=='mot' else 6
                    for prediction in row['outputs']:
                        x,y,x2,y2=prediction['box'];tid=prediction['track_id']
                        raw.append(f'{row["frame"]},{tid},{x:.3f},{y:.3f},{x2-x:.3f},{y2-y:.3f},1,-1,-1,-1\n')
                        official.append(f'{index},{tid},{x:.{precision}f},{y:.{precision}f},{x2-x:.{precision}f},{y2-y:.{precision}f},1,-1,-1,-1\n')
            text=''.join(raw).encode();assert text==(run/'predictions.txt').read_bytes()
            sequence=manifest['sequence'] if manifest['task']=='mot' else 'query_'+manifest['sequence']
            assert ''.join(official)==(run/'official_evaluation/trackers/groundjev/data'/(sequence+'.txt')).read_text()
            destination=args.out/(arm+'_'+scope);destination.mkdir()
            (destination/'predictions.txt.gz').write_bytes(gzip.compress(text,mtime=0))
            for name in ('result.json','official_metrics.json'):(destination/name).write_bytes((run/name).read_bytes())
            completed[arm+'_'+scope]=dict(record,raw_and_official_tracker_inputs_exactly_reconstructed_from_every_output=True,
                original_completed_scorer_result_retained_not_recomputed=True,predictions_snapshot=str(destination/'predictions.txt.gz'))
    report=dict(status='NATIVE_HISTORY_REFRESH_BEFORE_OLD_REFRESHED_FIRST3_ALL9_UNITS_FROZEN',
        completed=completed,training_receipts=trained,remaining_unfrozen_TAO_arms=['old','refreshed'],
        protocol='Registered additional600updates/2400questions each from identical600FP32 masters; same per-step task/family order and400changed history rows, not token/FLOP matched. Each arm has one checkpoint for all tasks.',
        integrity='Fresh native candidate order/geometry/labels/scores/frame/GT isolation/decoded video/final memory checks plus exact raw/official tracker-input reconstruction. Completed owner metrics compared to unchanged official result; no repeated model/scorer calls.',
        new_GPU_or_optimizer_updates=0,combined_registered_feature_upper_bytes=26834314206,
        global_temporary_feature_budget_bytes=26843545600,new_metadata_reservation_bytes=1048576,
        common_model_promoted=False,full_goal_complete=False)
    (args.out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    actual=sum(p.stat().st_size for p in args.out.rglob('*') if p.is_file());assert actual<1048576
    print(json.dumps(dict(status=report['status'],units=len(completed),bytes=actual,
        HOTA_percent={k:100*r['official_metrics']['HOTA'] for k,r in completed.items()})),flush=True)


if __name__=='__main__':main()
