"""One CPU owner for the registered full-video learned hybrid retrieval control."""
import argparse
import gzip
from itertools import zip_longest
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    from tools.groundjev.analyze_natural_recovery_case import analyze
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    assert plan['combined_registered_feature_upper_bytes']<plan['global_temporary_feature_budget_bytes']
    audit=json.loads((ROOT/plan['coverage_audit']).read_text())
    assert audit['status']=='FULL_NATURAL_FROZEN_TRAJECTORY_NATIVE_HYBRID_SHORTLIST_COVERAGE_VALIDATED'
    out=ROOT/plan['output'];out.mkdir(parents=True,exist_ok=False)
    (out/'registration.json').write_text(json.dumps(plan,indent=2)+'\n')
    state=dict(status='REGISTERED_FULL_NATURAL_HYBRID_CPU_CONTROL',owner_pid=os.getpid(),
        owner_starttick=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        completed={},new_GPU_or_optimizer_updates=0,full_goal_complete=False)
    def save():
        tmp=out/'owner_state.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(out/'owner_state.json')
        print(json.dumps(dict(status=state['status'],completed=list(state['completed']))),flush=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
        PYTHONPATH=str(ROOT/'external/OneJev'),PYTHONUNBUFFERED='1')
    python=str(ROOT/'.venv-onejev-py310/bin/python');run=out/'hybrid16_16'
    def command(stage,argv):
        state.update(status=stage,command=argv);save()
        with (out/(stage+'.log')).open('w') as log:
            child=subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            state['child_pid']=child.pid;save()
            for line in child.stdout:log.write(line);log.flush();print(line,end='',flush=True)
            code=child.wait()
        state.pop('child_pid',None);assert code==0,f'{stage} exited{code}; preserve completed prefix'
    save()
    try:
        command('COMPLETE_NATIVE_CPU_HYBRID_INFERENCE',[python,'tools/groundjev/run_video.py','--task','mot',
            '--video',str(ROOT/plan['manifest']),'--backend','compact','--compact-device','cpu',
            '--checkpoint',str(ROOT/plan['checkpoint']),'--compact-shortlist-policy','hybrid16_16',
            '--candidate-replay',str(ROOT/plan['source']),'--omit-replay-feature-vectors',
            '--memory-checkpoint-format','binary','--memory-export-interval','0','--out',str(run)])
        command('OFFICIAL_FULL_SCOPE_TRACKING_METRICS',[python,'tools/groundjev/evaluate_video.py',
            '--video',str(ROOT/plan['manifest']),'--run',str(run)])
        command('POSTHOC_IDENTITY_EVENTS',[python,'tools/groundjev/analyze_tracking_events.py',
            '--video',str(ROOT/plan['manifest']),'--run',str(run)])
        manifest=json.loads((ROOT/plan['manifest']).read_text());recovery=analyze(manifest,run)
        runtime=json.loads((run/'result.json').read_text());official=json.loads((run/'official_metrics.json').read_text())
        baseline=ROOT/plan['baseline'];old_runtime=json.loads((baseline/'result.json').read_text())
        old_official=json.loads((baseline/'official_metrics.json').read_text())
        assert runtime['all_input_frames_processed'] and runtime['frames']==official['frames']==plan['frames']
        assert runtime['backend']=='compact' and runtime['decision_device']=='cpu'
        assert runtime['compact_shortlist_policy']=='hybrid16_16' and old_runtime.get('compact_shortlist_policy','geometry')=='geometry'
        assert runtime['memory_policy']==old_runtime['memory_policy']=='gated'
        assert runtime['lost_seconds']==old_runtime['lost_seconds']==3
        assert Path(runtime['checkpoint']).resolve()==Path(old_runtime['checkpoint']).resolve()==(ROOT/plan['checkpoint']).resolve()
        assert runtime['shared_input_features_source']==old_runtime['shared_input_features_source']==str(ROOT/plan['source'])
        assert not runtime['gt_read_by_runtime'] and not runtime['future_frames_read_by_runtime']
        assert runtime['perception_rejected_frames']==old_runtime['perception_rejected_frames']
        counts=0;predictions=[]
        with (run/'frame_decisions.jsonl').open() as fresh,(baseline/'frame_decisions.jsonl').open() as previous:
            for new,old in zip_longest(fresh,previous):
                assert new is not None and old is not None
                new,old=json.loads(new),json.loads(old)
                for key in ('frame','image_path','candidates','perception'):assert new[key]==old[key]
                assert new['state']['timestamp']==old['state']['timestamp']
                counts+=len(new['candidates'])
                for p in new['outputs']:
                    x,y,x2,y2=p['box'];predictions.append(f'{new["frame"]},{p["track_id"]},{x:.3f},{y:.3f},{x2-x:.3f},{y2-y:.3f},1,-1,-1,-1\n')
        assert counts==plan['candidates']
        raw=''.join(predictions).encode();assert raw==(run/'predictions.txt').read_bytes()
        decoded=int(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames',
            '-show_entries','stream=nb_read_frames','-of','default=noprint_wrappers=1:nokey=1',str(run/'tracks.mp4')],text=True))
        assert decoded==plan['frames']
        snapshot=out/'completed_result';snapshot.mkdir()
        compressed=gzip.compress(raw,mtime=0);assert gzip.decompress(compressed)==raw
        (snapshot/'predictions.txt.gz').write_bytes(compressed)
        for name in ('result.json','official_metrics.json','natural_recovery_case.json'):
            (snapshot/name).write_bytes((run/name).read_bytes())
        events=json.loads((run/'association_events.json').read_text());events.pop('events')
        (snapshot/'association_events_summary.json').write_text(json.dumps(events,indent=2)+'\n')
        report=dict(status='FULL_NATURAL_LEARNED_HYBRID32_CONTROL_COMPLETE_AND_VALIDATED',
            plan=str(args.plan),frames=runtime['frames'],native_candidates=counts,
            encoded_video_decoded_frames=decoded,
            all_frozen_candidate_metadata_and_timestamps_exact=True,
            baseline_metrics=old_official['metrics'],hybrid_metrics=official['metrics'],
            HOTA_change_percentage_points=100*(official['metrics']['HOTA']-old_official['metrics']['HOTA']),
            natural_recovery_counts=recovery['counts'],new_GPU_or_optimizer_updates=0,
            fixed_shared_checkpoint=True,no_runtime_GT_or_future_correction=True,
            isolated_speed_or_end_to_end_claim=False,three_task_or_Full_performance_claim=False,
            ordinary_model_performance_achieved=False,full_goal_complete=False)
        (snapshot/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
        assert sum(p.stat().st_size for p in snapshot.rglob('*') if p.is_file())<plan['small_reports_and_compressed_predictions_reserve_bytes']
        state.update(status=report['status'],completed=report);save()
    except Exception as error:
        state.update(status='NATURAL_HYBRID_CONTROL_FAILED',error=repr(error));save();raise


if __name__=='__main__':main()
