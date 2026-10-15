"""Validate frozen input sources and existing baseline evidence before new scores."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.groundjev.run_video import ReplayFrames


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    rows={}
    for name,job in plan['scopes'].items():
        manifest=json.loads((ROOT/job['manifest']).read_text())
        assert len(manifest['images'])==len(manifest['frame_numbers'])==job['frames']
        assert all(Path(p).is_file() for p in manifest['images'])
        if manifest['task']=='ovmot':
            assert len(manifest['vocabulary'])==1203 and manifest.get('expression') is None
        source=ReplayFrames(ROOT/job['candidates'])
        count,rejections=0,[]
        try:
            for number,path in zip(manifest['frame_numbers'],manifest['images']):
                row=source.frame(number)
                assert row['frame']==number and row['image_path']==path
                count+=len(row['candidates'])
                if row['perception'].get('model_response_rejected'):
                    rejections.append(number)
        finally:
            source.close()
        assert count==job['candidates_count'],(name,count,job['candidates_count'])
        baseline=None
        if job['baseline_run']:
            run=ROOT/job['baseline_run']
            runtime=json.loads((run/'result.json').read_text())
            official=json.loads((run/'official_metrics.json').read_text())
            assert runtime['frames']==official['frames']==job['frames']
            assert runtime['all_input_frames_processed'] and runtime['backend']=='full'
            assert not runtime['gt_read_by_runtime'] and not runtime['future_frames_read_by_runtime']
            assert Path(runtime['checkpoint']).resolve()==(ROOT/plan['checkpoints']['before']).resolve()
            assert Path(runtime['shared_input_features_source']).resolve()==(ROOT/job['candidates']).resolve()
            assert runtime['perception_rejected_frames']==official['perception_rejected_frames']==rejections
            baseline={'run':str(run),'metrics':official['metrics'],'unchanged_existing_artifacts':True}
        rows[name]={'frames':job['frames'],'native_candidates':count,'parser_rejected_frames':rejections,
            'manifest':job['manifest'],'source':job['candidates'],'before_baseline':baseline}
    dependencies={}
    for name in ('training','gpu5_release'):
        dependency=plan[name];pid,tick=dependency['identity'];proc=Path('/proc')/str(pid)
        if proc.exists():
            assert proc.joinpath('cwd').resolve()==ROOT
            assert dependency['script'] in proc.joinpath('cmdline').read_bytes().decode().split('\0')
            assert proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]==tick
            dependencies[name]={'identity':dependency['identity'],'live_process_identity_verified':True}
        else:
            state=json.loads((ROOT/dependency['directory']/'owner_state.json').read_text())
            assert state['status']==dependency['required_terminal_status']
            dependencies[name]={'identity':dependency['identity'],'terminal_success_verified':True}
    report={'status':'HISTORY_REFRESH_FOUR_SCOPE_SOURCES_AND_DEPENDENCIES_VALIDATED',
        'plan':str(args.plan),'scopes':rows,'dependencies':dependencies,
        'new_model_calls_or_feature_archive':False,'official_scores_or_runtime_policy_changed':False,
        'full_goal_complete':False}
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
