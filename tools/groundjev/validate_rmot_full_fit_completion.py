"""Validate all completed native RMOT observations and exact input archives."""
import argparse
import gzip
from itertools import zip_longest
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    from tools.groundjev.run_video import ReplayFrames
    from tools.groundjev.service_process_group import members
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'outputs/groundjev/training_data/rmot_allfit1814_query18_shared_gpu7_v2'
    export=ROOT/'outputs/groundjev/training_data/rmot_allfit1814_query18_full_v3_groups_v2_shared_gpu7'
    producer=json.loads((source/'owner_state.json').read_text())
    exported=json.loads((export/'owner_state.json').read_text())
    assert producer['status']=='ALL1814_FIT_PHYSICAL_FRAMES_18_NATIVE_QUERY_ROLLOUTS_COMPLETE'
    assert exported['status']=='ALL18_NATIVE_RMOT_QUERY_ROLLOUTS_FULL_V3_GROUPS_AND_CPU_PLANS_VALIDATED'
    assert len(producer['completed'])==len(exported['completed'])==18
    assert producer['completed'].keys()==exported['completed'].keys()
    assert all(not members(g) for g in producer['owned_service_groups'])
    units={};physical=set();archive_total=0
    for name,unit in producer['completed'].items():
        run=Path(unit['run']);manifest=json.loads(Path(unit['manifest']).read_text())
        runtime=json.loads((run/'result.json').read_text());summary=exported['completed'][name]
        assert runtime['frames']==unit['frames']==summary['counts']['frames']==len(manifest['images'])
        assert runtime['all_input_frames_processed'] and not runtime['gt_read_by_runtime'] and not runtime['future_frames_read_by_runtime']
        assert runtime['specification']==dict(task='rmot',vocabulary=manifest['vocabulary'],expression=manifest['expression'])
        assert summary['dropped_rows']==0 and summary['all_actor_outputs_states_exact']
        assert summary['status']=='REAL_CLIP_FULL_V3_GROUP_REFS_PIXELS_AND_ALL_NATIVE_CPU_ROWS_VALIDATED'
        archive=Path(runtime['native_candidate_archive']);replay=ReplayFrames(archive)
        raw_predictions=[];count=0;rejected=[]
        try:
            with (run/'frame_decisions.jsonl').open() as trace:
                for expected,line in zip_longest(zip(manifest['frame_numbers'],manifest['images']),trace):
                    assert expected is not None and line is not None
                    frame,path=expected;row=json.loads(line);original=replay.frame(frame)
                    assert row['frame']==frame and row['image_path']==path==original['image_path']
                    assert row['candidates']==[{k:v for k,v in c.items() if k not in ('native_region_feature','temporal_region_feature')} for c in original['candidates']]
                    if row['perception'].get('model_response_rejected'):rejected.append(frame)
                    count+=len(original['candidates']);physical.add((unit['video'],frame))
                    for prediction in row['outputs']:
                        x,y,x2,y2=prediction['box'];raw_predictions.append(f'{frame},{prediction["track_id"]},{x:.3f},{y:.3f},{x2-x:.3f},{y2-y:.3f},1,-1,-1,-1\n')
        finally:replay.close()
        assert count==unit['native_candidates']==summary['counts']['candidates']
        assert rejected==runtime['perception_rejected_frames'] and len(rejected)==unit['rejected_frames']
        raw=''.join(raw_predictions).encode();assert raw==(run/'predictions.txt').read_bytes()
        packed=sum(p.stat().st_size for p in (archive.parent/'native_region_features').glob('*.npz'))
        archive_bytes=archive.stat().st_size+packed
        assert archive_bytes==unit['archive_bytes']==runtime['native_candidate_archive_bytes']
        archive_total+=archive_bytes
        decoded=int(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames',
            '-show_entries','stream=nb_read_frames','-of','default=noprint_wrappers=1:nokey=1',str(run/'tracks.mp4')],text=True))
        assert decoded==unit['frames']
        compressed=gzip.compress(raw,mtime=0);assert gzip.decompress(compressed)==raw
        (args.out/(name+'_predictions.txt.gz')).write_bytes(compressed)
        units[name]=dict(frames=decoded,candidates=count,rejected_frames=rejected,
            exact_packed_FP32_archive_bytes=archive_bytes,supervised_rows=summary['counts']['rows'],
            semantic_membership_rows=summary['counts'].get('family:query_membership',0),
            semantic_presence_rows=summary['counts'].get('family:empty_query_presence',0),
            all_actor_states_outputs_and_native_pixels_exact=summary['all_actor_outputs_states_exact'],
            all_native_CPU_rows_zero_drops=True,raw_predictions_reconstructed_and_gzip_exact=True)
    prior=json.loads((ROOT/'configs/groundjev/rmot_native72_20261014.json').read_text())
    old={(c['video'],f) for c in prior['clips'] for f in c['frames']}
    assert len(physical)==1814 and len(old)==72 and old<=physical
    assert sum(u['frames'] for u in units.values())==3628
    report=dict(status='ALL18_NATIVE_RMOT_ROLLOUTS_ARCHIVES_RAW_PREDICTIONS_VIDEOS_AND_GROUP_EXPORTS_VALIDATED',
        units=units,physical_frames=1814,new_physical_frames_beyond_old72=1742,query_observations=3628,
        actual_native_archive_bytes=archive_total,supervised_rows=sum(u['supervised_rows'] for u in units.values()),
        future_eligible_semantic_rows=sum(u['semantic_membership_rows']+u['semantic_presence_rows'] for u in units.values()),
        source_and_export_original_exit_codes={'17236':0,'78069':0},
        original_owned_DLM_and_ROI_service_groups_empty=True,new_optimizer_updates=0,
        auxiliary_unknown_birth_proxies_not_automatically_trained=True,
        no_new_image_or_feature_archive=True,full_goal_complete=False)
    (args.out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='units'}),flush=True)


if __name__=='__main__':main()
