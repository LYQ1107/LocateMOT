"""Official CPU BYTETracker on all native boxes, with declared constant scores."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.groundjev.run_video import ReplayFrames


def load_tracker(plan):
    source=ROOT/plan['source']
    actual=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    assert actual==plan['upstream_commit']
    subprocess.run(['git','-C',str(source),'diff','--quiet'],check=True)
    sys.path.insert(0,plan['reference_packages'])
    sys.path.append(plan['read_only_torch_packages'])
    # Only import the real tracker subpackage. The upstream root __init__
    # initializes unrelated YOLOX detector/visualization utilities; no detector
    # is used in this fixed-proposal control. No tracker source is replaced.
    package=types.ModuleType('yolox');package.__path__=[str(source/'yolox')]
    sys.modules['yolox']=package
    from yolox.tracker.byte_tracker import BYTETracker
    import numpy,scipy,lap,torch
    assert numpy.__version__=='1.23.5'  # upstream still uses np.float
    assert not torch.cuda.is_initialized()
    return BYTETracker,{'numpy':numpy.__version__,'scipy':scipy.__version__,
        'lap':lap.__version__,'torch':torch.__version__,'tracker_source_commit':actual,
        'upstream_tracker_source_modified':False,'CUDA_initialized':False,
        'loader':'tracker-only upstream package namespace; detector root initialization unused'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    BYTETracker,versions=load_tracker(plan)
    import numpy as np
    import torch
    manifest=json.loads((ROOT/plan['manifest']).read_text())
    assert manifest['task']=='mot' and len(manifest['images'])==plan['frames']
    width,height=manifest['image_size']
    tracker=BYTETracker(types.SimpleNamespace(**plan['tracker_arguments']),frame_rate=manifest['fps'])
    args.out.mkdir(parents=True,exist_ok=False)
    state={'status':'RUNNING_OFFICIAL_CPU_BYTETRACK_UNSCORED_REFERENCE',
        'owner_pid':os.getpid(),'physical_gpus':[],'upstream_commit':plan['upstream_commit'],
        'implementation_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'plan':str(args.plan),'versions':versions,'GT_read_by_runtime':False,'full_goal_complete':False}
    (args.out/'owner_state.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(state),flush=True)
    source=ReplayFrames(ROOT/plan['candidates'])
    frames,rejected,count,predictions,seconds=[],[],0,0,0.
    try:
        with (args.out/'frame_decisions.jsonl').open('w') as trace,(args.out/'predictions.txt').open('w') as mot:
            for number,path in zip(manifest['frame_numbers'],manifest['images']):
                row=source.frame(number)
                assert row['image_path']==path
                candidates=row['candidates']
                assert all(c.get('native_score') is None for c in candidates)
                boxes=np.asarray([list(c['box'])+[1.] for c in candidates],dtype=np.float64).reshape(-1,5)
                original_boxes=boxes[:,:4].copy()
                start=time.perf_counter()
                tracks=tracker.update(boxes,(height,width),(height,width))
                seconds+=time.perf_counter()-start
                assert np.array_equal(boxes[:,:4],original_boxes)
                outputs=[]
                for track in tracks:
                    box=list(map(float,track.tlbr))
                    assert all(np.isfinite(box))
                    outputs.append({'track_id':int(track.track_id),'box':box,'label':'person','score':1.})
                    x,y,x2,y2=box
                    mot.write(f'{number},{track.track_id},{x:.6f},{y:.6f},{x2-x:.6f},{y2-y:.6f},1,-1,-1,-1\n')
                clean=[{k:v for k,v in c.items() if k not in ('native_region_feature','temporal_region_feature')} for c in candidates]
                trace.write(json.dumps({'frame':number,'image_path':path,'candidates':clean,
                    'outputs':outputs,'perception':row['perception'],
                    'state':{'association_backend':'official BYTETracker unscored reference',
                        'persistent_ids':[int(t.track_id) for t in tracker.tracked_stracks+tracker.lost_stracks]}})+'\n')
                if row['perception'].get('model_response_rejected'):
                    rejected.append(number)
                frames.append(number);count+=len(candidates);predictions+=len(outputs)
        assert frames==manifest['frame_numbers'] and count==plan['native_candidates']
        assert len(rejected)==plan['parser_rejected_frames'] and not torch.cuda.is_initialized()
        memory={'frame_id':tracker.frame_id,'tracked_and_lost':[{'id':int(t.track_id),
            'mean':t.mean.tolist(),'covariance':t.covariance.tolist(),'state':int(t.state),
            'last_frame':int(t.frame_id),'start_frame':int(t.start_frame)}
            for t in tracker.tracked_stracks+tracker.lost_stracks]}
        (args.out/'final_tracker_state.json').write_text(json.dumps(memory)+'\n')
        report={'status':'VIDEO_INFERENCE_COMPLETE_WITH_REJECTED_PERCEPTION' if rejected else 'VIDEO_INFERENCE_COMPLETE',
            'task':'mot','backend':'official_bytetrack_unscored_reference','scope':manifest['scope'],
            'frames':len(frames),'all_input_frames_processed':True,'native_candidates':count,
            'outputs':predictions,'checkpoint':None,'upstream_commit':plan['upstream_commit'],
            'tracker_arguments':plan['tracker_arguments'],'actual_max_time_lost_frames':tracker.max_time_lost,
            'score_policy':plan['input_score_policy'],'low_score_branch_input_count':0,
            'output_policy':plan['output_policy'],'Kalman_output_boxes_retained':True,
            'association_core_CPU_seconds':seconds,'end_to_end_fps':None,'association_replay_fps':None,
            'gt_read_by_runtime':False,'future_frames_read_by_runtime':False,
            'perception_rejected_frames':rejected,'perception_rejection_rate':len(rejected)/len(frames),
            'all_original_native_candidates_retained':True,
            'shared_input_features_source':str(ROOT/plan['candidates']),
            'replay_feature_vectors_omitted_from_log':True,'new_native_feature_archive':False,
            'versions':versions,'source_scope_limit':plan['scope'],'official_metrics':'PENDING_OFFICIAL_EVALUATOR'}
        (args.out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        subprocess.run(['python3','tools/groundjev/evaluate_video.py','--video',str(ROOT/plan['manifest']),
            '--run',str(args.out)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
        official=json.loads((args.out/'official_metrics.json').read_text())
        assert official['frames']==1203 and official['perception_rejected_frames']==rejected
        state.update(status='OFFICIAL_UNSCORED_BYTETRACK_COMPLETE_FULL_VIDEO_VALIDATED',
            frames=len(frames),native_candidates=count,metrics=official['metrics'],
            original_all_frames_candidates_and_rejections_retained=True,
            full_YOLOX_ByteTrack_published_result_reproduction=False)
        (args.out/'owner_state.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps(state),flush=True)
    except Exception as error:
        state.update(status='UNSCORED_BYTETRACK_REFERENCE_FAILED',error=repr(error))
        (args.out/'owner_state.json').write_text(json.dumps(state,indent=2)+'\n')
        raise
    finally:
        source.close()


if __name__=='__main__':
    main()
