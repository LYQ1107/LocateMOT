"""Generic non-MOT20 cached-box replay with a real final-G2 visual provider.

Only the frozen visual encoder/projector is run to produce ROI features.
There is no language decoding and no regeneration or correction of boxes.
The current checkpoint supports ordinary MOT only.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tools.groundjev.eval_no_mot20.common import RUN,G2_SHA,load,save,job_manifest,sha_file


def replay(job,config):
    import torch
    from PIL import Image
    from locatemot.groundjev.coupled.g2_feature_bridge import G2FeatureBridge
    from locatemot.groundjev.coupled.g2_shared_runtime import CoupledGroundJEVRuntime
    from tools.groundjev.coupled.evaluate_fixed_candidate_clip import load_model
    from tools.groundjev.eval_no_mot20.replay_simple import audit_input
    manifest=job_manifest(job);out=job/'coupled';out.mkdir(exist_ok=True)
    if manifest['task']!='mot':
        save(out/'status.json',{'status':'NOT_SUPPORTED_BY_CURRENT_CHECKPOINT','task':manifest['task']})
        return
    if not (job/'simple/frozen_receipt.json').exists():raise ValueError('Simple must be frozen first')
    with (job/'.coupled.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'frozen_receipt.json').exists():return load(out/'frozen_receipt.json')
        audit=load(RUN/'model_audit.json');checkpoint=Path(audit['coupled']['path']);device=torch.device('cuda:0')
        started=time.perf_counter();model=load_model(checkpoint,device)
        bridge=G2FeatureBridge.from_checkpoint(config['g2_model'],device=str(device),model_sha256=G2_SHA)
        runtime=CoupledGroundJEVRuntime(bridge,model,admit_threshold=.5)
        load_seconds=time.perf_counter()-started
        rows=[json.loads(l) for l in (job/'detector_candidates.jsonl').open()]
        index=0;prior=[];snapshot=out/'latest_runtime.pt'
        if snapshot.exists():
            saved=torch.load(snapshot,map_location=device,weights_only=False)
            if saved['checkpoint_sha256']!=audit['coupled']['sha256']:raise ValueError('Coupled resume checkpoint differs')
            runtime.memory=saved['memory'];index=int(saved['next_index'])
            bridge.last_frame_id=manifest['frame_numbers'][index-1] if index else None
            with (out/'frame_decisions.jsonl').open() as handle:
                for _ in range(index):prior.append(json.loads(next(handle)))
        wall=time.perf_counter();io_seconds=feature_seconds=association_seconds=0.;rejected=[]
        with (out/'frame_decisions.jsonl').open('w') as log,(out/'predictions.txt').open('w') as predictions:
            def write(row):
                log.write(json.dumps(row)+'\n');log.flush()
                for p in row['outputs']:
                    x,y,x2,y2=p['box'];predictions.write(f"{row['frame']},{p['track_id']},{x:.3f},{y:.3f},{x2-x:.3f},{y2-y:.3f},1,-1,-1,-1\n")
                predictions.flush()
            for row in prior:write(row)
            for index in range(index,len(rows)):
                row=rows[index];candidates=row['candidates'];frame=row['frame']
                began=time.perf_counter()
                with Image.open(row['image_path']) as image:rgb=image.convert('RGB')
                io_seconds+=time.perf_counter()-began
                # Bridge/provider timing is recorded by the native visual
                # extractor. It never runs the GroundAnything text decoder.
                began=time.perf_counter()
                result=runtime.step(rgb,[c['box'] for c in candidates],[c['label'] for c in candidates],
                    frame_id=frame,image_key=row['image_path'],task_query=None)
                torch.cuda.synchronize();duration=time.perf_counter()-began
                native=float(result['state'].metadata.get('seconds',0.))
                feature_seconds+=native;association_seconds+=max(0,duration-native)
                outputs=[{'track_id':int(track),'box':list(candidates[i]['box']),'label':candidates[i]['label']}
                         for i,track in result['assignments'].items()]
                if row['perception']['model_response_rejected']:rejected.append(frame)
                decision={'frame':frame,'image_path':row['image_path'],'candidates':candidates,'outputs':outputs,
                          'seconds':duration,'perception':row['perception'],
                          'state':{'decisions':result['decisions'],'assignments':result['assignments'],
                                   'memory_report':result['memory_report'],'persistent_ids':list(runtime.memory.tracks)}}
                write(decision)
                if (index+1)%32==0 or index+1==len(rows):
                    temporary=snapshot.with_suffix('.tmp')
                    torch.save({'memory':runtime.memory,'next_index':index+1,
                                'checkpoint_sha256':audit['coupled']['sha256']},temporary)
                    temporary.replace(snapshot)
        actual=audit_input(job,out)
        save(out/'result.json',{'status':'VIDEO_INFERENCE_COMPLETE_WITH_REJECTED_PERCEPTION' if rejected else 'VIDEO_INFERENCE_COMPLETE',
            'task':'mot','backend':'coupled_fixed_candidate','scope':manifest['scope'],'frames':len(rows),
            'specification':{'task':'mot','vocabulary':manifest['vocabulary'],'expression':None},
            'video_manifest':str(job/'manifest.json'),'all_input_frames_processed':True,'perception_rejected_frames':rejected,
            'perception_rejection_rate':len(rejected)/len(rows),'end_to_end_fps':None,
            'association_replay_fps':(len(rows)-len(prior))/association_seconds if association_seconds else None,
            'complete_video_wall_seconds':time.perf_counter()-wall,'model_initialization_seconds':load_seconds,
            'feature_seconds':feature_seconds,'association_seconds':association_seconds,'image_io_seconds':io_seconds,
            'G2_detection_calls':0,'native_visual_provider':config['g2_model'],'native_visual_provider_SHA':G2_SHA,
            'feature_cache_bytes_on_disk':0,'resumed_completed_frames':len(prior),
            'checkpoint':str(checkpoint),'checkpoint_sha256':audit['coupled']['sha256'],
            'feature_distribution_shift':audit['coupled']['evaluation_feature_shift'],
            'gt_read_by_runtime':False,'future_frames_read_by_runtime':False,
            'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated()})
        # The official scorer runs in OneJev's environment, not the DLM one.
        from tools.groundjev.eval_no_mot20.common import merged_eval_manifest
        save(job/'scoring_manifest.json',merged_eval_manifest(job))
        with (out/'official_scoring.log').open('a') as handle:
            subprocess.run([config['onejev_python'],str(ROOT/'tools/groundjev/evaluate_video.py'),
                '--video',str(job/'scoring_manifest.json'),'--run',str(out)],cwd=ROOT,
                stdout=handle,stderr=subprocess.STDOUT,check=True)
        frozen={'status':'COUPLED_MOT_FIXED_CANDIDATE_EVALUATION_COMPLETE','input':actual,
                'metrics':load(out/'official_metrics.json'),'checkpoint':str(checkpoint),
                'checkpoint_sha256':audit['coupled']['sha256'],'G2_detector_calls':0,
                'prediction_sha256':sha_file(out/'predictions.txt')}
        save(out/'frozen_receipt.json',frozen);return frozen


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(replay(args.job,load(args.config))),flush=True)


if __name__=='__main__':main()
