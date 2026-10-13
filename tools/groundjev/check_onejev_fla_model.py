"""Paired real multimodal OneJev probabilities with reference versus isolated FLA."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def worker(mode,out):
    if mode=='fla':sys.path.insert(0,str(ROOT/'.tools/onejev-fla052'))
    import torch
    from PIL import Image
    from locatemot.groundjev.decision import OneJevDecision,SimpleDecision
    from locatemot.groundjev.tracker import OnlineTracker
    decision=OneJevDecision(ROOT/'models/OneJev-0.8B',evidence_format='v3_crop_motion')
    history=OnlineTracker(SimpleDecision())
    source=ROOT/'outputs/groundjev/runs/mot17_first_native_clip_repair2/frame_decisions.jsonl'
    rows=[json.loads(s) for s in source.read_text().splitlines()][:3]
    records=[]
    for index,row in enumerate(rows):
        with Image.open(row['image_path']) as original:image=original.convert('RGB')
        candidates=row['candidates'][:3];timestamp=row['frame']/30
        tracks=list(history.tracks.values())
        request,media=decision.build_request(image,candidates,tracks,timestamp,{'task':'mot','vocabulary':['person'],'expression':None})
        if index==0:decision.engine.decide(request,media=media)
        torch.cuda.synchronize();start=time.perf_counter()
        response,metadata=decision.engine.decide(request,media=media)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        records.append({'frame':row['frame'],'candidate_count':len(candidates),'history_count':len(tracks),
            'answers':{k:v.model_dump() for k,v in response.answers.items()},'seconds':elapsed})
        history.step(image,candidates,timestamp,{'task':'mot'})
    report={'mode':mode,'records':records,'checkpoint':str(ROOT/'models/OneJev-0.8B'),
        'GT_read':False,'history':'same GT-free geometry-control history in both subprocesses',
        'official_fork_check':'unchanged automatic check and original0.05tolerance',
        'peak_GPU_allocated_bytes':torch.cuda.max_memory_allocated(),'GPU':torch.cuda.get_device_name(0),
        'scope':'three multimodal requests with three real proposals each; not video performance'}
    (out/(mode+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'mode':mode,'request_seconds':[r['seconds'] for r in records]}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',choices=['reference','fla']);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if args.worker:worker(args.worker,args.out);return
    for mode in ['reference','fla']:
        command=[sys.executable,str(Path(__file__).resolve()),'--worker',mode,'--out',str(args.out)]
        with (args.out/(mode+'.log')).open('w') as log:
            child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,cwd=ROOT)
            for line in child.stdout:log.write(line);log.flush();print(line,end='',flush=True)
            if child.wait():raise RuntimeError(f'Preserve first {mode} failure; do not adopt the accelerator')
    a=json.loads((args.out/'reference.json').read_text());b=json.loads((args.out/'fla.json').read_text())
    differences=[];non_numeric_differences=[]
    def compare(x,y,path):
        if isinstance(x,dict):
            assert set(x)==set(y)
            for key in x:compare(x[key],y[key],path+'/'+key)
        elif isinstance(x,list):
            assert len(x)==len(y)
            for i,(u,v) in enumerate(zip(x,y)):compare(u,v,path+'/'+str(i))
        elif isinstance(x,(int,float)) and not isinstance(x,bool):differences.append(abs(x-y))
        elif x!=y:non_numeric_differences.append({'field':path,'reference':x,'fla':y})
    for x,y in zip(a['records'],b['records']):
        assert (x['frame'],x['candidate_count'],x['history_count'])==(y['frame'],y['candidate_count'],y['history_count'])
        compare(x['answers'],y['answers'],str(x['frame']))
    maximum=max(differences,default=0.);accepted=maximum<=1e-3 and not non_numeric_differences
    report={'status':'REAL_ONEJEV_ACCELERATOR_PARITY_PASS' if accepted else 'REAL_ONEJEV_ACCELERATOR_PARITY_REJECTED',
        'maximum_typed_numeric_answer_difference':maximum,'preregistered_tolerance':1e-3,
        'changed_typed_choices':non_numeric_differences,'reference_request_seconds':[r['seconds'] for r in a['records']],
        'fla_request_seconds':[r['seconds'] for r in b['records']],
        'component_speed_ratio':sum(r['seconds'] for r in a['records'])/sum(r['seconds'] for r in b['records']),
        'scope':a['scope'],'GT_read':False,'original_training_changed':False,'end_to_end_speed_claim':False,
        'timing_conditions':'component timing under concurrent project/foreign workloads; startup and first warmup excluded'}
    (args.out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    if not accepted:raise RuntimeError('FLA deviates beyond the preregistered model comparison')


if __name__=='__main__':main()
