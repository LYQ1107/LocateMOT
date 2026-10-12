"""Reference-only full-vocabulary1/2/4-process throughput and proposal parity."""
from __future__ import annotations
import argparse,hashlib,json,multiprocessing as mp,os,sys,time
from pathlib import Path
ROOT=Path('/data2/user/LocateMOT');sys.path.insert(0,str(ROOT))
CHECKPOINT=ROOT/'outputs/unified/u2/perception_adaptation/training/epoch_1.pth'
SHA='1d9077ae603f40de0e955a11a88083f914f98faa967bc46cdd06bcdfb5129688'

def worker(index,count,gpu,records,entities,barrier,out):
 import torch
 from locatemot.unified.runtime.grounding_inference import load_grounding_runtime,run_grounding_chunked
 from locatemot.unified.runtime.u2_frozen_text_cache import frozen_text_cache
 torch.set_num_threads(2)
 runtime=load_grounding_runtime(checkpoint=CHECKPOINT,device=f'cuda:{gpu}',max_per_img=900)
 assert runtime.checkpoint_sha256==SHA
 selected=records[index::count]
 signatures=[];backbones={'calls':0}
 def backbone(module,inputs,output):backbones['calls']+=1
 hook=runtime.model.backbone.register_forward_hook(backbone)
 with frozen_text_cache(runtime) as stats:
  run_grounding_chunked(runtime,selected[0]['image'],entities)
  torch.cuda.synchronize(runtime.device);backbones['calls']=0
  barrier.wait()
  start=time.perf_counter()
  for record in selected:
   result=run_grounding_chunked(runtime,record['image'],entities)
   predictions=result.pred_instances
   digest=hashlib.sha256()
   for name in ('bboxes','scores','labels'):
    value=getattr(predictions,name).detach().cpu().contiguous()
    digest.update(name.encode());digest.update(str(value.dtype).encode());digest.update(value.numpy().tobytes())
   signatures.append({'key':record['key'],'signature':digest.hexdigest()})
  torch.cuda.synchronize(runtime.device);elapsed=time.perf_counter()-start
 hook.remove()
 assert backbones['calls']==len(selected)
 payload={'index':index,'count':count,'images':len(selected),'seconds':elapsed,'max_memory_allocated_bytes':torch.cuda.max_memory_allocated(runtime.device),'signatures':signatures,'backbone_calls':backbones['calls'],'text_cache':stats}
 Path(out,f'worker_{index}.json').write_text(json.dumps(payload,indent=2)+'\n')

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,default=1);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
 args.out.mkdir(parents=True,exist_ok=True)
 protocol=json.loads((ROOT/'outputs/unified/u2/protocol/tao_portability.json').read_text())
 data=json.loads(Path(protocol['annotation_path']).read_text());seen={};records=[]
 for row in data['images']:
  source=Path(row['file_name']).parts[1]
  if source not in seen:
   seen[source]=True;records.append({'key':f"tao/{row['id']}",'image':str(Path(protocol['image_root'])/row['file_name'])})
 del data
 # Repeat the identical registered source examples to compare exact outputs
 # and measured steady native throughput; never save formal measurements.
 records=(records*2)[:12]
 assert len(records)==12
 records=[{**r,'key':f"{i}/{r['key']}"} for i,r in enumerate(records)]
 entities=[c['name'] for c in json.loads(Path(protocol['vocabulary_path']).read_text())['categories']]
 ctx=mp.get_context('spawn');cases=[];reference=None
 for count in (1,2,4):
  out=args.out/f'workers_{count}';out.mkdir(exist_ok=True)
  barrier=ctx.Barrier(count+1)
  children=[ctx.Process(target=worker,args=(i,count,args.gpu,records,entities,barrier,str(out))) for i in range(count)]
  for p in children:p.start()
  print(json.dumps({'state':'WARMING','same_gpu_workers':count,'images':len(records)}),flush=True)
  barrier.wait(timeout=300)
  start=time.perf_counter()
  for p in children:p.join()
  wall=time.perf_counter()-start
  assert all(p.exitcode==0 for p in children),[p.exitcode for p in children]
  results=[json.loads((out/f'worker_{i}.json').read_text()) for i in range(count)]
  signatures={r['key']:r['signature'] for p in results for r in p['signatures']}
  assert len(signatures)==len(records)
  if reference is None:reference=signatures
  else:assert signatures==reference,'Concurrent singleton outputs differ from singleton reference'
  # Runtime load and warm-up excluded; longest worker measures the complete
  # shared-GPU processing interval, excluding process teardown.
  duration=max(r['seconds'] for r in results)
  case={'workers_per_gpu':count,'images':len(records),'seconds':duration,'parent_wall_seconds_with_teardown':wall,'images_per_second':len(records)/duration,'all_proposals_bit_exact':True,'peak_worker_memory_bytes':max(r['max_memory_allocated_bytes'] for r in results),'backbone_calls':sum(r['backbone_calls'] for r in results)}
  cases.append(case);print(json.dumps({'state':'MEASURED',**case}),flush=True)
 report={'status':'PASS','reference_only':True,'checkpoint_sha256':SHA,'gpu':args.gpu,'frames':records,'cases':cases,'full_vocabulary':1203,'native_singleton_fp32':True,'selected_workers_per_gpu':max(cases,key=lambda c:c['images_per_second'])['workers_per_gpu'],'scope':'12 repeated original validation frames spanning TAO source domains; actual parent/worker proposal parity, not formal metric evidence','ready_for_u3':False}
 (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'state':'COMPLETE','cases':cases,'selected_workers_per_gpu':report['selected_workers_per_gpu']}),flush=True)

if __name__=='__main__':main()
