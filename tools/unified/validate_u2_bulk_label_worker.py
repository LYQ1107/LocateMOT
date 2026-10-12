"""Real adapted TAO/C-TAO smoke worker parity for bulk native label metadata."""
from __future__ import annotations
import json,subprocess,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
 checkpoint=ROOT/'outputs/unified/u2/perception_adaptation/training/epoch_1.pth'
 sha=json.loads((ROOT/'outputs/unified/u2/perception_adaptation/eval_epoch_1/checkpoint_contract.json').read_text())['foundation_sha256']
 with tempfile.TemporaryDirectory(prefix='u2_bulk_label_worker_') as temp:
  base=Path(temp);cases=[]
  for enabled in (False,True):
   out=base/('bulk' if enabled else 'native')
   command=[sys.executable,'-u',str(ROOT/'tools/unified/u2_tao_portability.py'),'--smoke','--worker-index','0','--worker-count','1','--gpu','1','--out',str(out),'--measurements-dir',str(base/'measurements'),'--extraction-pid','625610','--checkpoint',str(checkpoint),'--expected-foundation-sha256',sha,'--reuse-frozen-text']
   if enabled:command.append('--bulk-label-metadata')
   start=time.perf_counter();subprocess.run(command,cwd=ROOT,check=True)
   rows=[json.loads(line) for line in (out/'worker_0.jsonl').read_text().splitlines()]
   result=json.loads((out/'worker_0.result.json').read_text())
   assert result['state']=='SMOKE_COMPLETE' and result['metadata']['smoke']
   assert result['metadata']['foundation_sha256']==sha and all(r['backbone_calls']==1 for r in rows)
   cases.append({'rows':rows,'metadata':result['metadata'],'seconds':time.perf_counter()-start,'bulk_stats':result.get('bulk_label_metadata')})
  assert cases[0]['rows']==cases[1]['rows'] and cases[0]['metadata']==cases[1]['metadata']
  assert len(cases[0]['rows'])==2 and cases[1]['bulk_stats']['head_calls']==76
 report={'status':'PASS','reference_only':True,'checkpoint_sha256':sha,'rows_and_gt_hit_counts_exact':True,'resume_metadata_identical':True,'backbone_once_per_image':True,'frames':2,'original_two_source_scope':cases[0]['rows'],'native_elapsed_seconds':cases[0]['seconds'],'bulk_elapsed_seconds':cases[1]['seconds'],'bulk_stats':cases[1]['bulk_stats'],'full_scope_evaluation_complete':False,'ready_for_u3':False}
 (ROOT/'outputs/unified/u2/eval/bulk_label_worker_contract.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
