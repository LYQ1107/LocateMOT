"""Compare real epoch1 native/bulk-label predictions with the original singleton probe."""
import sys,json,hashlib,time
from pathlib import Path
ROOT=Path('/data2/user/LocateMOT');sys.path.insert(0,str(ROOT))
import torch
from locatemot.unified.runtime.grounding_inference import load_grounding_runtime,run_grounding_chunked
from locatemot.unified.runtime.u2_frozen_text_cache import frozen_text_cache
from locatemot.unified.runtime.u2_label_metadata import bulk_label_metadata
root=ROOT/'outputs/unified/u2/performance/same_gpu_epoch1'
prior=json.loads((root/'report.json').read_text());reference=json.loads((root/'workers_1/worker_0.json').read_text())
records=prior['frames'];protocol=json.loads((ROOT/'outputs/unified/u2/protocol/tao_portability.json').read_text())
entities=[c['name'] for c in json.loads(Path(protocol['vocabulary_path']).read_text())['categories']]
torch.set_num_threads(2);torch.cuda.set_device(1)
runtime=load_grounding_runtime(checkpoint=ROOT/'outputs/unified/u2/perception_adaptation/training/epoch_1.pth',device='cuda:1',max_per_img=900)
assert runtime.checkpoint_sha256==prior['checkpoint_sha256']
def run(record):return run_grounding_chunked(runtime,record['image'],entities).pred_instances
with frozen_text_cache(runtime):
 native=run(records[0]);native_arrays={n:getattr(native,n).detach().cpu().clone() for n in ('bboxes','scores','labels')};native_names=list(native.label_names)
 with bulk_label_metadata(runtime) as stats:
  warm=run(records[0])
  assert all(torch.equal(native_arrays[n],getattr(warm,n).cpu()) for n in native_arrays)
  assert warm.label_names==native_names
  torch.cuda.synchronize(runtime.device);start=time.perf_counter();signatures=[]
  for record in records:
   result=run(record);digest=hashlib.sha256()
   for name in ('bboxes','scores','labels'):
    value=getattr(result,name).detach().cpu().contiguous();digest.update(name.encode());digest.update(str(value.dtype).encode());digest.update(value.numpy().tobytes())
   signatures.append({'key':record['key'],'signature':digest.hexdigest()})
  torch.cuda.synchronize(runtime.device);elapsed=time.perf_counter()-start
assert signatures==reference['signatures'],'Bulk integer metadata changed native proposals'
assert 'predict' not in runtime.model.__dict__ and 'predict' not in runtime.model.bbox_head.__dict__
report={'status':'PASS','reference_only':True,'checkpoint_sha256':runtime.checkpoint_sha256,'images':len(records),'native_seconds':reference['seconds'],'bulk_metadata_seconds':elapsed,'observed_speedup':reference['seconds']/elapsed,'proposal_boxes_scores_labels':'BIT_EXACT_ALL_12_FRAMES','native_label_names':'EXACT_FIRST_FRAME_FULL_VOCABULARY','native_methods_restored':True,'integer_only_cpu_copy':True,'native_singleton_fp32':True,'full_vocabulary':1203,'stats':stats,'actual_peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(runtime.device),'performance_scope':'12 repeated original validation frames spanning TAO source domains; no formal benchmark metric','ready_for_u3':False}
output=ROOT/'outputs/unified/u2/eval/bulk_label_metadata_contract.json';output.write_text(json.dumps(report,indent=2)+'\n')
# The concurrency probe's default-device memory counter was on GPU0; do not
# publish those zero values as actual GPU1 memory measurements.
for case in prior['cases']:case.pop('peak_worker_memory_bytes',None)
prior['memory_measurement']='not collected on the target device in the initial concurrency probe';(root/'report.json').write_text(json.dumps(prior,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
