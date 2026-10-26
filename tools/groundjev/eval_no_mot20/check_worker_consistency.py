"""One registered real input through four independent TP=1 G2 workers.

This is an explicitly duplicated engineering input required by the concurrency
contract. It is never counted as additional evaluation coverage.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tools.groundjev.eval_no_mot20.common import RUN,load,save,candidate_payload


def main():
    from PIL import Image
    from locatemot.groundjev.perception import GroundAnythingAdapter
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    config=load(args.config);registry=load(RUN/'jobs.json')
    entry=min((j for j in registry['jobs'] if j['dataset']=='sportsmot'),key=lambda j:(j['frames'],j['id']))
    job=Path(entry['path']);manifest=load(job/'manifest.json')
    # Cached single-device baseline was produced before this concurrent test.
    from tools.groundjev.eval_no_mot20.cache_store import CacheStore
    store=CacheStore(job);baseline=store.frame(manifest['frame_numbers'][0]);store.close()
    if baseline is None:raise RuntimeError('Wait for the registered native baseline input')
    def request(worker):
        frontend=GroundAnythingAdapter(f"http://127.0.0.1:{worker['port']}/v1",parser_policy='official_scoring',
            decoder_policy='tracking_dense',allow_vocabulary_fallback=False)
        with Image.open(manifest['images'][0]) as image:
            candidates,metadata=frontend(image.convert('RGB'),'mot',['person'],image_path=manifest['images'][0])
        return {'worker':worker['name'],'gpu':worker['gpu'],'candidates':candidates,'metadata':metadata}
    began=time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(request,config['workers']))
    comparisons=[]
    for value in results:
        row={'frame':baseline['frame'],'image_path':baseline['image_path'],'candidates':value['candidates'],
             'perception':{'model_response_rejected':False}}
        comparisons.append({'worker':value['worker'],'equal':candidate_payload(row)==candidate_payload(baseline)})
    receipt={'status':'FOUR_WORKER_NATIVE_CONSISTENCY_PASSED' if all(v['equal'] for v in comparisons) else
                      'FOUR_WORKER_NATIVE_CONSISTENCY_FAILED',
             'baseline_job':str(job),'frame':baseline['frame'],'comparisons':comparisons,'results':results,
             'wall_seconds':time.perf_counter()-began,'functional_only':True,'new_evaluation_frames':0}
    save(RUN/'engineering_checks/worker_consistency.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='results'}),flush=True)
    if not all(v['equal'] for v in comparisons):raise SystemExit(1)


if __name__=='__main__':main()
