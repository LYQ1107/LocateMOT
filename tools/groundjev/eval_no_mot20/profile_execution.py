"""Same-input eager/fast-eager/eager diagnostic; never contributes eval frames."""
from __future__ import annotations

import argparse
import fcntl
import gzip
import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tools.groundjev.eval_no_mot20.common import load,save,fingerprint,job_manifest
from tools.groundjev.eval_no_mot20.dispatch_4gpu import start_service,stop_service


def select_samples(run):
    registry=load(run/'jobs.json')['jobs'];samples=[]
    for dataset in ('sportsmot','animaltrack','refer_kitti_v2','tao_amodal'):
        complete=[j for j in registry if j['dataset']==dataset and (Path(j['path'])/'cache_receipt.json').exists()]
        if not complete:raise ValueError(f'No completed native scope for {dataset}')
        entry=min(complete,key=lambda j:(j['frames'],j['id']));job=Path(entry['path'])
        db=sqlite3.connect(f'file:{job}/detector.sqlite?mode=ro',uri=True)
        for frame,chunk,blob in db.execute('SELECT frame,chunk,data FROM requests ORDER BY frame,chunk LIMIT 4'):
            original=json.loads(gzip.decompress(blob))
            samples.append({'dataset':dataset,'manifest':job_manifest(job),'frame':frame,'chunk':chunk,
                            'image_path':original['image_path'],'vocabulary':original['vocabulary'],
                            'original_raw_output':original['metadata']['result']['raw_output']})
        db.close()
    return samples


def profile(config,worker,mode='fast_eager'):
    from PIL import Image
    from locatemot.groundjev.perception import GroundAnythingAdapter,GroundingProtocolError
    from tools.groundjev.eval_no_mot20.export_g2_candidates import bounded_request
    run=Path(config['output_root']);directory=run/'engineering_checks'/mode
    directory.mkdir(parents=True,exist_ok=True);receipt=directory/'receipt.json'
    with (directory/'.probe.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return {'status':'PROFILE_ALREADY_RUNNING'}
        if receipt.exists():return load(receipt)
        samples=select_samples(run);phases={};began=time.perf_counter()
        # Reuse this reserved physical slot; never start a fifth GPU service.
        worker=dict(worker,port=worker['port']+100)
        try:
            candidate_phase='fast' if mode=='fast_eager' else 'graph'
            for name,candidate in (('baseline',False),(candidate_phase,True),('baseline_repeat',False)):
                target=directory/name;target.mkdir(exist_ok=True);process=handle=None;rows=[]
                try:
                    process,handle=start_service(config,dict(worker,
                        fast_eager_diagnostics=candidate and mode=='fast_eager',
                        triton_block_graph=candidate and mode=='triton_graph'),target)
                    frontend=GroundAnythingAdapter(f"http://127.0.0.1:{worker['port']}/v1",
                        parser_policy='official_scoring',decoder_policy='tracking_dense',
                        allow_vocabulary_fallback=False,request_timeout=config['request_timeout'])
                    with (target/'requests.jsonl').open('w') as log:
                        for index,sample in enumerate(samples):
                            with Image.open(sample['image_path']) as source:image=source.convert('RGB')
                            started=time.perf_counter();rejected=False
                            try:
                                candidates,metadata=bounded_request(frontend,image,sample['manifest'],
                                    sample['vocabulary'],sample['image_path'],config['request_timeout'])
                            except GroundingProtocolError as error:
                                candidates=[];metadata=error.metadata;rejected=True
                            elapsed=time.perf_counter()-started;raw=metadata['result']['raw_output']
                            row={'sample':index,'dataset':sample['dataset'],'frame':sample['frame'],
                                'chunk':sample['chunk'],'seconds':elapsed,'candidates':candidates,
                                'rejected':rejected,'raw_output':raw,'server_usage':metadata.get('response_usage'),
                                'original_raw_equal':raw==sample['original_raw_output'],
                                'signature':fingerprint({'raw':raw,'candidates':candidates,'rejected':rejected})}
                            rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
                            print(json.dumps({'event':'EXECUTION_PROFILE_REQUEST','phase':name,
                                              'sample':index,'seconds':elapsed}),flush=True)
                finally:stop_service(process,handle)
                phases[name]=rows
            equal=all(a['signature']==b['signature']==c['signature'] and a['original_raw_equal']
                for a,b,c in zip(phases['baseline'],phases[candidate_phase],phases['baseline_repeat']))
            times={name:sum(r['seconds'] for r in rows) for name,rows in phases.items()}
            result={'status':'NATIVE_OUTPUT_EQUIVALENCE_PASSED' if equal else 'NATIVE_OUTPUT_EQUIVALENCE_FAILED',
                'requests_per_phase':len(samples),'physical_gpu':worker['gpu'],'uuid':worker['uuid'],
                'phase_request_seconds':times,'execution_variant':mode,
                'candidate_speedup_vs_two_baseline_mean':
                    (times['baseline']+times['baseline_repeat'])/2/times[candidate_phase],
                'candidate_speedup_vs_warm_repeat':times['baseline_repeat']/times[candidate_phase],
                'per_dataset':{dataset:{phase:sum(r['seconds'] for r in rows if r['dataset']==dataset)
                    for phase,rows in phases.items()} for dataset in {s['dataset'] for s in samples}},
                'all_raw_text_candidates_rejection_identical':equal,'new_evaluation_frames':0,
                'weights_decoder_parser_seed_precision_changed':False,
                'isolated_change':('native per-forward diagnostic synchronize/print suppression' if mode=='fast_eager'
                                   else 'official Triton single-request B32 CUDA Graph; same BF16 decoding'),
                'total_wall_seconds':time.perf_counter()-began,
                'scope':'one paired execution probe under shared GPU load; not full pipeline FPS or a quality gain'}
            save(receipt,result);return result
        except BaseException as error:
            save(receipt,{'status':'EXECUTION_PROFILE_FAILED','error':repr(error),
                         'completed_phases':list(phases),'new_evaluation_frames':0})
            raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--worker',required=True)
    parser.add_argument('--mode',choices=['fast_eager','triton_graph'],default='fast_eager')
    args=parser.parse_args();config=load(args.config)
    worker=next(w for w in config['workers'] if w['name']==args.worker)
    print(json.dumps(profile(config,worker,args.mode)),flush=True)


if __name__=='__main__':main()
