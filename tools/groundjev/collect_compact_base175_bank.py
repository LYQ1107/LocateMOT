"""One owned ROI-only service; resumable selected-Base FP32 bank, no images copied."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    import numpy as np
    from PIL import Image
    from locatemot.groundjev.feature_client import NativeFeatureClient
    from tools.groundjev.service_process_group import stop_owned_service
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    plan=json.loads(args.plan.read_text());out=ROOT/plan['output']
    proof=json.loads((out/'object_validation.json').read_text())
    assert proof['status']=='EXISTING_BASE175_TEACHER_OBJECTS_VALIDATED_NATIVE_FEATURES_NOT_YET_COLLECTED'
    assert plan['combined_registered_feature_upper_bytes']<plan['global_temporary_feature_budget_bytes']
    assert plan['physical_gpu']==4
    objects=[json.loads(line) for line in (out/'objects.jsonl').open()]
    assert [r['feature_row'] for r in objects]==list(range(len(objects)))
    by_image=defaultdict(list)
    for row in objects:by_image[row['image_path']].append(row)
    progress=out/'completed_images.jsonl';partial=out/'regions.partial.npy';final=out/'regions.npy'
    state_path=out/'owner_state.json'
    if args.resume:
        previous=json.loads(state_path.read_text());pid=previous['owner_pid']
        stat=Path('/proc')/str(pid)/'stat'
        if stat.exists():
            fields=stat.read_text().rsplit(')',1)[1].split()
            assert fields[0]=='Z' or fields[19]!=previous['owner_starttick'], 'Original collection owner is still running'
        assert not final.exists()
        bank=np.lib.format.open_memmap(partial,mode='r+')
        assert bank.shape==(len(objects),2560) and bank.dtype==np.float32
        completed={}
        if progress.exists():
            with progress.open('r+') as file:
                while True:
                    position=file.tell();line=file.readline()
                    if not line:break
                    if not line.endswith('\n'):
                        file.seek(position);file.truncate();break
                    row=json.loads(line);completed[row['image_path']]=row
    else:
        assert not state_path.exists() and not partial.exists() and not final.exists()
        completed={};bank=np.lib.format.open_memmap(partial,mode='w+',dtype=np.float32,shape=(len(objects),2560))
    assert set(completed)<=set(by_image)
    state=dict(status='WAITING_SHARED_GPU4_MEMORY_FOR_SELECTED_BASE175_ROI_ONLY',owner_pid=os.getpid(),
        owner_starttick=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        physical_gpu=4,objects=len(objects),images=len(by_image),completed_images=len(completed),
        existing_GPU_processes_stopped=False,new_DLM_or_optimizer_calls=0,full_goal_complete=False)
    def save():
        tmp=out/'owner_state.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(state_path)
        print(json.dumps(state),flush=True)
    for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
        os.environ.pop(key,None)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    server=None;log=None;save()
    try:
        while True:
            free=int(subprocess.check_output(['nvidia-smi','-i','4','--query-gpu=memory.free',
                '--format=csv,noheader,nounits'],text=True).strip())
            if free>=plan['launch_min_free_MiB']:break
            time.sleep(20)
        attempt=1
        while (out/f'feature_service_{attempt}.json').exists():attempt+=1
        service_receipt=out/f'feature_service_{attempt}.json'
        log=(out/f'feature_service_{attempt}.log').open('w')
        server=subprocess.Popen(['python3','tools/groundjev/run_with_resources.py','--gpus','4','--',
            str(ROOT/'external/GroundAnything/.venv-serve/bin/python'),'tools/groundjev/serve_native_features.py',
            '--port',str(plan['feature_port']),'--receipt',str(service_receipt)],cwd=ROOT,env=env,
            stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        state.update(status='WAITING_OWNED_SELECTED_BASE175_NATIVE_VISION_SERVICE',service_group=server.pid,
            launch_free_MiB=free,service_receipt=str(service_receipt));save()
        while not service_receipt.exists():
            assert server.poll() is None, 'Owned ROI service exited before readiness; inspect its retained log'
            time.sleep(1)
        receipt=json.loads(service_receipt.read_text())
        assert receipt['region_dimension']==2560 and not receipt['language_decoder_loaded']
        assert receipt['native_parameters']>10000000
        client=NativeFeatureClient(receipt['url'])
        state.update(status='COLLECTING_ONLY_REGISTERED_SELECTED_BASE175_NATIVE_REGIONS',native_service=receipt);save()
        with progress.open('a') as done:
            for path,rows in sorted(by_image.items()):
                if path in completed:continue
                assert server.poll() is None, 'Owned native ROI service ended during collection'
                with Image.open(path) as source:
                    image=source.convert('RGB')
                assert image.size==tuple(rows[0]['image_size'])
                candidates=[{'box':row['box']} for row in rows]
                enriched,metadata=client(image,candidates,image_path=path)
                raw=[candidate['native_region_feature'] for candidate in enriched]
                vectors=np.asarray(raw,dtype=np.float32)
                assert vectors.shape==(len(rows),2560) and vectors.tolist()==raw and np.isfinite(vectors).all()
                assert metadata['feature_source']=='official_DLM_native_MoonViT_and_trained_projector'
                for row,vector in zip(rows,vectors):bank[row['feature_row']]=vector
                bank.flush()
                record=dict(image_path=path,annotation_ids=[row['annotation_id'] for row in rows],
                    feature_rows=[row['feature_row'] for row in rows],native_metadata=metadata)
                done.write(json.dumps(record)+'\n');done.flush();completed[path]=record
                state['completed_images']=len(completed)
                if len(completed)%50==0 or len(completed)==len(by_image):save()
        assert set(completed)==set(by_image) and np.isfinite(bank).all()
        bank.flush();del bank;partial.replace(final)
        stop_owned_service(server);server=None;log.close();log=None
        result=dict(status='SELECTED_BASE175_NATIVE_FP32_TEACHER_BANK_COMPLETE',
            selected_draws=proof['sampled_root_and_leaf_questions_each'],unique_objects=len(objects),
            unique_images=len(by_image),supervised_Base_classes=175,full_input_vocabulary=1203,
            shape=[len(objects),2560],dtype='float32',bank_path=str(final),bank_bytes=final.stat().st_size,
            all_registered_images_and_observations_completed=True,all_native_values_preserved_without_quantization=True,
            feature_provider=receipt,GT_scope='registered train-Base teacher boxes; class labels never sent to native ROI service',
            own_service_group_stopped=True,old_training_or_evaluation_processes_stopped=False,
            new_DLM_or_optimizer_calls=0,full_goal_complete=False)
        (out/'feature_validation.json').write_text(json.dumps(result,indent=2)+'\n')
        state.update(status=result['status'],result=result);save()
        assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<plan['new_component_cap_bytes']
    except Exception as error:
        state.update(status='SELECTED_BASE175_NATIVE_ROI_BANK_FAILED',error=repr(error));save();raise
    finally:
        if server is not None:stop_owned_service(server)
        if log is not None:log.close()


if __name__=='__main__':main()
