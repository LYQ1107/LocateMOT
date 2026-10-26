"""Bounded real-GPU frozen-backbone training and matched DEV evaluation."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import yaml

from locatemot.groundjev.dense_head.spatial_backbone import SpatialBackbone,BoundedFeatureStore
from locatemot.groundjev.dense_head.object_decoder import ObjectDecoder
from locatemot.groundjev.dense_head.hungarian_criterion import HungarianCriterion
from locatemot.groundjev.dense_head.model import targets_on_grid,valid_ratio
from tools.groundjev.dense_head.prepare_train_data import write_json,digest
from tools.groundjev.dense_head.evaluate_dense_head import load_rows,predict,evaluate_predictions,select_threshold


def save_checkpoint(path,head,optimizer,scheduler,step,rng,config,extra):
    temporary=path.with_suffix('.tmp')
    torch.save({'head':head.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'step':step,'sampler_rng':rng.getstate(),'torch_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state(),'config':config,**extra},temporary)
    temporary.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--resume',type=Path)
    a=p.parse_args();c=yaml.safe_load(a.config.read_text());torch.set_num_threads(4)
    assert torch.cuda.is_available()
    arm=c['arm'];run=Path(c['output_root']);run.mkdir(parents=True,exist_ok=True)
    if (run/'receipt.json').exists():
        raise RuntimeError('This pilot is complete; use a new output_root for another experiment.')
    if (run/'last.pt').exists() and not a.resume:
        raise RuntimeError('Preserved checkpoint exists; explicitly resume it or use a new output_root.')
    audit=json.loads((Path(c['root'])/'audits'/arm/'receipt.json').read_text())
    assert audit['status']=='REAL_GPU_SPATIAL_SMOKE_PASSED'
    assert Path(c['old_pause_receipt']).exists()
    assert json.loads(Path(c['old_pause_receipt']).read_text())['status']=='PAUSED_BY_USER_FOR_ARCHITECTURE_PIVOT'
    registration=json.loads((Path(c['root'])/'GPU_REGISTRATION.json').read_text())
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(registration['arms'][arm]['gpu'])
    train=load_rows(c['train_index']);dev=load_rows(c['dev_index']);full_dev=load_rows(c['full_dev_index'])
    protocol=json.loads((Path(c['root'])/'data/protocol.json').read_text())
    assert c['queries']==protocol['query_selection']['queries']
    backbone=SpatialBackbone(c['model'],max_tokens=c['max_tokens'])
    torch.manual_seed(c['seed']);np.random.seed(c['seed']);rng=random.Random(c['seed'])
    head=ObjectDecoder(hidden_dim=c['hidden_dim'],queries=c['queries'],layers=c['layers'],
                       spatial_attention_sigma=c.get('spatial_attention_sigma')).cuda()
    initial_hash=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in head.parameters())).hexdigest()
    optimizer=torch.optim.AdamW(head.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:min(1.,(s+1)/50)*(.1 if s>=800 else 1.))
    criterion=HungarianCriterion();store=BoundedFeatureStore(backbone,run/'features',c['feature_cache_bytes'])
    grouped=defaultdict(lambda:defaultdict(list))
    for row in train:grouped[row['dataset']][row['sequence']].append(row)
    domains=sorted(grouped);sequence_lists={d:sorted(grouped[d]) for d in domains}
    first=1;best=-1.;history=[];sample_hash=hashlib.sha256();used=set();exposures=defaultdict(int)
    if a.resume:
        saved=torch.load(a.resume,map_location='cpu',weights_only=False)
        assert saved['config']==c
        head.load_state_dict(saved['head']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        first=saved['step']+1;rng.setstate(saved['sampler_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
        best=saved.get('best_macro_AP50',-1.)
        # Reconstruct the prefix trace from the exact deterministic sampler.
        # This preserves truthful unique-image/exposure/order receipts on resume.
        prefix=random.Random(c['seed'])
        for prior in range(1,first):
            for batch in range(c['batch_images']):
                domain=domains[((prior-1)*c['batch_images']+batch)%len(domains)]
                sequence=prefix.choice(sequence_lists[domain]);row=prefix.choice(grouped[domain][sequence])
                key=row['image_path'];sample_hash.update((key+'\n').encode());used.add(key);exposures[domain]+=1
        assert prefix.getstate()==rng.getstate()
    started=time.perf_counter();training_seconds=0.;stop_requested=[]
    signal.signal(signal.SIGTERM,lambda *_:stop_requested.append(True))
    versions={n:p._version for n,p in backbone.visual.named_parameters()}
    write_json(run/'registration.json',{'status':'REAL_GPU_PILOT_RUNNING','arm':arm,'pid':os.getpid(),
        'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],'GPU_UUID':registration['arms'][arm]['uuid'],
        'config':c,'model_SHA':audit['model_SHA256'],'initial_head_SHA256':initial_hash,
        'train_index_SHA256':digest(c['train_index']),'DEV_index_SHA256':digest(c['dev_index']),
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'frozen_parameters':backbone.parameter_count,'trainable_parameters':sum(p.numel() for p in head.parameters()),
        'feature_cache_budget_bytes':c['feature_cache_bytes'],'start_unix':time.time()})
    print(json.dumps({'event':'PILOT_STARTED','arm':arm,'steps':c['steps'],'effective_batch':c['batch_images'],'initial_head_SHA':initial_hash}),flush=True)
    for step in range(first,c['steps']+1):
        if stop_requested or time.perf_counter()-started>c['max_wall_seconds']:
            save_checkpoint(run/'last.pt',head,optimizer,scheduler,step-1,rng,c,{'best_macro_AP50':best})
            write_json(run/'budget_stop.json',{'status':'BUDGET_STOP_RESUMABLE','step':step-1,'elapsed':time.perf_counter()-started})
            return
        began=time.perf_counter();head.train();optimizer.zero_grad(set_to_none=True);losses=defaultdict(float)
        for batch in range(c['batch_images']):
            # Alternate domains, then sample physical video uniformly within
            # domain: long videos cannot monopolize the pilot.
            domain=domains[((step-1)*c['batch_images']+batch)%len(domains)]
            sequence=rng.choice(sequence_lists[domain]);row=rng.choice(grouped[domain][sequence])
            key=row['image_path'];sample_hash.update((key+'\n').encode());used.add(key);exposures[domain]+=1
            spatial,meta=store.get(key)
            target=targets_on_grid(row['boxes'],meta,'cuda:0');ignored=targets_on_grid(row['ignored'],meta,'cuda:0')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                prediction=head(spatial,valid_ratio(meta));parts=criterion(prediction,target,ignored,row['exhaustive'])
                loss=parts['loss']/c['batch_images']
            if not torch.isfinite(loss):raise RuntimeError(f'nonfinite loss: step{step} image{key}')
            loss.backward()
            for name,value in parts.items():losses[name]+=float(value.detach())/c['batch_images']
        grad=torch.nn.utils.clip_grad_norm_(head.parameters(),.1)
        assert torch.isfinite(grad)
        optimizer.step();scheduler.step();torch.cuda.synchronize()
        elapsed=time.perf_counter()-began;training_seconds+=elapsed
        record={'step':step,**losses,'lr':optimizer.param_groups[0]['lr'],'gradient_norm':float(grad),
                'step_seconds':elapsed,'train_seconds':training_seconds,'cache_hits':store.hits,'cache_misses':store.misses,
                'peak_GPU_MiB':torch.cuda.max_memory_allocated()/2**20}
        with (run/'loss.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        if step==1 or step%50==0:
            print(json.dumps({'event':'PILOT_PROGRESS','arm':arm,**record}),flush=True)
        if step%c['eval_every']==0 or step==c['steps']:
            predictions,timing=predict(head,store,dev)
            metrics=evaluate_predictions(dev,predictions,.5)
            macro=float(np.mean([v['AP50'] for v in metrics.values()]))
            evaluation={'step':step,'metrics_at_0.5':metrics,'timing':timing,'selection_macro_AP50':macro}
            history.append(evaluation);write_json(run/f'dev_step{step:04d}.json',evaluation)
            if macro>best:
                best=macro;save_checkpoint(run/'best_dev.pt',head,optimizer,scheduler,step,rng,c,{'best_macro_AP50':best})
            save_checkpoint(run/'last.pt',head,optimizer,scheduler,step,rng,c,{'best_macro_AP50':best})
            print(json.dumps({'event':'PILOT_DEV','arm':arm,'step':step,'macro_AP50':macro,
                              'Recall50_at_half':{k:v['Recall50'] for k,v in metrics.items()}}),flush=True)
    assert all(p.grad is None and not p.requires_grad and p._version==versions[n] for n,p in backbone.visual.named_parameters())
    saved=torch.load(run/'best_dev.pt',map_location='cpu',weights_only=False);head.load_state_dict(saved['head'])
    predictions,_=predict(head,store,dev)
    threshold,threshold_audit=select_threshold(dev,predictions)
    write_json(run/'frozen_selection.json',{'best_step':saved['step'],'threshold':threshold,
        'threshold_candidates':threshold_audit,'selection_scope':'TRAIN-internal DEV sampled frames only',
        'architecture_and_training_fixed':True,'time_unix':time.time()})
    # Full DEV physical videos, not merely the sampled development frames.
    predictions,timing=predict(head,store,full_dev,use_cache=False)
    metrics=evaluate_predictions(full_dev,predictions,threshold,per_video=True)
    fixed=evaluate_predictions(full_dev,predictions,.5,compute_map=False)
    write_json(run/'dev_full_predictions.json',predictions)
    write_json(run/'dev_full_results.json',{'status':'COMPLETE_FULL_TRAIN_INTERNAL_DEV','metrics':metrics,
        'metrics_fixed_threshold_0.5':fixed,'timing':timing,'threshold':threshold,'best_step':saved['step'],
        'DEV_not_independent_of_G2_pretraining':True})
    # Hyperparameters, best step and threshold were frozen on internal DEV.
    # A0 native predictions are reused; no autoregressive G2 inference is run.
    external=load_rows(c['external_index'])
    external_predictions,external_timing=predict(head,store,external,use_cache=False)
    write_json(run/'external_predictions.json',external_predictions)
    write_json(run/'external_results.json',{'status':'FROZEN_EXTERNAL_COMPARISON_COMPLETE',
        'metrics':evaluate_predictions(external,external_predictions,threshold,per_video=True),
        'metrics_fixed_threshold_0.5':evaluate_predictions(external,external_predictions,.5,compute_map=False),
        'timing':external_timing,'best_step':saved['step'],'threshold_from_internal_DEV':threshold,
        'index_SHA256':digest(c['external_index']),'test_time_parameter_tuning':False})
    elapsed=time.perf_counter()-started
    receipt={'status':'REAL_GPU_PILOT_AND_FULL_DEV_COMPLETE','arm':arm,'optimizer_steps':c['steps'],
        'selected_step':saved['step'],'selected_threshold':threshold,'elapsed_seconds':elapsed,
        'GPU_hours':elapsed/3600,'train_seconds':training_seconds,'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],
        'peak_allocated_GPU_MiB':torch.cuda.max_memory_allocated()/2**20,
        'peak_reserved_GPU_MiB':torch.cuda.max_memory_reserved()/2**20,
        'unique_optimizer_images':len(used),'sample_exposures':dict(exposures),'sampling_order_SHA256':sample_hash.hexdigest(),
        'initial_head_SHA256':initial_hash,'frozen_backbone_unchanged':True,'no_GT_in_features':True,
        'temporary_feature_bytes':store.size,'checkpoint_last':str(run/'last.pt'),'checkpoint_best':str(run/'best_dev.pt'),
        'checkpoint_last_SHA256':digest(run/'last.pt'),'checkpoint_best_SHA256':digest(run/'best_dev.pt'),
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'best_DEV_macro_AP50':best,'final_results':str(run/'dev_full_results.json'),
        'scientific_improvement_claim':'requires matched A0/A1/A2 final comparison'}
    write_json(run/'receipt.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
