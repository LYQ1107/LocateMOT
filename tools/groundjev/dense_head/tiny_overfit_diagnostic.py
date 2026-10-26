"""Bounded TRAIN-only capacity/geometry diagnostic, separate from matched pilot."""
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import torch
from PIL import Image

from locatemot.groundjev.dense_head.spatial_backbone import SpatialBackbone
from locatemot.groundjev.dense_head.object_decoder import ObjectDecoder
from locatemot.groundjev.dense_head.model import targets_on_grid,valid_ratio,boxes_to_original
from locatemot.groundjev.dense_head.hungarian_criterion import HungarianCriterion
from tools.groundjev.dense_head.prepare_train_data import RUN,write_json
from tools.groundjev.dense_head.evaluate_dense_head import evaluate_predictions


def main():
    started=time.perf_counter();torch.set_num_threads(2);torch.manual_seed(42)
    registration=json.loads((RUN/'DIAGNOSTIC_GPU_REGISTRATION.json').read_text())
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(registration['gpu'])
    train=[json.loads(x) for x in (RUN/'data/train.jsonl').open()]
    rows=[next(r for r in train if r['dataset']==d) for d in ['sportsmot','animaltrack']]
    backbone=SpatialBackbone(ROOT/'models/GroundAnything-DLM-G2-CLEAN-V2')
    inputs=[]
    for row in rows:
        with Image.open(row['image_path']) as im:f,m=backbone.encode(im.convert('RGB'))
        inputs.append((f,m,targets_on_grid(row['boxes'],m,'cuda:0')))
    head=ObjectDecoder(queries=100).cuda();optimizer=torch.optim.AdamW(head.parameters(),lr=5e-4,weight_decay=1e-4)
    criterion=HungarianCriterion();losses=[];steps=300
    for step in range(1,steps+1):
        if time.perf_counter()-started>180:break
        optimizer.zero_grad(set_to_none=True)
        values=[]
        for f,m,t in inputs:
            pred=head(f,valid_ratio(m));loss=criterion(pred,t)['loss']/len(inputs)
            assert torch.isfinite(loss);loss.backward();values.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step();losses.append(sum(values))
    head.eval();predictions=[]
    with torch.no_grad():
        for f,m,t in inputs:
            p=head(f,valid_ratio(m));predictions.append({'boxes':boxes_to_original(p['boxes'][0],m).cpu().tolist(),
                                                       'scores':p['logits'][0].sigmoid().cpu().tolist()})
    metrics=evaluate_predictions(rows,predictions,.5,compute_map=True)
    elapsed=time.perf_counter()-started
    receipt={'status':'REAL_GPU_TRAIN_ONLY_TINY_OVERFIT_DIAGNOSTIC','optimizer_steps':len(losses),
        'maximum_optimizer_steps':steps,'maximum_wall_seconds':180,'GPU':os.environ['CUDA_VISIBLE_DEVICES'],
        'elapsed_seconds':elapsed,'GPU_hours':elapsed/3600,'losses':losses,'metrics_on_these_same_2_TRAIN_frames':metrics,
        'input_images':[r['image_path'] for r in rows],'not_in_main_pilot_optimizer':True,
        'no_DEV_or_external_selection':True,'mock_training':False,'backbone_frozen':True,
        'scientific_generalization_claim':False}
    assert all(p.grad is None and not p.requires_grad for p in backbone.visual.parameters())
    write_json(RUN/'audits/tiny_overfit_diagnostic.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='losses'},indent=2),flush=True)


if __name__=='__main__':main()
