"""Real GPU image-only feature, geometry, gradient and restore audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open

from locatemot.groundjev.dense_head.spatial_backbone import SpatialBackbone
from locatemot.groundjev.dense_head.object_decoder import ObjectDecoder
from locatemot.groundjev.dense_head.hungarian_criterion import HungarianCriterion,xyxy_to_cxcywh
from locatemot.groundjev.dense_head.model import targets_on_grid,boxes_to_original,valid_ratio
from tools.groundjev.dense_head.prepare_train_data import RUN,write_json


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def state_sha(module):
    digest=hashlib.sha256()
    for name,t in module.state_dict().items():
        digest.update(name.encode());digest.update(tensor_sha(t).encode())
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['g0','g2'],required=True);p.add_argument('--out',type=Path,default=RUN);a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42)
    assert torch.cuda.is_available() and len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==1
    model_path=ROOT/'models'/('GroundAnything-DLM' if a.arm=='g0' else 'GroundAnything-DLM-G2-CLEAN-V2')
    expected='3cf09355cde8cf4877cdc76e0a72301d572c1d6263b2831f8a03368285bb65e2' if a.arm=='g0' else '7c4a81349e333a304e32f42cf1999db289802d8c31eba415b85827038d14801f'
    digest=hashlib.sha256()
    with (model_path/'model.safetensors').open('rb') as f:
        for chunk in iter(lambda:f.read(16<<20),b''):digest.update(chunk)
    assert digest.hexdigest()==expected
    backbone=SpatialBackbone(model_path)
    row=json.loads((a.out/'data/train.jsonl').open().readline());dev=json.loads((a.out/'data/dev.jsonl').open().readline())
    protocol=json.loads((a.out/'data/protocol.json').read_text());queries=protocol['query_selection']['queries']
    directory=a.out/'audits'/a.arm;directory.mkdir(parents=True,exist_ok=True)
    with Image.open(row['image_path']) as image:
        image=image.convert('RGB')
        # Intervention: two real GT files differ, but neither is passed to vision.
        write_json(directory/'GT_original.json',row['boxes']);write_json(directory/'GT_deleted.json',[])
        features,meta=backbone.encode(image);again,_=backbone.encode(image)
        assert tensor_sha(features)==tensor_sha(again)
        bf16=features.float()
        backbone.visual.float();fp32,_=backbone.encode(image);backbone.visual.bfloat16()
    numerical={'relative_L2':float((bf16-fp32).norm()/fp32.norm()),
               'cosine':float(F.cosine_similarity(bf16.flatten(),fp32.flatten(),dim=0)),
               'max_abs':float((bf16-fp32).abs().max())}
    assert numerical['cosine']>.98, numerical
    backbone_hash=state_sha(backbone.visual)
    head=ObjectDecoder(queries=queries).cuda();criterion=HungarianCriterion();optimizer=torch.optim.AdamW(head.parameters(),lr=1e-4)
    targets=targets_on_grid(row['boxes'],meta,'cuda:0');ignored=targets_on_grid(row['ignored'],meta,'cuda:0')
    roundtrip=boxes_to_original(xyxy_to_cxcywh(targets),meta)
    assert torch.allclose(roundtrip,torch.tensor(row['boxes'],device='cuda:0'),atol=1e-3)
    losses=[]
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True);pred=head(features,valid_ratio(meta));loss=criterion(pred,targets,ignored)['loss'];assert torch.isfinite(loss);loss.backward();optimizer.step();losses.append(float(loss))
    grads={n:float(p.grad.norm()) for n,p in head.named_parameters() if p.grad is not None}
    assert len(grads)==len(list(head.parameters())) and all(v>0 for v in grads.values()),grads
    assert all(p.grad is None and not p.requires_grad for p in backbone.visual.parameters())
    assert state_sha(backbone.visual)==backbone_hash
    # With GT deleted, parallel queries still emit boxes in original coordinates.
    head.eval()
    with torch.no_grad():
        before=head(features,valid_ratio(meta));pixel=boxes_to_original(before['boxes'][0],meta)
    assert torch.isfinite(pixel).all() and pixel.shape==(queries,4)
    torch.save({'head':head.state_dict(),'optimizer':optimizer.state_dict()},directory/'smoke_checkpoint.pt')
    restored=ObjectDecoder(queries=queries).cuda();restored.load_state_dict(torch.load(directory/'smoke_checkpoint.pt',weights_only=False)['head']);restored.eval()
    with torch.no_grad():after=restored(features,valid_ratio(meta))
    assert torch.equal(before['boxes'],after['boxes']) and torch.equal(before['logits'],after['logits'])
    with Image.open(dev['image_path']) as im:df,dm=backbone.encode(im.convert('RGB'))
    with torch.no_grad():dp=head(df,valid_ratio(dm))
    # Future online projector path has a real gradient; this is not pilot training.
    with Image.open(row['image_path']) as im:differentiable=backbone.differentiable_encode(im.convert('RGB'),train_projector=True)
    differentiable.float().square().mean().backward()
    projector_gradient=float(backbone.visual.projector.fc2.weight.grad.norm())
    assert projector_gradient>0
    backbone.visual.zero_grad(set_to_none=True);backbone.visual.requires_grad_(False)
    write_json(directory/'receipt.json',{'status':'REAL_GPU_SPATIAL_SMOKE_PASSED','arm':a.arm,'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],
        'model_path':str(model_path.resolve()),'model_SHA256':expected,'model_step':18227 if a.arm=='g2' else 0,
        'feature_shape':list(features.shape),'feature_sha_GT_original':tensor_sha(features),'feature_sha_GT_deleted':tensor_sha(again),
        'GT_intervention_files':['GT_original.json','GT_deleted.json'],'GT_or_text_forward_arguments':False,
        'metadata':meta,'BF16_FP32':numerical,'backbone_state_SHA_before_after':backbone_hash,
        'frozen_parameters':backbone.parameter_count,'trainable_parameters':sum(p.numel() for p in head.parameters()),
        'all_head_parameter_gradient_norms':grads,'losses':losses,'coordinate_roundtrip_max_abs':float((roundtrip-torch.tensor(row['boxes'],device='cuda:0')).abs().max()),
        'GT_deleted_prediction_shape':list(pixel.shape),'checkpoint_restore_exact':True,'independent_head_DEV_prediction_shape':list(dp['boxes'].shape),
        'future_online_projector_gradient_norm':projector_gradient,'future_Qwen_LoRA_path':'Not covered by vision-only path; requires answer-free causal language design',
        'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'scientific_improvement_claim':False})
    print(json.dumps({'status':'REAL_GPU_SPATIAL_SMOKE_PASSED','arm':a.arm,'shape':list(features.shape),'BF16_FP32':numerical,'losses':losses,'receipt':str(directory/'receipt.json')}),flush=True)


if __name__=='__main__':main()
