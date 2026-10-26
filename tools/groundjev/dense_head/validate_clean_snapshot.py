"""Real GPU inference from an exported public Git snapshot, no training update."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import torch
from PIL import Image

from locatemot.groundjev.native_features import NativeGroundFeatures
from locatemot.groundjev.dense_head.spatial_backbone import SpatialBackbone
from locatemot.groundjev.dense_head.object_decoder import ObjectDecoder
from locatemot.groundjev.dense_head.hungarian_criterion import HungarianCriterion
from locatemot.groundjev.dense_head.model import valid_ratio,boxes_to_original,targets_on_grid
from tools.groundjev.dense_head.prepare_train_data import write_json,digest


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--tree-sha',required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    started=time.perf_counter();torch.set_num_threads(4)
    symbols=[NativeGroundFeatures,SpatialBackbone,ObjectDecoder,HungarianCriterion,boxes_to_original]
    imports={s.__name__:str(Path(inspect.getfile(s)).resolve()) for s in symbols}
    assert all(Path(v).is_relative_to(ROOT) for v in imports.values()),imports
    receipt=json.loads((a.run_root/'g2/receipt.json').read_text())
    assert receipt['status']=='REAL_GPU_PILOT_AND_FULL_DEV_COMPLETE'
    checkpoint=Path(receipt['checkpoint_best']);assert digest(checkpoint)==receipt['checkpoint_best_SHA256']
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);c=saved['config']
    b=SpatialBackbone(c['model']);head=ObjectDecoder(queries=c['queries'],
        spatial_attention_sigma=c.get('spatial_attention_sigma')).cuda()
    head.load_state_dict(saved['head'],strict=True);head.eval()
    restored=ObjectDecoder(queries=c['queries'],spatial_attention_sigma=c.get('spatial_attention_sigma')).cuda()
    restored.load_state_dict(saved['head'],strict=True);restored.eval()
    rows=[json.loads(line) for line in Path(c['dev_index']).open()]
    examples=[next(r for r in rows if r['dataset']==dataset) for dataset in ['sportsmot','animaltrack']]
    outputs=[]
    for row in examples:
        with Image.open(row['image_path']) as im:features,metadata=b.encode(im.convert('RGB'))
        with torch.no_grad():
            prediction=head(features,valid_ratio(metadata));same=restored(features,valid_ratio(metadata))
            boxes=boxes_to_original(prediction['boxes'][0],metadata)
        assert torch.equal(prediction['boxes'],same['boxes']) and torch.equal(prediction['logits'],same['logits'])
        assert torch.isfinite(boxes).all() and torch.isfinite(prediction['logits']).all()
        outputs.append({'dataset':row['dataset'],'image_path':row['image_path'],'image_size':metadata['image_size'],
            'feature_shape':list(features.shape),'feature_SHA256':tensor_sha(features),'raw_query_boxes':len(boxes),
            'above_frozen_DEV_threshold':int((prediction['logits'][0].sigmoid()>=receipt['selected_threshold']).sum()),
            'finite_original_pixel_coordinates':True,'native_feature_metadata':metadata})
    assert all(not p.requires_grad and p.grad is None for p in b.visual.parameters())
    # Actual detection-loss -> online projector gradient, on TRAIN only.
    # No optimizer is created and no parameter is updated; this is the future
    # adaptation interface audit, not G2 joint fine-tuning.
    train=json.loads(Path(c['train_index']).open().readline())
    versions={n:p._version for n,p in b.visual.named_parameters()}
    with Image.open(train['image_path']) as im:
        image=im.convert('RGB');_,meta=b.encode(image)
        online=b.differentiable_encode(image,train_projector=True)
    predicted=head(online,valid_ratio(meta))
    loss=HungarianCriterion()(predicted,targets_on_grid(train['boxes'],meta,'cuda:0'))['loss']
    assert torch.isfinite(loss);loss.backward()
    projector_grad=float(b.visual.projector.fc2.weight.grad.float().norm())
    assert projector_grad>0 and all(p.grad is None for p in b.visual.vision_tower.parameters())
    b.visual.zero_grad(set_to_none=True);b.visual.requires_grad_(False);head.zero_grad(set_to_none=True)
    assert all(p._version==versions[n] for n,p in b.visual.named_parameters())
    result={'status':'CLEAN_PUBLIC_GIT_SNAPSHOT_REAL_GPU_CHECK_PASSED','source_commit':a.source_commit,
        'tree_SHA':a.tree_sha,'snapshot_root':str(ROOT),'repository_module_paths':imports,
        'checkpoint':str(checkpoint),'checkpoint_SHA256':receipt['checkpoint_best_SHA256'],'head_step':saved['step'],
        'backbone_G2_step':18227,'backbone_G2_model_SHA256':'7c4a81349e333a304e32f42cf1999db289802d8c31eba415b85827038d14801f',
        'spatial_attention_sigma':c.get('spatial_attention_sigma'),'physical_GPU_environment':os.environ['CUDA_VISIBLE_DEVICES'],
        'CUDA_device_properties':str(torch.cuda.get_device_properties(0)),'frames':outputs,
        'checkpoint_restore_exact':True,'optimizer_updates':0,'GT_or_answer_tokens_input':False,
        'online_TRAIN_detection_loss':float(loss.detach()),'online_projector_fc2_gradient_norm':projector_grad,
        'encoder_gradients':None,'G2_parameter_versions_unchanged':True,
        'new_autoregressive_generation':False,'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
        'elapsed_seconds':time.perf_counter()-started,'GPU_hours':(time.perf_counter()-started)/3600,
        'performance_scope':'Two real images for checkout/loading/inference integrity, not a detection benchmark'}
    write_json(a.out,result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
