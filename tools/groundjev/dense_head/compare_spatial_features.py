"""Verify the actual G0/G2 vision weights and the projector-only feature change."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import torch
from safetensors import safe_open
from PIL import Image

from locatemot.groundjev.dense_head.spatial_backbone import SpatialBackbone
from tools.groundjev.dense_head.prepare_train_data import RUN,write_json


def main():
    torch.set_num_threads(4)
    paths=[ROOT/'models'/name/'model.safetensors' for name in ['GroundAnything-DLM','GroundAnything-DLM-G2-CLEAN-V2']]
    differences=[];projector={}
    with safe_open(paths[0],framework='pt') as g0,safe_open(paths[1],framework='pt') as g2:
        for key in g0.keys():
            if not key.startswith('base_model.model.visual.'):continue
            a,b=g0.get_tensor(key),g2.get_tensor(key)
            if not torch.equal(a,b):
                differences.append({'key':key,'relative_L2':float((a.float()-b.float()).norm()/a.float().norm().clamp_min(1e-12))})
            if key.startswith('base_model.model.visual.projector.'):
                projector[key.split('base_model.model.visual.projector.')[1]]=a
    assert len(differences)==5 and all('projector.' in d['key'] for d in differences)
    backbone=SpatialBackbone(ROOT/'models/GroundAnything-DLM-G2-CLEAN-V2')
    row=json.loads((RUN/'data/train.jsonl').open().readline())
    with Image.open(row['image_path']) as im:
        im=im.convert('RGB');g2,m=backbone.encode(im)
        backbone.visual.projector.load_state_dict(projector)
        g0,_=backbone.encode(im)
    write_json(RUN/'audits/spatial_comparison.json',{'status':'REAL_GPU_G0_G2_FEATURE_COMPARISON',
        'unchanged_vision_encoder':True,'changed_visual_weights':differences,
        'feature_shape':list(g2.shape),'feature_relative_L2':float((g2.float()-g0.float()).norm()/g0.float().norm()),
        'feature_cosine':float(torch.nn.functional.cosine_similarity(g2.flatten().float(),g0.flatten().float(),dim=0)),
        'G0_control_method':'exact same encoder weights; replace all native projector tensors with G0; identical image processor',
        'Qwen_LoRA_effect_in_this_feature_path':False,'metadata':m})
    print((RUN/'audits/spatial_comparison.json').read_text())


if __name__=='__main__':main()
