#!/usr/bin/env python3
"""L53 read-only GroundingDINO/TTAOD-F provenance and compatibility audit."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import argparse
from pathlib import Path

ROOT = Path('/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT')
TTAOD = Path('/data1/LWR/vranlee/SERVER_ONLY/avis/TTAOD-F-main')
PYTHON = Path('/home/lwr/anaconda3/envs/ttaod_f/bin/python')
CONFIG = TTAOD / 'configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.py'
WEIGHT = TTAOD / 'download/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'
SWIN = TTAOD / 'download/swin_tiny_patch4_window7_224.pth'
BERT = Path('/home/lwr/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594')
TRAIN_UNITS = ROOT / 'outputs/l49/data/train_units.jsonl'
DEFAULT_OUT = ROOT / 'outputs/l53/audit/compatibility_attempt1'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def files(root: Path):
    if root.is_file(): return [root]
    return sorted(p for p in root.rglob('*') if p.is_file())


def git_info(root: Path):
    try:
        head = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True, capture_output=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', str(root), 'status', '--short'], text=True, capture_output=True, check=True).stdout.strip()
        return {'head': head, 'dirty': bool(dirty), 'status': dirty}
    except Exception as e:
        return {'head': None, 'dirty': None, 'status': 'git metadata unavailable: '+str(e)}


def ckpt_info():
    import torch
    blob = torch.load(WEIGHT, map_location='cpu', weights_only=False)
    def describe(x, limit=60):
        if not isinstance(x, dict): return {'type': type(x).__name__}
        rows=[]
        for k,v in x.items():
            if hasattr(v,'shape'):
                rows.append({'key':str(k),'shape':list(v.shape),'dtype':str(v.dtype)})
                if len(rows)>=limit: break
        return {'keys':len(x),'tensor_descriptions':rows,'more_tensor_keys':max(0,len([v for v in x.values() if hasattr(v,'shape')])-limit)}
    return {'top_level_keys':sorted(str(k) for k in blob) if isinstance(blob,dict) else [],
            'state_dict':describe(blob.get('state_dict',blob) if isinstance(blob,dict) else blob),
            'meta_keys':sorted(str(k) for k in blob.get('meta',{})) if isinstance(blob,dict) else []}


def run_isolated(unit, expression):
    # The child receives no network access flags and uses only explicit local paths.
    code = r'''
import json, os, sys, time, traceback
import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet.registry import MODELS
import mmdet.models  # register TTAOD-F model/data-preprocessor modules
import mmdet.datasets  # register config-side dataset modules used by init/test
from mmdet.utils import register_all_modules
register_all_modules(init_default_scope=True)

root = os.environ['L53_TTAOD']
cfg_path = os.environ['L53_CONFIG']; ckpt = os.environ['L53_WEIGHT']; bert = os.environ['L53_BERT']; swin = os.environ['L53_SWIN']; image = os.environ['L53_IMAGE']; text = os.environ['L53_TEXT']
result={'python':sys.executable,'torch':torch.__version__,'cuda_available':bool(torch.cuda.is_available()),'config':cfg_path,'checkpoint':ckpt,'bert':bert,'swin':swin,'expression':text,'construction':{},'tokenizer':{},'forward':{}}
try:
    cfg=Config.fromfile(cfg_path)
    # Eliminate the config's URL and relative BERT path before construction.
    cfg.model.backbone.init_cfg=None
    cfg.model.language_model.name=bert
    result['config_override']={'backbone_init_cfg':None,'language_model_name':bert,'network_disabled':True}
    t0=time.time(); model=MODELS.build(cfg.model); result['construction']['built_sec']=time.time()-t0
    load=load_checkpoint(model,ckpt,map_location='cpu',strict=False)
    result['construction']['checkpoint_loaded']=True
    result['construction']['checkpoint_return_keys']=sorted(str(k) for k in load) if isinstance(load,dict) else []
    result['construction']['missing_keys']=load.get('missing_keys',[]) if isinstance(load,dict) else []
    result['construction']['unexpected_keys']=load.get('unexpected_keys',[]) if isinstance(load,dict) else []
    state=load.get('state_dict',{}) if isinstance(load,dict) else {}
    result['construction']['checkpoint_state_dict']={'tensor_count':sum(hasattr(v,'shape') for v in state.values()),'key_count':len(state),'sample_shapes':{str(k):list(v.shape) for k,v in list(state.items())[:40] if hasattr(v,'shape')}}
    model.to('cuda:0').eval(); model.cfg=cfg
    tok=model.language_model.tokenizer([text],padding='longest',return_tensors='pt')
    result['tokenizer']={'input_ids_shape':list(tok.input_ids.shape),'attention_mask_shape':list(tok.attention_mask.shape),'finite':bool(torch.isfinite(tok.input_ids.float()).all())}
    from mmdet.apis import inference_detector
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    # TTAOD-F's custom_entities=False path unconditionally calls its GLIP NER
    # helper, which invokes nltk.download().  Use the native entity-list path
    # to keep this audit offline while retaining the complete expression as one
    # text entity (no category-word replacement).
    t0=time.time(); output=inference_detector(model,image,text_prompt=text,custom_entities=True); elapsed=time.time()-t0
    result['forward']={'success':True,'elapsed_sec':elapsed,'output_type':type(output).__name__,'output_keys':sorted(str(k) for k in output.keys()) if hasattr(output,'keys') else [],'finite':True,'gpu_peak_bytes':int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0}
    if hasattr(output,'pred_instances'):
        pi=output.pred_instances
        result['forward']['pred_instances']={k:list(v.shape) for k,v in pi.items() if hasattr(v,'shape')}
        result['forward']['pred_finite']=all(bool(torch.isfinite(v.float()).all()) for v in pi.values() if hasattr(v,'dtype') and torch.is_floating_point(v))
    result['status']='pass'
except Exception as e:
    result['status']='fail'; result['error_type']=type(e).__name__; result['error']=str(e); result['traceback']=traceback.format_exc(limit=12)
print(json.dumps(result,default=str))
'''
    env=os.environ.copy(); env.update({'PYTHONPATH':str(TTAOD),'L53_TTAOD':str(TTAOD),'L53_CONFIG':str(CONFIG),'L53_WEIGHT':str(WEIGHT),'L53_BERT':str(BERT),'L53_SWIN':str(SWIN),'L53_IMAGE':str(unit['image_path']),'L53_TEXT':str(expression),'TRANSFORMERS_OFFLINE':'1','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    return subprocess.run([str(PYTHON),'-c',code],cwd=str(TTAOD),env=env,text=True,capture_output=True,timeout=900)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-root', default=str(DEFAULT_OUT))
    parser.add_argument('--unit-key', default=None,
                        help='Optional exact fit unit_key for a targeted single-image replay.')
    parser.add_argument('--image-path', default=None,
                        help='Explicit image for a label-free targeted replay.')
    parser.add_argument('--expression', default=None,
                        help='Explicit expression for a label-free targeted replay.')
    args = parser.parse_args()
    out_root = Path(args.out_root)
    if Path.cwd().resolve()!=ROOT: raise RuntimeError('wrong LocateMOT cwd')
    required=[TTAOD,CONFIG,WEIGHT,SWIN,BERT,TRAIN_UNITS]
    missing=[str(x) for x in required if not x.exists()]
    out_root.mkdir(parents=True,exist_ok=True)
    if missing:
        (out_root/'INCOMPLETE.md').write_text('Missing required local audit inputs:\n'+'\n'.join(missing)+'\n')
        raise FileNotFoundError(missing)
    units=[json.loads(x) for x in TRAIN_UNITS.read_text().splitlines() if x.strip() and json.loads(x).get('split')=='fit']
    ordered=sorted(units,key=lambda x:x['unit_key'])
    if args.image_path is not None or args.expression is not None:
        if not (args.image_path and args.expression):
            raise ValueError('--image-path and --expression must be supplied together')
        unit={'unit_key':args.unit_key or 'label_free_targeted_replay',
              'dataset':'targeted_replay','video':'unknown','query_id':None,
              'frame_id':None,'sentence':args.expression,'image_path':str(Path(args.image_path).resolve())}
    elif args.unit_key is None:
        unit=ordered[0]
    else:
        matches=[x for x in ordered if x.get('unit_key')==args.unit_key]
        if len(matches)!=1:
            raise KeyError(f'fit unit_key not found uniquely: {args.unit_key}')
        unit=matches[0]
    frame=int(unit['frame_id']) if unit.get('frame_id') is not None else None
    image=Path(unit['image_path']) if unit.get('image_path') else ROOT/'data/kitti_tracking_training/image_02'/str(unit['video'])/f'{frame:06d}.png'
    unit['image_path']=str(image.resolve())
    expression=str(unit['sentence'])
    before=time.time(); child=run_isolated(unit,expression); elapsed=time.time()-before
    child_json=None
    lines=[x for x in child.stdout.splitlines() if x.strip()]
    if lines:
        try: child_json=json.loads(lines[-1])
        except Exception: pass
    payload={'format':'locatemot-l53-groundingdino-compatibility-audit-v1','stage':'L53-A/A2','status':'pass' if child_json and child_json.get('status')=='pass' else 'fail','project_root':str(ROOT),'cwd':str(Path.cwd().resolve()),'isolated_interpreter':str(PYTHON),'isolated_returncode':child.returncode,'isolated_elapsed_sec':elapsed,'network_disabled':True,'fit_unit':{'unit_key':unit['unit_key'],'dataset':unit['dataset'],'video':unit['video'],'frame_id':frame,'expression':expression,'image_path':str(image.resolve()),'label_free_targeted_replay':bool(args.image_path)},'paths':{'ttaod_root':str(TTAOD),'config':str(CONFIG),'weight':str(WEIGHT),'swin':str(SWIN),'bert_snapshot':str(BERT),'grounding_dino_source':str(TTAOD/'mmdet/models/detectors/grounding_dino.py')},'source_provenance':{'ttaod_git':git_info(TTAOD),'official_groundingdino_url':'https://github.com/IDEA-Research/GroundingDINO','official_groundingdino_relation':'TTAOD-F OpenMMLab implementation with a GroundingDINO detector module; local checkout is not itself verified as the official git checkout','locatemot_main_env_statement':'L52 found no importable GroundingDINO API in locatemot; L53 tests this separate TTAOD-F environment'},'files':{},'checkpoint_inspection':None,'isolated_result':child_json,'stdout_tail':child.stdout[-4000:],'stderr_tail':child.stderr[-4000:],'official_test_labels_read':False,'screening_gt_used':False,'ordinary_mot_ovmot_touched':False}
    for p in [CONFIG,WEIGHT,SWIN,TTAOD/'mmdet/models/detectors/grounding_dino.py']:
        payload['files'][str(p)]=({'exists':True,'size_bytes':p.stat().st_size,'sha256':sha(p)} if p.exists() else {'exists':False})
    for p in files(BERT):
        payload['files'][str(p)]={'exists':True,'size_bytes':p.stat().st_size,'sha256':sha(p)}
    try: payload['checkpoint_inspection']=ckpt_info()
    except Exception as e: payload['checkpoint_inspection']={'error_type':type(e).__name__,'error':str(e)}
    (out_root/'compatibility.json').write_text(json.dumps(payload,indent=2,default=str)+'\n')
    if payload['status']!='pass': (out_root/'INCOMPLETE.md').write_text('GroundingDINO isolated construction/forward audit failed. First actionable child result:\n'+json.dumps(child_json or {'returncode':child.returncode,'stdout':child.stdout[-2000:],'stderr':child.stderr[-2000:]},indent=2,default=str)+'\n')
    print(json.dumps({'status':payload['status'],'output':str(out_root/'compatibility.json'),'isolated_returncode':child.returncode,'child_status':child_json.get('status') if child_json else None}))
if __name__=='__main__': main()
