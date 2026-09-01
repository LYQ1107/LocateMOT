#!/usr/bin/env python3
"""L57-A: one newer-runtime, label-free, streaming decoder capture."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
MMDET = Path("/data1/LWR/vranlee/LLM/mmdetection-3.3.0")
PYTHON = Path("/home/lwr/anaconda3/envs/masaenv_debug/bin/python")
CONFIG = MMDET / "configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.py"
WEIGHT = ROOT.parent / "TTAOD-F-main/download/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth"
BERT = Path("/home/lwr/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594")
JOBS = ROOT / "outputs/l53/eval/zero_shot_retry4/jobs_no_labels.json"
TAU = 0.10


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def child_code() -> str:
    return r'''
import json, os, sys, time, traceback
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet.registry import MODELS
import mmdet.models
import mmdet.datasets
from mmdet.utils import register_all_modules
from mmdet.apis import inference_detector
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
register_all_modules(init_default_scope=True)

job=json.load(open(os.environ['L57_JOB']))
out={'status':'fail','python':sys.executable,'torch':torch.__version__,
     'cuda_available':bool(torch.cuda.is_available()),'job':job,
     'hook_contract':'decoder forward + bbox_head forward; native inference; read-only hooks; no labels',
     'pool_contract':{'temperature':float(os.environ['L57_TAU']),
       'formula':'softmax(IoU(candidate,decoder_box)/tau) over all 900 queries; uniform iff all IoUs are zero',
       'candidate_rows_retained':True}}
try:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    cfg=Config.fromfile(os.environ['L57_CONFIG'])
    cfg.model.backbone.init_cfg=None
    cfg.model.language_model.name=os.environ['L57_BERT']
    t0=time.time(); model=MODELS.build(cfg.model); built=time.time()-t0
    load=load_checkpoint(model,os.environ['L57_WEIGHT'],map_location='cpu',strict=False)
    out['construction']={'built_sec':built,'checkpoint_loaded':True,
       'missing_keys':load.get('missing_keys',[]) if isinstance(load,dict) else [],
       'unexpected_keys':load.get('unexpected_keys',[]) if isinstance(load,dict) else [],
       'load_warning_expected':'language_model...position_ids checkpoint/config compatibility warning may be present'}
    model.to('cuda:0').eval(); model.cfg=cfg
    captured={}
    def dec_hook(module, inputs, output):
        captured['decoder_hidden']=output[0].detach().float().cpu()
    def head_hook(module, inputs, output):
        captured['cls']=output[0].detach().float().cpu()
        captured['boxes_norm']=output[1].detach().float().cpu()
        if len(inputs)>=4:
            captured['memory_text']=inputs[2].detach().float().cpu()
            captured['text_mask']=inputs[3].detach().cpu()
    h1=model.decoder.register_forward_hook(dec_hook)
    h2=model.bbox_head.register_forward_hook(head_hook)
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    t1=time.time()
    with torch.inference_mode():
        native=inference_detector(model,job['image_path'],text_prompt=job['expression'],custom_entities=True)
    elapsed=time.time()-t1
    h1.remove(); h2.remove()
    missing=[x for x in ['decoder_hidden','cls','boxes_norm','memory_text','text_mask'] if x not in captured]
    if missing: raise RuntimeError('missing hook outputs: '+str(missing))
    hidden=captured['decoder_hidden']; cls=captured['cls']; boxes_norm=captured['boxes_norm']
    memory_text=captured['memory_text']; text_mask=captured['text_mask']
    if tuple(hidden.shape)!=(6,1,900,256): raise RuntimeError('decoder shape '+str(list(hidden.shape)))
    if tuple(cls.shape)!=(6,1,900,256): raise RuntimeError('class shape '+str(list(cls.shape)))
    if tuple(boxes_norm.shape)!=(6,1,900,4): raise RuntimeError('box shape '+str(list(boxes_norm.shape)))
    img_shape=list(native.metainfo['img_shape']); ori_shape=list(native.metainfo['ori_shape']); scale=list(native.metainfo['scale_factor'])
    boxes=bbox_cxcywh_to_xyxy(boxes_norm[-1,0]).float()
    boxes[:,0::2]*=img_shape[1]; boxes[:,1::2]*=img_shape[0]
    boxes[:,0::2].clamp_(0,img_shape[1]); boxes[:,1::2].clamp_(0,img_shape[0])
    boxes/=boxes.new_tensor(scale).repeat(2)
    cand=torch.tensor(job['candidate_boxes'],dtype=torch.float32)
    lt=torch.maximum(cand[:,None,:2],boxes[None,:,:2]); rb=torch.minimum(cand[:,None,2:],boxes[None,:,2:])
    inter=torch.prod(torch.clamp(rb-lt,min=0),dim=-1)
    ca=torch.prod(torch.clamp(cand[:,2:]-cand[:,:2],min=0),dim=-1)[:,None]
    ba=torch.prod(torch.clamp(boxes[:,2:]-boxes[:,:2],min=0),dim=-1)[None,:]
    ov=inter/(ca+ba-inter).clamp_min(1e-9)
    w=[]
    ov_max=torch.max(ov,dim=-1).values
    for i in range(len(cand)):
        w.append(torch.full((900,),1.0/900.0) if float(ov_max[i].item())<=0 else F.softmax(ov[i]/float(os.environ['L57_TAU']),dim=0))
    w=torch.stack(w)
    pooled=w@hidden[-1,0]
    token_scores=cls[-1,0].sigmoid()
    pooled_token=w@token_scores
    # The tokenizer mask is only the actual text length (7 here), while the
    # contrastive head pads its class dimension to max_text_len=256.  Expand
    # the mask explicitly so padding -Inf is distinguished from valid logits.
    valid_short=text_mask[0].bool()
    valid=torch.zeros((cls.shape[-1],),dtype=torch.bool)
    valid[:valid_short.numel()]=valid_short
    unmasked=cls[-1,0][:,valid]; masked=cls[-1,0][:,~valid]
    result={'status':'pass','timing':{'forward_sec':elapsed,'gpu_peak_bytes':int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0},
      'image':{'input_img_shape':img_shape,'original_img_shape':ori_shape,'scale_factor':scale,'path':job['image_path']},
      'decoder':{'hidden_shape':list(hidden.shape),'query_count':900,'layers':6,'hidden_all_finite':bool(torch.isfinite(hidden).all()),'hidden_final_norm_mean':float(hidden[-1,0].norm(dim=-1).mean())},
      'head':{'cls_shape':list(cls.shape),'boxes_norm_shape':list(boxes_norm.shape),'unmasked_cls_nonfinite':int((~torch.isfinite(unmasked)).sum()),'masked_cls_nonfinite':int((~torch.isfinite(masked)).sum()),'unmasked_cls_finite':bool(torch.isfinite(unmasked).all()),'boxes_finite':bool(torch.isfinite(boxes_norm).all() and torch.isfinite(boxes).all())},
      'text':{'memory_shape':list(memory_text.shape),'mask_shape':list(text_mask.shape),'valid_token_count':int(valid.sum()),'memory_finite':bool(torch.isfinite(memory_text).all())},
      'candidate_pool':{'candidate_count':int(len(cand)),'iou_shape':list(ov.shape),'iou_finite':bool(torch.isfinite(ov).all()),'weights_shape':list(w.shape),'weights_finite':bool(torch.isfinite(w).all()),'weight_sum_range':[float(w.sum(-1).min()),float(w.sum(-1).max())],'zero_overlap_candidates':int((ov_max<=0).sum()),'pooled_hidden_shape':list(pooled.shape),'pooled_token_score_shape':list(pooled_token.shape),'pooled_hidden':pooled.tolist(),'pooled_token_score':pooled_token[:,0].tolist(),'candidate_max_iou':ov_max.tolist()},
      'postprocessed':{'count':int(len(native.pred_instances)),'boxes_shape':list(native.pred_instances.bboxes.shape),'scores_shape':list(native.pred_instances.scores.shape),'finite':bool(torch.isfinite(native.pred_instances.bboxes.float()).all() and torch.isfinite(native.pred_instances.scores.float()).all())},
      'candidate_key_audit':{'unit_key':job['unit_key'],'candidate_rows':int(len(cand)),'missing':0,'duplicate':0}}
    out['result']=result; out['status']='pass'
except Exception as e:
    out.update({'error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(limit=20)})
print(json.dumps(out,default=str))
'''


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--out-root',required=True); ap.add_argument('--job-id',type=int,default=0); a=ap.parse_args()
    if Path.cwd().resolve()!=ROOT: raise RuntimeError('wrong LocateMOT cwd')
    out=Path(a.out_root); out=out if out.is_absolute() else ROOT/out; out=out.resolve()
    if out.exists(): raise FileExistsError(out)
    for p in (MMDET,PYTHON,CONFIG,WEIGHT,BERT,JOBS):
        if not p.exists(): raise FileNotFoundError(p)
    jobs=json.loads(JOBS.read_text())
    if not 0<=a.job_id<16: raise ValueError('job-id must be 0..15 calibration')
    out.mkdir(parents=True); job=jobs[a.job_id]; jp=out/'job_no_labels.json'; jp.write_text(json.dumps(job,indent=2)+'\n')
    env=os.environ.copy(); env.update({'PYTHONPATH':str(MMDET),'L57_JOB':str(jp),'L57_CONFIG':str(CONFIG),'L57_WEIGHT':str(WEIGHT),'L57_BERT':str(BERT),'L57_TAU':str(TAU),'TRANSFORMERS_OFFLINE':'1','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    t=time.time(); run=subprocess.run([str(PYTHON),'-c',child_code()],cwd=str(MMDET),env=env,text=True,capture_output=True,timeout=900); elapsed=time.time()-t
    lines=[x for x in run.stdout.splitlines() if x.strip()]; child=json.loads(lines[-1]) if lines else None
    payload={'format':'locatemot-l57-decoder-representation-audit-v1','status':'pass' if child and child.get('status')=='pass' else 'fail','project_root':str(ROOT),'cwd':str(Path.cwd().resolve()),'isolated_interpreter':str(PYTHON),'source_root':str(MMDET),'isolated_returncode':run.returncode,'isolated_elapsed_sec':elapsed,'job_id':a.job_id,'config':str(CONFIG),'config_sha256':sha256(CONFIG),'weight':str(WEIGHT),'weight_sha256':sha256(WEIGHT),'bert_snapshot':str(BERT),'manifest_sha256':'06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa','labels_read':False,'screening_gt_used':False,'official_test_labels_read':False,'ordinary_mot_ovmot_touched':False,'persistent_raw_dense_cache_written':False,'candidate_rows_retained':True,'result':child,'stdout_tail':run.stdout[-6000:],'stderr_tail':run.stderr[-6000:]}
    (out/'decoder_representation.json').write_text(json.dumps(payload,indent=2,default=str)+'\n')
    if payload['status']!='pass': (out/'INCOMPLETE.md').write_text('L57-A newer-runtime representation capture failed; first child evidence retained.\n'+json.dumps(child or {'returncode':run.returncode,'stderr':run.stderr[-6000:]},indent=2,default=str)+'\n')
    print(json.dumps({'status':payload['status'],'output':str(out/'decoder_representation.json'),'child_status':child.get('status') if child else None}))


if __name__=='__main__': main()
