#!/usr/bin/env python3
"""L56-A: one label-free streaming decoder-representation capture.

No labels are loaded.  The child uses the known-good L53 native inference path,
captures decoder outputs before postprocessing, and immediately pools all 900
queries against the complete L19 candidate set.  The single-unit JSON is an
audit artifact, not a reusable feature cache or training target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
TTAOD = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TTAOD-F-main")
PYTHON = Path("/home/lwr/anaconda3/envs/ttaod_f/bin/python")
CONFIG = TTAOD / "configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.py"
WEIGHT = TTAOD / "download/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth"
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

job=json.load(open(os.environ['L56_JOB']))
out={'status':'fail','python':sys.executable,'torch':torch.__version__,
     'cuda_available':bool(torch.cuda.is_available()),'job':job,
     'hook_contract':'decoder forward + bbox_head forward; native inference first; no manual tokenizer before inference',
     'pool_contract':{'temperature':float(os.environ['L56_TAU']),
       'formula':'softmax(IoU(candidate,decoder_box)/tau) over all decoder queries; uniform over all queries when max IoU=0',
       'zero_overlap':'uniform all-query weights','candidate_rows_retained':True}}
try:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    cfg=Config.fromfile(os.environ['L56_CONFIG'])
    cfg.model.backbone.init_cfg=None
    cfg.model.language_model.name=os.environ['L56_BERT']
    t0=time.time(); model=MODELS.build(cfg.model); built=time.time()-t0
    loaded=load_checkpoint(model,os.environ['L56_WEIGHT'],map_location='cpu',strict=False)
    out['construction']={'built_sec':built,'checkpoint_loaded':True,
       'missing_keys':loaded.get('missing_keys',[]) if isinstance(loaded,dict) else [],
       'unexpected_keys':loaded.get('unexpected_keys',[]) if isinstance(loaded,dict) else []}
    model.to('cuda:0').eval(); model.cfg=cfg
    captured={}
    def dec_hook(module, inputs, output):
        # GroundingDinoTransformerDecoder returns (inter_states, references).
        captured['decoder_hidden']=output[0].detach().float().cpu()
        captured['decoder_refs']=output[1]
    def head_hook(module, inputs, output):
        captured['cls']=output[0].detach().float().cpu()
        captured['boxes_norm']=output[1].detach().float().cpu()
        if len(inputs) >= 4:
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
    required=['decoder_hidden','cls','boxes_norm','memory_text','text_mask']
    missing=[x for x in required if x not in captured]
    if missing: raise RuntimeError('missing hook outputs: '+str(missing))
    hidden=captured['decoder_hidden']; cls=captured['cls']; boxes_norm=captured['boxes_norm']
    memory_text=captured['memory_text']; text_mask=captured['text_mask']
    if hidden.dim()!=4 or hidden.shape[0] != 6 or hidden.shape[1] != 1 or hidden.shape[2] != 900 or hidden.shape[3] != 256:
        raise RuntimeError('unexpected decoder hidden shape '+str(list(hidden.shape)))
    if cls.shape[:3] != (6,1,900) or boxes_norm.shape[:3] != (6,1,900):
        raise RuntimeError('unexpected head shapes cls=%s boxes=%s'%(list(cls.shape),list(boxes_norm.shape)))
    # Compute original-pixel boxes exactly as the postprocessor does.
    img_shape=list(native.metainfo['img_shape']); ori_shape=list(native.metainfo['ori_shape']); scale=list(native.metainfo['scale_factor'])
    b=bbox_cxcywh_to_xyxy(boxes_norm[-1,0]).float()
    b[:,0::2]*=img_shape[1]; b[:,1::2]*=img_shape[0]
    b[:,0::2].clamp_(0,img_shape[1]); b[:,1::2].clamp_(0,img_shape[0])
    b/=b.new_tensor(scale).repeat(2)
    cand=torch.tensor(job['candidate_boxes'],dtype=torch.float32)
    lt=torch.maximum(cand[:,None,:2],b[None,:,:2]); rb=torch.minimum(cand[:,None,2:],b[None,:,2:])
    inter=torch.prod(torch.clamp(rb-lt,min=0),dim=-1)
    ca=torch.prod(torch.clamp(cand[:,2:]-cand[:,:2],min=0),dim=-1)[:,None]
    ba=torch.prod(torch.clamp(b[:,2:]-b[:,:2],min=0),dim=-1)[None,:]
    ov=inter/(ca+ba-inter).clamp_min(1e-9)
    weights=[]
    for i in range(len(cand)):
        if float(ov[i].max()) <= 0.0: weights.append(torch.full((900,),1.0/900.0))
        else: weights.append(F.softmax(ov[i]/float(os.environ['L56_TAU']),dim=0))
    w=torch.stack(weights)
    pooled=w @ hidden[-1,0]
    token_scores=cls[-1,0].sigmoid()
    pooled_token=w @ token_scores
    valid_mask=text_mask[0].bool()
    raw_unmasked=cls[-1,0][:,valid_mask]
    result={
      'status':'pass','timing':{'forward_sec':elapsed,'gpu_peak_bytes':int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0},
      'image':{'input_img_shape':img_shape,'original_img_shape':ori_shape,'scale_factor':scale,'path':job['image_path']},
      'decoder':{'hidden_shape':list(hidden.shape),'final_hidden_shape':list(hidden[-1,0].shape),'query_count':int(hidden.shape[2]),'layer_count':int(hidden.shape[0]),'hidden_all_finite':bool(torch.isfinite(hidden).all()),'hidden_norm_mean':float(hidden[-1,0].norm(dim=-1).mean())},
      'head':{'cls_shape':list(cls.shape),'boxes_norm_shape':list(boxes_norm.shape),'final_boxes_original_shape':list(b.shape),'unmasked_cls_nonfinite':int((~torch.isfinite(raw_unmasked)).sum()),'masked_cls_nonfinite':int((~torch.isfinite(cls[-1,0][:,~valid_mask])).sum()),'boxes_finite':bool(torch.isfinite(boxes_norm).all() and torch.isfinite(b).all())},
      'text':{'memory_shape':list(memory_text.shape),'mask_shape':list(text_mask.shape),'valid_token_count':int(valid_mask.sum()),'memory_finite':bool(torch.isfinite(memory_text).all()),'mask_true_count':int(valid_mask.sum())},
      'candidate_pool':{'candidate_count':int(len(cand)),'iou_shape':list(ov.shape),'iou_finite':bool(torch.isfinite(ov).all()),'weights_shape':list(w.shape),'weights_finite':bool(torch.isfinite(w).all()),'weight_row_sums_min':float(w.sum(-1).min()),'weight_row_sums_max':float(w.sum(-1).max()),'zero_overlap_candidates':int((ov.max(-1)<=0).sum()),'pooled_hidden_shape':list(pooled.shape),'pooled_entity_score_shape':list(pooled_token.shape),'pooled_hidden':pooled.tolist(),'pooled_entity_score':pooled_token[:,0].tolist() if pooled_token.shape[1] else [],'candidate_max_iou':ov.max(-1).values.tolist()},
      'postprocessed':{'count':int(len(native.pred_instances)),'boxes_shape':list(native.pred_instances.bboxes.shape),'scores_shape':list(native.pred_instances.scores.shape),'finite':bool(torch.isfinite(native.pred_instances.bboxes.float()).all() and torch.isfinite(native.pred_instances.scores.float()).all())},
      'candidate_key':{'unit_key':job['unit_key'],'candidate_rows_retained':int(len(cand)),'missing':0,'duplicate':0}
    }
    out['result']=result; out['status']='pass'
except Exception as e:
    out.update({'error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(limit=20)})
print(json.dumps(out,default=str))
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--job-id", type=int, default=0)
    args = ap.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("wrong LocateMOT cwd")
    out = Path(args.out_root)
    if not out.is_absolute(): out = ROOT / out
    out = out.resolve()
    if out.exists(): raise FileExistsError(out)
    jobs=json.loads(JOBS.read_text())
    if not 0 <= args.job_id < 16: raise ValueError("job-id must be calibration 0..15")
    out.mkdir(parents=True)
    job=jobs[args.job_id]
    job_path=out/'job_no_labels.json'; job_path.write_text(json.dumps(job,indent=2)+'\n')
    env=os.environ.copy(); env.update({'PYTHONPATH':str(TTAOD),'L56_JOB':str(job_path),'L56_CONFIG':str(CONFIG),'L56_WEIGHT':str(WEIGHT),'L56_BERT':str(BERT),'L56_TAU':str(TAU),'TRANSFORMERS_OFFLINE':'1','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    started=time.time(); run=subprocess.run([str(PYTHON),'-c',child_code()],cwd=str(TTAOD),env=env,text=True,capture_output=True,timeout=900); elapsed=time.time()-started
    lines=[x for x in run.stdout.splitlines() if x.strip()]
    child=json.loads(lines[-1]) if lines else None
    payload={'format':'locatemot-l56-decoder-representation-audit-v1','status':'pass' if child and child.get('status')=='pass' else 'fail','project_root':str(ROOT),'cwd':str(Path.cwd().resolve()),'isolated_interpreter':str(PYTHON),'isolated_returncode':run.returncode,'isolated_elapsed_sec':elapsed,'job_id':args.job_id,'config':str(CONFIG),'config_sha256':sha256(CONFIG),'weight':str(WEIGHT),'weight_sha256':sha256(WEIGHT),'bert_snapshot':str(BERT),'manifest_sha256':'06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa','labels_read':False,'screening_gt_used':False,'official_test_labels_read':False,'ordinary_mot_ovmot_touched':False,'persistent_dense_cache_written':False,'candidate_rows_retained':True,'result':child,'stdout_tail':run.stdout[-5000:],'stderr_tail':run.stderr[-5000:]}
    (out/'decoder_representation.json').write_text(json.dumps(payload,indent=2,default=str)+'\n')
    if payload['status']!='pass': (out/'INCOMPLETE.md').write_text('L56 decoder representation capture failed; retained first child evidence.\n'+json.dumps(child or {'returncode':run.returncode,'stderr':run.stderr[-5000:]},indent=2,default=str)+'\n')
    print(json.dumps({'status':payload['status'],'output':str(out/'decoder_representation.json'),'child_status':child.get('status') if child else None}))


if __name__ == '__main__': main()
