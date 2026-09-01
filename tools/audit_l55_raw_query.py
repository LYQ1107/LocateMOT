#!/usr/bin/env python3
"""One clean label-free raw decoder-query compatibility attempt for L55."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess
from pathlib import Path

ROOT=Path('/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT')
TTAOD=Path('/data1/LWR/vranlee/SERVER_ONLY/avis/TTAOD-F-main')
PYTHON=Path('/home/lwr/anaconda3/envs/ttaod_f/bin/python')
CONFIG=TTAOD/'configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.py'
WEIGHT=TTAOD/'download/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'
BERT=Path('/home/lwr/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594')
JOBS=ROOT/'outputs/l53/eval/zero_shot_retry4/jobs_no_labels.json'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def child_code():
    return r'''
import json, os, sys, time, traceback, torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules
import mmdet.models
import mmdet.datasets
register_all_modules(init_default_scope=True)
from mmdet.apis import inference_detector
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
job=json.load(open(os.environ['L55_JOB']))
out={'status':'fail','job':job,'python':sys.executable,'torch':torch.__version__,
     'cuda_available':bool(torch.cuda.is_available()),'construction':{},'raw_query':{},
     'timing':{},'hook_contract':'bbox_head forward hook; no manual tokenizer/model text call before native inference'}
try:
    cfg=Config.fromfile(os.environ['L55_CONFIG'])
    cfg.model.backbone.init_cfg=None
    cfg.model.language_model.name=os.environ['L55_BERT']
    t=time.time(); model=MODELS.build(cfg.model); out['construction']['built_sec']=time.time()-t
    load=load_checkpoint(model,os.environ['L55_WEIGHT'],map_location='cpu',strict=False)
    out['construction'].update({'checkpoint_loaded':True,'missing_keys':load.get('missing_keys',[]) if isinstance(load,dict) else [],'unexpected_keys':load.get('unexpected_keys',[]) if isinstance(load,dict) else []})
    model.to('cuda:0').eval(); model.cfg=cfg; captured={}
    def capture(module,inputs,output):
        captured['all_cls']=output[0].detach().float().cpu()
        captured['all_box']=output[1].detach().float().cpu()
    hook=model.bbox_head.register_forward_hook(capture)
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    t=time.time()
    with torch.inference_mode():
        native=inference_detector(model,job['image_path'],text_prompt=job['expression'],custom_entities=True)
    hook.remove(); elapsed=time.time()-t
    if 'all_cls' not in captured: raise RuntimeError('bbox_head hook captured no decoder outputs')
    # Only after native inference: inspect the exact entity/token map used by this API.
    tokenized,caption,tokens_positive,entities=model.get_tokens_and_prompts(job['expression'],True)
    pos_map,_=model.get_positive_map(tokenized,tokens_positive)
    all_cls=captured['all_cls']; all_box=captured['all_box']; raw=all_cls[-1,0]; boxes_norm=all_box[-1,0]
    img_shape=list(native.metainfo['img_shape']); ori_shape=list(native.metainfo['ori_shape']); scale=list(native.metainfo['scale_factor'])
    boxes=bbox_cxcywh_to_xyxy(boxes_norm).float(); boxes[:,0::2]*=img_shape[1]; boxes[:,1::2]*=img_shape[0]; boxes[:,0::2].clamp_(0,img_shape[1]); boxes[:,1::2].clamp_(0,img_shape[0]); boxes/=boxes.new_tensor(scale).repeat(2)
    sig=raw.sigmoid(); ent=[]
    for k in sorted(pos_map,key=lambda x:int(x)):
        idx=torch.tensor(pos_map[k],dtype=torch.long); ent.append(sig[:,idx].mean(-1))
    ent=torch.stack(ent,1) if ent else torch.empty((raw.shape[0],0))
    out['text']={'original_expression':job['expression'],'caption_string':caption,'custom_entities':True,'entities':entities,'entity_count':len(entities),'token_count':int(tokenized.input_ids.shape[1]),'tokenized_input_ids_shape':list(tokenized.input_ids.shape),'attention_mask_shape':list(tokenized.attention_mask.shape),'tokens_positive':tokens_positive,'positive_map_label_to_token':{str(k):list(v) for k,v in pos_map.items()},'positive_map_shape':list(_ .shape)}
    out['raw_query']={'decoder_layers':int(all_cls.shape[0]),'query_count':int(raw.shape[0]),'raw_cls_logits_shape':list(raw.shape),'raw_cls_logits':raw.tolist(),'sigmoid_token_scores_shape':list(sig.shape),'entity_scores_shape':list(ent.shape),'entity_scores':ent.tolist(),'normalized_boxes_cxcywh':boxes_norm.tolist(),'final_boxes_xyxy_original':boxes.tolist(),'input_img_shape':img_shape,'original_img_shape':ori_shape,'scale_factor':scale,'configured_max_per_img':int(model.bbox_head.test_cfg.get('max_per_img',-1)),'finite':bool(torch.isfinite(raw).all() and torch.isfinite(boxes_norm).all() and torch.isfinite(ent).all()),'postprocess_contract':'last decoder layer -> sigmoid token logits -> positive-map mean -> topk(max_per_img); raw capture bypassed only postprocess'}
    out['native_postprocessed']={'count':int(len(native.pred_instances)),'scores_shape':list(native.pred_instances.scores.shape),'labels_shape':list(native.pred_instances.labels.shape),'boxes_shape':list(native.pred_instances.bboxes.shape),'finite':bool(torch.isfinite(native.pred_instances.scores.float()).all())}
    out['timing']={'forward_sec':elapsed,'gpu_peak_bytes':int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0}; out['status']='pass'
except Exception as e:
    out.update({'error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(limit=20)})
print(json.dumps(out,default=str))
'''

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out-root',required=True); p.add_argument('--job-id',type=int,default=0); a=p.parse_args()
    if Path.cwd().resolve()!=ROOT: raise RuntimeError('wrong LocateMOT cwd')
    out=Path(a.out_root); out=out if out.is_absolute() else ROOT/out; out=out.resolve()
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    jobs=json.loads(JOBS.read_text());
    if not 0<=a.job_id<16: raise ValueError('job-id must be one of the 16 calibration jobs')
    job=jobs[a.job_id]; job_path=out/'job_no_labels.json'; job_path.write_text(json.dumps(job,indent=2)+'\n')
    env=os.environ.copy(); env.update({'PYTHONPATH':str(TTAOD),'L55_JOB':str(job_path),'L55_CONFIG':str(CONFIG),'L55_WEIGHT':str(WEIGHT),'L55_BERT':str(BERT),'TRANSFORMERS_OFFLINE':'1','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    run=subprocess.run([str(PYTHON),'-c',child_code()],cwd=str(TTAOD),env=env,text=True,capture_output=True,timeout=900)
    lines=[x for x in run.stdout.splitlines() if x.strip()]; child=json.loads(lines[-1]) if lines else None
    payload={'format':'locatemot-l55-raw-query-compatibility-v1','status':'pass' if child and child.get('status')=='pass' else 'fail','project_root':str(ROOT),'cwd':str(Path.cwd().resolve()),'isolated_interpreter':str(PYTHON),'isolated_returncode':run.returncode,'job_id':a.job_id,'job':job,'config':str(CONFIG),'config_sha256':sha(CONFIG),'weight':str(WEIGHT),'weight_sha256':sha(WEIGHT),'bert_snapshot':str(BERT),'network_disabled':True,'labels_read':False,'screening_gt_used':False,'official_test_labels_read':False,'ordinary_mot_ovmot_touched':False,'result':child,'stdout_tail':run.stdout[-4000:],'stderr_tail':run.stderr[-4000:]}
    (out/'raw_query.json').write_text(json.dumps(payload,indent=2,default=str)+'\n')
    if payload['status']!='pass': (out/'INCOMPLETE.md').write_text('Single clean raw-query compatibility attempt failed; first child evidence:\n'+json.dumps(child or {'returncode':run.returncode,'stderr':run.stderr[-4000:]},indent=2,default=str)+'\n')
    print(json.dumps({'status':payload['status'],'output':str(out/'raw_query.json'),'child_status':child.get('status') if child else None}))
if __name__=='__main__': main()
