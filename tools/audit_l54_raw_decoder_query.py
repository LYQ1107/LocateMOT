#!/usr/bin/env python3
"""One-unit, label-free audit of GroundingDINO decoder-query outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path('/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT')
TTAOD = Path('/data1/LWR/vranlee/SERVER_ONLY/avis/TTAOD-F-main')
PYTHON = Path('/home/lwr/anaconda3/envs/ttaod_f/bin/python')
CONFIG = TTAOD / 'configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.py'
WEIGHT = TTAOD / 'download/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'
BERT = Path('/home/lwr/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594')
JOBS = ROOT / 'outputs/l53/eval/zero_shot_retry4/jobs_no_labels.json'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def child_code() -> str:
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

job=json.load(open(os.environ['L54_JOB']))
cfg=Config.fromfile(os.environ['L54_CONFIG'])
cfg.model.backbone.init_cfg=None
cfg.model.language_model.name=os.environ['L54_BERT']
result={'status':'fail','job':job,'python':sys.executable,'torch':torch.__version__,
        'cuda_available':bool(torch.cuda.is_available()),'construction':{},'text':{},
        'raw_query':{},'timing':{}}
try:
  t0=time.time()
  model=MODELS.build(cfg.model)
  result['construction']['built_sec']=time.time()-t0
  load=load_checkpoint(model,os.environ['L54_WEIGHT'],map_location='cpu',strict=False)
  result['construction']['checkpoint_loaded']=True
  result['construction']['missing_keys']=load.get('missing_keys',[]) if isinstance(load,dict) else []
  result['construction']['unexpected_keys']=load.get('unexpected_keys',[]) if isinstance(load,dict) else []
  model.to('cuda:0').eval(); model.cfg=cfg

  tokenized, caption, tokens_positive, entities = model.get_tokens_and_prompts(job['expression'],True)
  pos_map_label_to_token, positive_map = model.get_positive_map(tokenized,tokens_positive)
  result['text']={'original_expression':job['expression'],'caption_string':caption,
    'custom_entities':True,'entities':entities,'tokenized_input_ids_shape':list(tokenized.input_ids.shape),
    'attention_mask_shape':list(tokenized.attention_mask.shape),'tokens_positive':tokens_positive,
    'positive_map_label_to_token':{str(k):list(v) for k,v in pos_map_label_to_token.items()},
    'positive_map_shape':list(positive_map.shape),'entity_count':len(entities),
    'token_count':int(tokenized.input_ids.shape[1])}
  if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
  t0=time.time()
  captured={}
  def capture_head(module, inputs, output):
    captured['all_cls']=output[0].detach().float().cpu()
    captured['all_box']=output[1].detach().float().cpu()
  hook=model.bbox_head.register_forward_hook(capture_head)
  with torch.no_grad():
    native=inference_detector(model,job['image_path'],text_prompt=job['expression'],custom_entities=True)
  hook.remove()
  elapsed=time.time()-t0
  if 'all_cls' not in captured or 'all_box' not in captured:
    raise RuntimeError('bbox-head forward hook did not capture raw decoder outputs')
  all_cls=captured['all_cls']; all_box=captured['all_box']
  raw_cls=all_cls[-1][0].detach().float().cpu()
  norm_box=all_box[-1][0].detach().float().cpu()
  img_shape=list(native.metainfo['img_shape']); ori_shape=list(native.metainfo['ori_shape']); scale=list(native.metainfo['scale_factor'])
  boxes=bbox_cxcywh_to_xyxy(norm_box).float()
  boxes[:,0::2]*=img_shape[1]; boxes[:,1::2]*=img_shape[0]
  boxes[:,0::2].clamp_(0,img_shape[1]); boxes[:,1::2].clamp_(0,img_shape[0])
  boxes/=boxes.new_tensor(scale).repeat(2)
  tok_score=raw_cls.sigmoid()
  entity_scores=[]
  for key in sorted(pos_map_label_to_token,key=lambda x:int(x)):
    inds=torch.tensor(pos_map_label_to_token[key],dtype=torch.long)
    entity_scores.append(tok_score[:,inds].mean(-1))
  entity_score=torch.stack(entity_scores,dim=1) if entity_scores else torch.empty((raw_cls.shape[0],0))
  result['raw_query']={'decoder_layers':int(all_cls.shape[0]),'query_count':int(raw_cls.shape[0]),
    'raw_cls_logits_shape':list(raw_cls.shape),'raw_cls_logits':raw_cls.tolist(),
    'sigmoid_token_scores_shape':list(tok_score.shape),'entity_scores_shape':list(entity_score.shape),
    'entity_scores':entity_score.tolist(),'normalized_box_shape':list(norm_box.shape),
    'normalized_boxes_cxcywh':norm_box.tolist(),'final_boxes_xyxy_original':boxes.tolist(),
    'input_img_shape':img_shape,'original_img_shape':ori_shape,'scale_factor':scale,
    'finite':bool(torch.isfinite(raw_cls).all() and torch.isfinite(norm_box).all() and torch.isfinite(entity_score).all()),
    'final_query_count_before_postprocess':int(raw_cls.shape[0]),'configured_max_per_img':int(model.bbox_head.test_cfg.get('max_per_img',-1)),
    'postprocess_note':'bbox head uses last decoder layer, sigmoid token logits, positive-map mean, then topk(max_per_img); this audit bypasses topk only after the native forward'}
  result['native_postprocessed']={'type':type(native).__name__,'count':int(len(native.pred_instances)),
    'scores_shape':list(native.pred_instances.scores.shape),'labels_shape':list(native.pred_instances.labels.shape),
    'boxes_shape':list(native.pred_instances.bboxes.shape),
    'finite':bool(torch.isfinite(native.pred_instances.scores.float()).all()),
    'metainfo':{k:native.metainfo[k] for k in ['img_shape','ori_shape','scale_factor'] if k in native.metainfo}}
  result['timing']={'forward_sec':elapsed,'gpu_peak_bytes':int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0}
  result['status']='pass'
except Exception as e:
  result['error_type']=type(e).__name__; result['error']=str(e); result['traceback']=traceback.format_exc(limit=20)
print(json.dumps(result,default=str))
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-root', required=True)
    p.add_argument('--job-id', type=int, default=0)
    args = p.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError('wrong LocateMOT cwd')
    out = Path(args.out_root)
    if not out.is_absolute():
        out = ROOT / out
    out = out.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    jobs = json.loads(JOBS.read_text())
    if args.job_id < 0 or args.job_id >= 16:
        raise ValueError('raw-query audit is restricted to the 16 pre-registered calibration jobs')
    job = jobs[args.job_id]
    job_path = out / 'job_no_labels.json'
    job_path.write_text(json.dumps(job, indent=2) + '\n')
    env = os.environ.copy()
    env.update({'PYTHONPATH': str(TTAOD), 'L54_JOB': str(job_path),
                'L54_CONFIG': str(CONFIG), 'L54_WEIGHT': str(WEIGHT),
                'L54_BERT': str(BERT), 'TRANSFORMERS_OFFLINE': '1',
                'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
                'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    run = subprocess.run([str(PYTHON), '-c', child_code()], cwd=str(TTAOD),
                         env=env, text=True, capture_output=True, timeout=900)
    lines = [x for x in run.stdout.splitlines() if x.strip()]
    child = json.loads(lines[-1]) if lines else None
    payload = {'format': 'locatemot-l54-raw-decoder-query-audit-v1',
               'status': 'pass' if child and child.get('status') == 'pass' else 'fail',
               'project_root': str(ROOT), 'cwd': str(Path.cwd().resolve()),
               'isolated_interpreter': str(PYTHON), 'isolated_returncode': run.returncode,
               'config': str(CONFIG), 'config_sha256': sha(CONFIG),
               'weight': str(WEIGHT), 'weight_sha256': sha(WEIGHT),
               'bert_snapshot': str(BERT), 'job_id': args.job_id, 'job': job,
               'calibration_only': True, 'labels_read': False,
               'screening_gt_used': False, 'official_test_labels_read': False,
               'ordinary_mot_ovmot_touched': False, 'network_disabled': True,
               'result': child, 'stdout_tail': run.stdout[-4000:],
               'stderr_tail': run.stderr[-4000:]}
    (out / 'raw_query.json').write_text(json.dumps(payload, indent=2, default=str) + '\n')
    if payload['status'] != 'pass':
        (out / 'INCOMPLETE.md').write_text('Raw decoder-query audit failed; first child error follows:\n' + json.dumps(child or {'returncode': run.returncode, 'stderr': run.stderr[-4000:]}, indent=2, default=str) + '\n')
    print(json.dumps({'status': payload['status'], 'output': str(out / 'raw_query.json'), 'child_status': child.get('status') if child else None}))


if __name__ == '__main__':
    main()
