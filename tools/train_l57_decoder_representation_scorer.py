#!/usr/bin/env python3
"""L57-B 100-step fit-only smoke with streaming frozen GroundingDINO."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
MMDET = Path("/data1/LWR/vranlee/LLM/mmdetection-3.3.0")
PYTHON = Path("/home/lwr/anaconda3/envs/masaenv_debug/bin/python")
CONFIG = MMDET / "configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.py"
WEIGHT = ROOT.parent / "TTAOD-F-main/download/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth"
BERT = Path("/home/lwr/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594")
UNITS = ROOT / "outputs/l49/data/train_units.jsonl"
SEED = 20260829
sys.path.insert(0, str(ROOT))
from locatemot.models.l57_decoder_representation_scorer import L57DecoderRepresentationScorer  # noqa: E402


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def load_units():
    return [json.loads(x) for x in UNITS.read_text().splitlines()
            if x.strip() and json.loads(x).get('split') == 'fit'
            and json.loads(x).get('dataset') in ('refer_kitti_v1','refer_kitti_v2')]


def deterministic_unit_order(units, seed: int, limit: int | None = None):
    """Seeded bucket round-robin, then a seeded remainder."""
    rng=random.Random(seed); buckets={}
    for unit in units:
        buckets.setdefault((unit['dataset'],unit.get('category','unknown')),[]).append(unit)
    for values in buckets.values(): rng.shuffle(values)
    required=[(d,c) for d in ('refer_kitti_v1','refer_kitti_v2')
              for c in ('positive','multi_positive','inactive','present_uncovered')]
    order=[]
    while True:
        added=False
        for key in required:
            if buckets.get(key): order.append(buckets[key].pop(0)); added=True
        if not added: break
    rest=[u for values in buckets.values() for u in values]; rng.shuffle(rest); order.extend(rest)
    return order if limit is None else order[:limit]


def balanced_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target=target.bool(); terms=[]
    if target.any(): terms.append(F.binary_cross_entropy_with_logits(logits[target],torch.ones_like(logits[target])))
    if (~target).any(): terms.append(F.binary_cross_entropy_with_logits(logits[~target],torch.zeros_like(logits[~target])))
    return torch.stack(terms).mean() if terms else logits.new_zeros(())


def unit_loss(out: dict, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
    score=out['relevance_logit']; zero=score.new_zeros(())
    pos=torch.nonzero(target,as_tuple=False).flatten(); neg=torch.nonzero(~target,as_tuple=False).flatten()
    with torch.no_grad():
        hard=neg[torch.argsort(score.detach()[neg],descending=True)[:min(24,len(neg))]] if len(neg) else neg
    bce=balanced_bce(score,target)
    pair=F.softplus(.2+score[hard][None,:]-score[pos][:,None]).mean() if len(pos) and len(hard) else zero
    listwise=torch.logsumexp(score,0)-torch.logsumexp(score[pos],0) if len(pos) else zero
    minpos=F.softplus(.2+score[hard].max()-score[pos]).mean() if len(pos) and len(hard) else zero
    null_target=score.new_tensor(float(not bool(target.any())))
    null=F.binary_cross_entropy_with_logits(out['null_logit'].reshape(()),null_target)
    prob=score.sigmoid(); calibration=((prob-target.float())**2).mean()
    total=bce+pair+.5*listwise+.5*minpos+.2*null+.1*calibration
    return total, {'total':float(total.detach()),'bce':float(bce.detach()),'pairwise':float(pair.detach()),'listwise':float(listwise.detach()),'min_positive':float(minpos.detach()),'null':float(null.detach()),'calibration':float(calibration.detach()),'positive_count':int(len(pos)),'negative_count':int(len(neg)),'hard_negative_count':int(len(hard)),'null_target':float(null_target)}


def build_detector():
    from mmengine.config import Config
    from mmengine.runner import load_checkpoint
    from mmdet.registry import MODELS
    import mmdet.models, mmdet.datasets
    from mmdet.utils import register_all_modules
    register_all_modules(init_default_scope=True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    cfg=Config.fromfile(str(CONFIG)); cfg.model.backbone.init_cfg=None; cfg.model.language_model.name=str(BERT)
    model=MODELS.build(cfg.model); loaded=load_checkpoint(model,str(WEIGHT),map_location='cpu',strict=False)
    model.to('cuda:0').eval()
    # mmdet's native inference helper reads model.cfg; keep this identical to
    # the verified L57 capture path without changing the detector contract.
    model.cfg=cfg
    for p in model.parameters(): p.requires_grad_(False)
    return model, cfg, loaded


def stream_rep(detector, unit: dict, bank: dict, tau: float=0.10):
    from mmdet.apis import inference_detector
    from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
    tensors=bank['tensors']; begin,end=int(unit['begin']),int(unit['end'])
    cand=tensors['box'][begin:end].float().to('cuda:0')
    numeric=torch.cat([tensors['geometry'][begin:end].float(),tensors['motion'][begin:end].float(),tensors['lifecycle'][begin:end].float(),tensors['objectness'][begin:end].float().reshape(-1,1)],1).to('cuda:0')
    captured={}
    def dh(m,i,o): captured['hidden']=o[0].detach()
    def hh(m,i,o): captured['cls']=o[0].detach(); captured['boxes']=o[1].detach(); captured['text']=i[2].detach(); captured['mask']=i[3].detach()
    h1=detector.decoder.register_forward_hook(dh); h2=detector.bbox_head.register_forward_hook(hh)
    image=str(Path('/data1/LWR/vranlee/SERVER_ONLY/avis/KITTI_tracking/training/image_02')/str(unit['video'])/f"{int(unit['frame_id']):06d}.png")
    with torch.inference_mode(): native=inference_detector(detector,image,text_prompt=unit['sentence'],custom_entities=True)
    h1.remove(); h2.remove()
    hidden=captured['hidden'][-1,0]; boxes_norm=captured['boxes'][-1,0]
    img_shape=list(native.metainfo['img_shape']); scale=list(native.metainfo['scale_factor'])
    boxes=bbox_cxcywh_to_xyxy(boxes_norm).float(); boxes[:,0::2]*=img_shape[1]; boxes[:,1::2]*=img_shape[0]; boxes[:,0::2].clamp_(0,img_shape[1]); boxes[:,1::2].clamp_(0,img_shape[0]); boxes/=boxes.new_tensor(scale).repeat(2)
    lt=torch.maximum(cand[:,None,:2],boxes[None,:,:2]); rb=torch.minimum(cand[:,None,2:],boxes[None,:,2:]); inter=torch.prod(torch.clamp(rb-lt,min=0),-1); ca=torch.prod(torch.clamp(cand[:,2:]-cand[:,:2],min=0),-1)[:,None]; ba=torch.prod(torch.clamp(boxes[:,2:]-boxes[:,:2],min=0),-1)[None,:]; ov=inter/(ca+ba-inter).clamp_min(1e-9)
    w=[]
    ov_max=torch.max(ov,dim=-1).values
    for i in range(len(cand)): w.append(torch.full((900,),1/900,device='cuda:0') if float(ov_max[i].item())<=0 else F.softmax(ov[i]/tau,dim=0))
    w=torch.stack(w); rep=w@hidden; tok=w@captured['cls'][-1,0].sigmoid()
    # inference_mode tensors cannot be saved by autograd when they enter the
    # trainable adapter. Clone at this streaming boundary; detector remains
    # frozen and no raw/dense feature cache is written.
    # candidate_index is not unique in the dual bank (legitimate cross-pool
    # rows can share it). Use the immutable bank row offset as the audit key;
    # candidate_index remains metadata and never enters the model.
    bank_key=str(Path(unit['bank_path']).resolve())
    row_key_tuples=[(str(unit['video']),int(unit['frame_id']),bank_key,begin+i) for i in range(len(cand))]
    if len(row_key_tuples)!=len(set(row_key_tuples)): raise AssertionError('duplicate immutable bank-row key')
    row_keys=[{'video':v,'frame_id':f,'bank_path':p,'row_offset':o} for v,f,p,o in row_key_tuples]
    return rep.float().clone(), captured['text'].float().clone(), captured['mask'].clone(), numeric.float().clone(), tok.float().clone(), {'candidate_count':len(cand),'candidate_keys':row_keys,'proposal_query_count':900,'entity_score_shape':list(tok.shape),'zero_overlap':int((ov_max<=0).sum()),'native_count':int(len(native.pred_instances)),'representation_finite':bool(torch.isfinite(rep).all()),'token_score_finite':bool(torch.isfinite(tok).all())}


class BankStore:
    def __init__(self): self.cache=OrderedDict()
    def get(self,path):
        path=str(path)
        if path not in self.cache:
            self.cache[path]=torch.load(path,map_location='cpu')
            while len(self.cache)>1: self.cache.popitem(last=False)
        return self.cache[path]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out-root',required=True); ap.add_argument('--steps',type=int,default=100); ap.add_argument('--seed',type=int,default=SEED); ap.add_argument('--unit-keys',default=None); a=ap.parse_args()
    if Path.cwd().resolve()!=ROOT: raise RuntimeError('wrong LocateMOT cwd')
    out=Path(a.out_root); out=out if out.is_absolute() else ROOT/out; out=out.resolve()
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    random.seed(a.seed); torch.manual_seed(a.seed)
    units=load_units()
    if a.unit_keys:
        by_key={x['unit_key']:x for x in units}; wanted=[x for x in a.unit_keys.split(',') if x]
        missing=[x for x in wanted if x not in by_key]
        if missing: raise KeyError('unknown fit unit keys: '+str(missing))
        units=[by_key[x] for x in wanted]
    else: units=deterministic_unit_order(units,a.seed)
    if not units: raise RuntimeError('no fit units')
    detector,cfg,load=build_detector()
    if any(p.requires_grad for p in detector.parameters()): raise AssertionError('detector parameter is trainable')
    adapter=L57DecoderRepresentationScorer().to('cuda:0'); opt=torch.optim.AdamW(adapter.parameters(),lr=2e-4,weight_decay=1e-4)
    store=BankStore(); trace=[]; counts=Counter(); finite_steps=0; nonzero_steps=0; max_resident=0; start=time.time()
    for step in range(1,a.steps+1):
        unit=units[(step-1)%len(units)]; bank=store.get(unit['bank_path']); begin,end=int(unit['begin']),int(unit['end'])
        if int(unit['candidate_count'])!=end-begin: raise AssertionError(f'candidate_count mismatch: {unit["unit_key"]}')
        pos_idx=torch.as_tensor(unit['positive_indices'],dtype=torch.long,device='cuda:0')
        if len(pos_idx) and (int(pos_idx.min())<0 or int(pos_idx.max())>=int(unit['candidate_count'])): raise AssertionError(f'positive index out of range: {unit["unit_key"]}')
        rep,text,mask,numeric,entity_scores,meta=stream_rep(detector,unit,bank)
        expected_keys=[{'video':str(unit['video']),'frame_id':int(unit['frame_id']),'bank_path':str(Path(unit['bank_path']).resolve()),'row_offset':begin+i} for i in range(end-begin)]
        if meta['candidate_keys']!=expected_keys: raise AssertionError(f'candidate key/order drift: {unit["unit_key"]}')
        target=torch.zeros(int(unit['candidate_count']),dtype=torch.bool,device='cuda:0'); target[pos_idx]=True
        outp=adapter(rep,text,mask,numeric,entity_scores); loss,parts=unit_loss(outp,target)
        if not torch.isfinite(loss): raise RuntimeError(f'nonfinite loss at step {step}')
        opt.zero_grad(set_to_none=True); loss.backward(); grad=torch.nn.utils.clip_grad_norm_(adapter.parameters(),5.0); opt.step()
        finite_steps+=1; nonzero_steps+=int(float(grad)>0 and torch.isfinite(grad)); counts.update({'units':1,'candidate_rows':meta['candidate_count'],'positive':parts['positive_count'],'negative':parts['negative_count'],'hard_negative':parts['hard_negative_count'],'multi_positive':int(unit['category']=='multi_positive'),'inactive':int(unit['category']=='inactive'),'present_uncovered':int(unit['category']=='present_uncovered'),'domain_'+unit['dataset']:1,'video_'+str(unit['video']):1})
        max_resident=max(max_resident,len(store.cache));
        parts.update({'step':step,'unit_key':unit['unit_key'],'dataset':unit['dataset'],'video':unit['video'],'category':unit['category'],'gradient_norm':float(grad),'entity_score_consumed':True,**meta}); trace.append(parts)
        del rep,text,mask,numeric,entity_scores,outp,loss,bank; gc.collect(); torch.cuda.empty_cache()
    elapsed=time.time()-start
    stage='fit-only-smoke' if a.steps==100 else 'fit-only-targeted-regression'
    checkpoint=out/f'checkpoint_l57_step{a.steps}.pt'; torch.save({'format':'locatemot-l57-decoder-representation-scorer-v1','stage':stage,'step':a.steps,'seed':a.seed,'model':adapter.state_dict(),'config':{'hidden':128,'tau':.10,'image_dim':256,'text_dim':256,'entity_dim':256,'numeric_dim':24,'entity_score_projection':'LayerNorm(256)->Linear(256,128)','numeric_features':['geometry7','motion8','lifecycle8','objectness1'],'full_candidate_set':True,'null_head':True},'detector_frozen':True,'labels_split':'fit_only'},checkpoint)
    reload_model=L57DecoderRepresentationScorer().to('cuda:0'); reload_model.load_state_dict(torch.load(checkpoint,map_location='cuda:0')['model'],strict=True); reload_ok=True
    metrics_name=f'metrics_l57_step{a.steps}.json'
    detector_info={'interpreter':str(PYTHON),'config':str(CONFIG),'config_sha256':sha256(CONFIG),'weight':str(WEIGHT),'weight_sha256':sha256(WEIGHT),'bert':str(BERT),'load_missing_keys':load.get('missing_keys',[]) if isinstance(load,dict) else [],'load_unexpected_keys':load.get('unexpected_keys',[]) if isinstance(load,dict) else []}
    payload={'format':'locatemot-l57-decoder-representation-metrics-v2','status':'complete','stage':stage,'project_root':str(ROOT),'cwd':str(Path.cwd().resolve()),'seed':a.seed,'steps':a.steps,'finite_steps':finite_steps,'nonzero_gradient_steps':nonzero_steps,'checkpoint':str(checkpoint),'checkpoint_reload':reload_ok,'train_split':'fit','unit_count':len(units),'sampling_counts':dict(counts),'sampling_categories_present':sorted({x.get('category') for x in units}),'sampling_domains_present':sorted({x.get('dataset') for x in units}),'same_class_hard_negative_metadata':'unavailable; all-negative fallback','candidate_truncation':False,'candidate_key_drift':0,'persistent_raw_dense_cache_written':False,'screening_gt_used':False,'official_test_labels_read':False,'ordinary_mot_ovmot_touched':False,'detector_frozen':all(not p.requires_grad for p in detector.parameters()),'adapter_parameter_count':sum(p.numel() for p in adapter.parameters()),'entity_score_projection_parameter_count':sum(p.numel() for p in adapter.entity_score_proj.parameters()),'entity_score_contract':'pooled decoder entity/token scores [N,256] consumed by entity_score_proj; text memory remains separate','detector':detector_info,'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':'0','max_bank_cache_entries':max_resident,'elapsed_sec':elapsed,'steps_per_sec':a.steps/max(elapsed,1e-9)},'loss_trace':trace}
    (out/metrics_name).write_text(json.dumps(payload,indent=2,default=str)+'\n'); (out/'config.json').write_text(json.dumps({'seed':a.seed,'steps':a.steps,'stage':stage,'tau':.10,'train_split':'fit','adapter_parameter_count':payload['adapter_parameter_count'],'entity_score_projection_parameter_count':payload['entity_score_projection_parameter_count'],'detector':detector_info},indent=2,default=str)+'\n'); (out/'sampling_trace.json').write_text(json.dumps({'counts':dict(counts),'unit_order':[x['unit_key'] for x in units[:min(100,len(units))]],'domains_present':sorted({x.get('dataset') for x in units}),'categories_present':sorted({x.get('category') for x in units})},indent=2)+'\n'); (out/'provenance.json').write_text(json.dumps({'cwd':str(ROOT),'seed':a.seed,'manifest_sha256':'06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa','train_units_sha256':sha256(UNITS),'fit_only':True,'same_class_hard_negative_metadata':'unavailable; all-negative fallback','screening_gt_used':False,'official_test_labels_read':False,'ordinary_mot_ovmot_touched':False,'persistent_raw_dense_cache_written':False,'entity_score_consumed':True},indent=2)+'\n')
    print(json.dumps({'status':'complete','metrics':str(out/metrics_name),'checkpoint':str(checkpoint),'finite_steps':finite_steps,'nonzero_gradient_steps':nonzero_steps,'elapsed_sec':elapsed}))


if __name__=='__main__': main()
