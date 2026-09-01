#!/usr/bin/env python3
"""L59 fit-only smoke: streaming post-fusion ROI/set scorer."""
from __future__ import annotations
import argparse, gc, hashlib, json, os, random, time
from collections import Counter, OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT=Path('/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT').resolve()
MMDET=Path('/data1/LWR/vranlee/LLM/mmdetection-3.3.0')
PYTHON=Path('/home/lwr/anaconda3/envs/masaenv_debug/bin/python')
CONFIG=MMDET/'configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.py'
WEIGHT=ROOT.parent/'TTAOD-F-main/download/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'
BERT=Path('/home/lwr/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594')
UNITS=ROOT/'outputs/l49/data/train_units.jsonl'
sys_path=str(ROOT)
import sys; sys.path.insert(0,sys_path)
from locatemot.models.l59_fused_roi_scorer import L59FusedROIScorer

def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1<<20), b''): h.update(block)
    return h.hexdigest()

def load_units():
    return [json.loads(x) for x in UNITS.read_text().splitlines() if x.strip()
            and json.loads(x).get('split')=='fit'
            and json.loads(x).get('dataset') in ('refer_kitti_v1','refer_kitti_v2')]

def ordered(units,seed):
    rng=random.Random(seed); buckets={}
    for u in units: buckets.setdefault((u['dataset'],u.get('category','unknown')),[]).append(u)
    for v in buckets.values(): rng.shuffle(v)
    keys=[(d,c) for d in ('refer_kitti_v1','refer_kitti_v2')
          for c in ('positive','multi_positive','inactive','present_uncovered')]
    out=[]
    while any(buckets.get(k) for k in keys):
        for k in keys:
            if buckets.get(k): out.append(buckets[k].pop(0))
    rest=[u for v in buckets.values() for u in v]; rng.shuffle(rest); return out+rest

def build_detector():
    from mmengine.config import Config
    from mmengine.runner import load_checkpoint
    from mmdet.registry import MODELS
    import mmdet.models, mmdet.datasets
    from mmdet.utils import register_all_modules
    register_all_modules(init_default_scope=True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    cfg=Config.fromfile(str(CONFIG)); cfg.model.backbone.init_cfg=None; cfg.model.language_model.name=str(BERT)
    model=MODELS.build(cfg.model); loaded=load_checkpoint(model,str(WEIGHT),map_location='cpu',strict=False)
    model.cfg=cfg; model.to('cuda:0').eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model,cfg,loaded

def roi_sample(memory,shapes,starts,invalid,boxes,img_hw,grid_size=4):
    _,_,dim=memory.shape; Himg,Wimg=img_hw; samples=[]
    frac=(torch.arange(grid_size,device=memory.device,dtype=torch.float32)+.5)/grid_size
    for level,(hh,ww) in enumerate(shapes.tolist()):
        hh,ww=int(hh),int(ww); start=int(starts[level])
        fmap=memory[0,start:start+hh*ww].reshape(hh,ww,dim).permute(2,0,1).unsqueeze(0)
        x1,x2=boxes[:,0],boxes[:,2]; y1,y2=boxes[:,1],boxes[:,3]
        x=(x1[:,None]+(x2-x1)[:,None]*frac[None,:])[:,None,:].expand(-1,grid_size,-1)
        y=(y1[:,None]+(y2-y1)[:,None]*frac[None,:])[:,:,None].expand(-1,-1,grid_size)
        gx=2*((x/Wimg)*ww+0.5)/ww-1; gy=2*((y/Himg)*hh+0.5)/hh-1
        grid=torch.stack([gx,gy],-1)
        feat=F.grid_sample(fmap.expand(len(boxes),-1,-1,-1),grid,mode='bilinear',align_corners=False)
        samples.append(feat.permute(0,2,3,1).reshape(len(boxes),grid_size*grid_size,dim))
    return torch.cat(samples,1)

def stream(detector,u,bank):
    from mmdet.apis import inference_detector
    b,e=int(u['begin']),int(u['end']); count=e-b
    if int(u['candidate_count'])!=count: raise AssertionError('candidate_count mismatch')
    boxes=bank['tensors']['box'][b:e].float().to('cuda:0')
    numeric=torch.cat([bank['tensors']['geometry'][b:e].float(),bank['tensors']['motion'][b:e].float(),bank['tensors']['lifecycle'][b:e].float(),bank['tensors']['objectness'][b:e].float().reshape(-1,1)],1).to('cuda:0')
    cap={}
    def enc_hook(m,args,kwargs,result):
        cap['visual_final']=result[0].detach()
        for k in ('spatial_shapes','level_start_index','key_padding_mask','text_attention_mask'):
            cap[k]=None if kwargs.get(k) is None else kwargs[k].detach()
    def fusion_hook(m,args,result):
        if not isinstance(result,(tuple,list)) or len(result)<2 or result[0] is None or result[1] is None: raise RuntimeError('invalid fused block output')
        cap['visual']=result[0].detach(); cap['text']=result[1].detach()
    h1=detector.encoder.register_forward_hook(enc_hook,with_kwargs=True); h2=detector.encoder.fusion_layers[-1].register_forward_hook(fusion_hook)
    image=str(ROOT.parent/'KITTI_tracking/training/image_02'/str(u['video'])/f'{int(u["frame_id"]):06d}.png')
    with torch.inference_mode(): native=inference_detector(detector,image,text_prompt=u['sentence'],custom_entities=True)
    h1.remove(); h2.remove()
    visual=cap['visual']; text=cap['text']; shapes=cap['spatial_shapes']; starts=cap['level_start_index']; invalid=cap['key_padding_mask']; tmask=cap['text_attention_mask']
    if visual.dim()!=3 or visual.shape[0]!=1: raise RuntimeError('unexpected fused visual shape')
    if shapes.dim()==3: shapes=shapes[0]
    if starts.dim()>1: starts=starts[0]
    total=int((shapes[:,0]*shapes[:,1]).sum())
    if invalid is None: invalid=torch.zeros((1,total),device=visual.device,dtype=torch.bool)
    if tmask is None: tmask=torch.zeros((1,text.shape[1]),device=text.device,dtype=torch.bool)
    sf=torch.as_tensor(native.metainfo['scale_factor'],device='cuda:0',dtype=torch.float32).flatten(); sf=sf.repeat(2) if sf.numel()==2 else sf
    if sf.numel()!=4: raise RuntimeError('bad scale factor')
    boxes_resized=boxes*sf
    if visual.shape[1]!=total or starts.numel()!=shapes.shape[0]: raise RuntimeError('memory metadata mismatch')
    roi=roi_sample(visual,shapes,starts,invalid,boxes_resized,(int(native.metainfo['img_shape'][0]),int(native.metainfo['img_shape'][1])))
    if not bool(torch.isfinite(roi).all() and torch.isfinite(text).all() and torch.isfinite(numeric).all()): raise RuntimeError('nonfinite streamed feature')
    keys=[(str(u['video']),int(u['frame_id']),str(Path(u['bank_path']).resolve()),b+i) for i in range(count)]
    if len(set(keys))!=count: raise AssertionError('immutable row key duplicate')
    return roi.clone().float(),text.clone().float(),(~tmask.bool()).clone(),numeric.clone().float(),{'candidate_count':count,'row_keys':keys,'native_count':len(native.pred_instances),'roi_shape':list(roi.shape),'fused_text_shape':list(text.shape)}

def loss_fn(out,target):
    s=out['relevance_logit']; z=s.new_zeros(()); pos=torch.where(target)[0]; neg=torch.where(~target)[0]
    hard=neg[torch.argsort(s.detach()[neg],descending=True)[:min(24,len(neg))]] if len(neg) else neg
    terms=[]
    if len(pos): terms.append(F.binary_cross_entropy_with_logits(s[pos],torch.ones_like(s[pos])))
    if len(neg): terms.append(F.binary_cross_entropy_with_logits(s[neg],torch.zeros_like(s[neg])))
    bce=torch.stack(terms).mean() if terms else z
    pair=F.softplus(.2+s[hard][None,:]-s[pos][:,None]).mean() if len(pos) and len(hard) else z
    listwise=torch.logsumexp(s,0)-torch.logsumexp(s[pos],0) if len(pos) else z
    minimum=F.softplus(.2+s[hard].max()-s[pos]).mean() if len(pos) and len(hard) else z
    null=F.binary_cross_entropy_with_logits(out['null_logit'],s.new_tensor(float(not len(pos))))
    cal=((s.sigmoid()-target.float())**2).mean()
    total=bce+pair+.5*listwise+.5*minimum+.2*null+.1*cal
    return total,{'total':float(total.detach()),'bce':float(bce.detach()),'pairwise':float(pair.detach()),'listwise':float(listwise.detach()),'min_positive':float(minimum.detach()),'null':float(null.detach()),'calibration':float(cal.detach()),'positive_count':len(pos),'negative_count':len(neg),'hard_negative_count':len(hard)}

class Store:
    def __init__(self): self.d=OrderedDict()
    def get(self,p):
        if p not in self.d: self.d[p]=torch.load(p,map_location='cpu')
        while len(self.d)>1: self.d.popitem(last=False)
        return self.d[p]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out-root',required=True); ap.add_argument('--steps',type=int,default=100); ap.add_argument('--seed',type=int,default=20260829); a=ap.parse_args()
    if Path.cwd().resolve()!=ROOT: raise RuntimeError('wrong cwd')
    out=Path(a.out_root); out=out if out.is_absolute() else ROOT/out; out=out.resolve()
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True); random.seed(a.seed); torch.manual_seed(a.seed)
    units=ordered(load_units(),a.seed)
    if not units: raise RuntimeError('no fit units')
    detector,cfg,loaded=build_detector()
    if any(p.requires_grad for p in detector.parameters()): raise AssertionError('detector not frozen')
    model=L59FusedROIScorer().cuda(); opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    store=Store(); trace=[]; counts=Counter(); finite=nonzero=0; start=time.time(); peak=0
    for step in range(1,a.steps+1):
        u=units[(step-1)%len(units)]; bank=store.get(u['bank_path']); roi,text,mask,num,meta=stream(detector,u,bank)
        target=torch.zeros(meta['candidate_count'],dtype=torch.bool,device='cuda:0'); pi=torch.as_tensor(u['positive_indices'],dtype=torch.long,device='cuda:0')
        if len(pi) and (int(pi.min())<0 or int(pi.max())>=meta['candidate_count']): raise AssertionError('positive index range')
        target[pi]=True; outp=model(roi,text,mask,num); loss,parts=loss_fn(outp,target)
        if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
        opt.zero_grad(set_to_none=True); loss.backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        finite+=1; nonzero+=int(float(grad)>0 and bool(torch.isfinite(grad))); counts.update({'units':1,'candidate_rows':meta['candidate_count'],'positive':parts['positive_count'],'inactive':int(u.get('category')=='inactive'),'multi_positive':int(u.get('category')=='multi_positive'),'present_uncovered':int(u.get('category')=='present_uncovered'),'domain_'+u['dataset']:1,'video_'+str(u['video']):1})
        parts.update({'step':step,'unit_key':u['unit_key'],'dataset':u['dataset'],'video':u['video'],'category':u.get('category'),'gradient_norm':float(grad),'candidate_key_drift':0,'candidate_truncation':False,'detector_frozen':True,**meta}); trace.append(parts); peak=max(peak,int(torch.cuda.max_memory_allocated())); del roi,text,mask,num,bank,outp,loss; gc.collect(); torch.cuda.empty_cache()
    elapsed=time.time()-start; ck=out/f'checkpoint_l59_step{a.steps}.pt'; torch.save({'format':'locatemot-l59-fused-roi-scorer-v1','step':a.steps,'seed':a.seed,'model':model.state_dict(),'config':{'hidden':128,'roi_tokens_per_candidate':64,'levels':4,'grid':'4x4','numeric_dim':24,'same_class_hard_negative_metadata':'unavailable; all-negative fallback'},'detector_frozen':True,'labels_split':'fit'},ck)
    reload=L59FusedROIScorer().cuda(); reload.load_state_dict(torch.load(ck,map_location='cuda:0')['model'],strict=True)
    info={'interpreter':str(PYTHON),'config':str(CONFIG),'config_sha256':sha256(CONFIG),'weight':str(WEIGHT),'weight_sha256':sha256(WEIGHT),'bert':str(BERT),'missing_keys':loaded.get('missing_keys',[]) if isinstance(loaded,dict) else [],'unexpected_keys':loaded.get('unexpected_keys',[]) if isinstance(loaded,dict) else []}
    payload={'format':'locatemot-l59-fused-roi-metrics-v1','status':'complete','stage':'fit-only-smoke','project_root':str(ROOT),'cwd':str(Path.cwd().resolve()),'seed':a.seed,'steps':a.steps,'finite_steps':finite,'nonzero_gradient_steps':nonzero,'checkpoint':str(ck),'checkpoint_reload':True,'train_split':'fit','unit_count':len(units),'sampling_counts':dict(counts),'domains_present':sorted({u['dataset'] for u in units}),'categories_seen':sorted({u.get('category') for u in units[:a.steps]}),'same_class_hard_negative_metadata':'unavailable; all-negative fallback','candidate_truncation':False,'candidate_key_drift':0,'persistent_raw_dense_cache_written':False,'screening_gt_used':False,'official_test_labels_read':False,'ordinary_mot_ovmot_touched':False,'detector_frozen':all(not p.requires_grad for p in detector.parameters()),'adapter_parameter_count':sum(p.numel() for p in model.parameters()),'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':'0','peak_memory_bytes':peak,'elapsed_sec':elapsed,'steps_per_sec':a.steps/max(elapsed,1e-9)},'detector':info,'loss_trace':trace}
    (out/f'metrics_l59_step{a.steps}.json').write_text(json.dumps(payload,indent=2,default=str)+'\n'); (out/'config.json').write_text(json.dumps(payload['detector']|{'seed':a.seed,'steps':a.steps,'stage':'fit-only-smoke','roi_contract':'post-fusion visual memory 4 levels x 4x4; complete candidate set'},indent=2)+'\n'); (out/'sampling_trace.json').write_text(json.dumps({'counts':dict(counts),'unit_order':[u['unit_key'] for u in units[:min(len(units),a.steps)]],'domains_present':payload['domains_present'],'categories_seen':payload['categories_seen']},indent=2)+'\n'); (out/'provenance.json').write_text(json.dumps({'cwd':str(ROOT),'seed':a.seed,'manifest_sha256':'06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa','train_units_sha256':sha256(UNITS),'fit_only':True,'detector_runtime':info,'screening_gt_used':False,'official_test_labels_read':False,'ordinary_mot_ovmot_touched':False,'persistent_raw_dense_cache_written':False,'candidate_rows_retained':True},indent=2)+'\n')
    print(json.dumps({'status':'complete','metrics':str(out/f'metrics_l59_step{a.steps}.json'),'checkpoint':str(ck),'finite_steps':finite,'nonzero_gradient_steps':nonzero,'elapsed_sec':elapsed}))
if __name__=='__main__': main()
