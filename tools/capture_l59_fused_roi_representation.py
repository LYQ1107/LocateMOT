#!/usr/bin/env python3
"""L59-A: label-free fused GroundingDINO encoder-memory ROI audit."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess
from pathlib import Path

ROOT=Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
MMDET=Path("/data1/LWR/vranlee/LLM/mmdetection-3.3.0")
PYTHON=Path("/home/lwr/anaconda3/envs/masaenv_debug/bin/python")
CONFIG=MMDET/"configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.py"
WEIGHT=ROOT.parent/"TTAOD-F-main/download/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth"
BERT=Path("/home/lwr/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594")
UNITS=ROOT/"outputs/l49/data/train_units.jsonl"

def sha256(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1<<20), b""): h.update(block)
    return h.hexdigest()

CHILD=r'''
import json, os, sys, time, traceback
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet.registry import MODELS
import mmdet.models, mmdet.datasets
from mmdet.utils import register_all_modules
from mmdet.apis import inference_detector
register_all_modules(init_default_scope=True)

def normalize_memory(x):
    if x.dim()!=3: raise RuntimeError("unexpected visual rank "+str(list(x.shape)))
    if x.shape[0]==1: return x
    if x.shape[1]==1: return x.permute(1,0,2).contiguous()
    raise RuntimeError("ambiguous visual orientation "+str(list(x.shape)))

def roi_sample(memory, shapes, starts, invalid, boxes, img_hw, grid_size=4):
    _,_,dim=memory.shape
    Himg,Wimg=img_hw
    samples=[]; valid_counts=[]; level_shapes=[]
    frac=(torch.arange(grid_size,device=memory.device,dtype=torch.float32)+.5)/grid_size
    for level,(hh,ww) in enumerate(shapes.tolist()):
        hh,ww=int(hh),int(ww); start=int(starts[level])
        level_shapes.append([hh,ww])
        fmap=memory[0,start:start+hh*ww].reshape(hh,ww,dim).permute(2,0,1).unsqueeze(0)
        bad=invalid[0,start:start+hh*ww].reshape(1,1,hh,ww).float()
        # Build an [N, grid_y, grid_x] lattice explicitly.  The previous
        # four-dimensional broadcasting expression mixed N and grid axes and
        # fails as soon as N != grid_size.
        x1,x2=boxes[:,0],boxes[:,2]
        y1,y2=boxes[:,1],boxes[:,3]
        x=(x1[:,None]+(x2-x1)[:,None]*frac[None,:])[:,None,:].expand(-1,grid_size,-1)
        y=(y1[:,None]+(y2-y1)[:,None]*frac[None,:])[:,:,None].expand(-1,-1,grid_size)
        gx=2*((x/Wimg)*ww+0.5)/ww-1; gy=2*((y/Himg)*hh+0.5)/hh-1
        grid=torch.stack([gx,gy],-1)
        feat=F.grid_sample(fmap.expand(len(boxes),-1,-1,-1),grid,mode="bilinear",align_corners=False)
        badv=F.grid_sample(bad.expand(len(boxes),-1,-1,-1),grid,mode="nearest",align_corners=False)[:,0]
        samples.append(feat.permute(0,2,3,1).reshape(len(boxes),grid_size*grid_size,dim))
        valid_counts.append((badv<.5).sum(-1))
    return torch.cat(samples,1),torch.stack(valid_counts,1),level_shapes

job=json.load(open(os.environ["L59_JOB"]))
out={"status":"fail","python":sys.executable,"torch":torch.__version__,"cuda_available":bool(torch.cuda.is_available()),"job":job,
     "hook_source":{"encoder_file":"mmdet/models/layers/transformer/grounding_dino_layers.py","encoder_lines":"135-253","detector_forward_file":"mmdet/models/detectors/grounding_dino.py","detector_forward_lines":"317-340","contract":"GroundingDinoTransformerEncoder returns (visual_memory,memory_text); kwargs carry spatial_shapes,level_start_index,key_padding_mask"},
     "roi_contract":{"grid_per_level":"4x4 fixed bilinear feature sampling","feature_construction_only":True,"candidate_rows_retained":True,"persistent_feature_cache_written":False}}
try:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    cfg=Config.fromfile(os.environ["L59_CONFIG"])
    cfg.model.backbone.init_cfg=None
    cfg.model.language_model.name=os.environ["L59_BERT"]
    model=MODELS.build(cfg.model)
    load=load_checkpoint(model,os.environ["L59_WEIGHT"],map_location="cpu",strict=False)
    model.cfg=cfg; model.to("cuda:0").eval()
    for p in model.parameters(): p.requires_grad_(False)
    if any(p.requires_grad for p in model.parameters()): raise RuntimeError("detector parameter requires_grad")
    captured={}
    def encoder_hook(module,args,kwargs,result):
        if not isinstance(result,(tuple,list)) or len(result)<2:
            raise RuntimeError("encoder result is not (visual,memory_text)")
        captured["encoder_return_visual_is_none"]=result[0] is None
        if result[0] is not None:
            captured["encoder_visual"]=result[0].detach()
        captured["encoder_memory_text_is_none"]=result[1] is None
        if result[1] is not None:
            captured["encoder_memory_text"]=result[1].detach()
        for name in ("spatial_shapes","level_start_index","key_padding_mask","text_attention_mask"):
            if name not in kwargs: raise RuntimeError("missing encoder kwarg "+name)
            value=kwargs[name]
            # A None key-padding mask is the valid no-padding contract for
            # this single-image inference.  Spatial metadata must still be
            # tensor-valued; do not fabricate those indices.
            if value is None:
                if name != "key_padding_mask":
                    raise RuntimeError("encoder kwarg is None: "+name)
                captured[name]=None
            else:
                captured[name]=value.detach()
    def final_fusion_hook(module,args,result):
        if not isinstance(result,(tuple,list)) or len(result)<2:
            raise RuntimeError("final fusion result is not (visual,memory_text)")
        if result[0] is None or result[1] is None:
            raise RuntimeError("final fusion returned None visual/text")
        captured["fused_visual"]=result[0].detach()
        captured["fused_memory_text"]=result[1].detach()
    handle=model.encoder.register_forward_hook(encoder_hook,with_kwargs=True)
    fusion_handle=model.encoder.fusion_layers[-1].register_forward_hook(final_fusion_hook)
    t=time.time()
    with torch.inference_mode():
        native=inference_detector(model,job["image_path"],text_prompt=job["expression"],custom_entities=True)
    elapsed=time.time()-t
    handle.remove()
    fusion_handle.remove()
    for name in ("fused_visual","fused_memory_text","spatial_shapes","level_start_index","key_padding_mask","text_attention_mask"):
        if name not in captured: raise RuntimeError("hook did not capture "+name)
    visual_raw=captured["fused_visual"]; visual=normalize_memory(visual_raw)
    text=captured["fused_memory_text"]; shapes=captured["spatial_shapes"]; starts=captured["level_start_index"]; invalid=captured["key_padding_mask"]; text_attention_mask=captured["text_attention_mask"]
    if shapes.dim()==3: shapes=shapes[0]
    if starts.dim()>1: starts=starts[0]
    if shapes.dim()!=2 or shapes.shape[-1]!=2: raise RuntimeError("bad spatial_shapes "+str(list(shapes.shape)))
    total=int((shapes[:,0]*shapes[:,1]).sum())
    if visual.shape[1]!=total or int(starts.numel())!=int(shapes.shape[0]):
        raise RuntimeError("memory/shape index mismatch")
    if text_attention_mask is None:
        text_attention_mask=torch.zeros((1,text.shape[1]),device=text.device,dtype=torch.bool)
        text_mask_was_none=True
    else:
        text_mask_was_none=False
    if invalid is None:
        invalid=torch.zeros((1,total),device=visual.device,dtype=torch.bool)
        invalid_was_none=True
    else:
        invalid_was_none=False
        if invalid.dim()==2 and invalid.shape[0]!=1: invalid=invalid[:1]
    img_hw=(int(native.metainfo["img_shape"][0]),int(native.metainfo["img_shape"][1]))
    scale_raw=torch.as_tensor(native.metainfo["scale_factor"],device="cuda:0",dtype=torch.float32).flatten()
    if scale_raw.numel()==2:
        scale=torch.stack((scale_raw[0],scale_raw[1],scale_raw[0],scale_raw[1]))
    elif scale_raw.numel()==4:
        scale=scale_raw
    else:
        raise RuntimeError("unexpected scale_factor shape "+str(list(scale_raw.shape)))
    boxes=torch.tensor(job["candidate_boxes"],device="cuda:0",dtype=torch.float32)
    if scale.numel() == 2:
        scale_xyxy=scale.repeat(2)
    elif scale.numel() == 4:
        scale_xyxy=scale
    else:
        raise RuntimeError("unexpected scale_factor shape "+str(list(scale.shape)))
    boxes_resized=boxes*scale_xyxy
    if not bool(torch.isfinite(visual).all() and torch.isfinite(text).all() and torch.isfinite(boxes_resized).all()):
        raise RuntimeError("nonfinite fused representation")
    tokens,valid_counts,level_shapes=roi_sample(visual,shapes,starts,invalid,boxes_resized,img_hw)
    out.update({"status":"pass",
      "construction":{"checkpoint_loaded":True,"missing_keys":load.get("missing_keys",[]) if isinstance(load,dict) else [],"unexpected_keys":load.get("unexpected_keys",[]) if isinstance(load,dict) else [],"expected_warning":"language_model...position_ids may be reported"},
      "runtime":{"forward_sec":elapsed,"gpu_peak_bytes":int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,"detector_frozen":all(not p.requires_grad for p in model.parameters())},
      "encoder":{"raw_visual_shape":list(visual_raw.shape),"normalized_visual_shape":list(visual.shape),"memory_text_shape":list(text.shape),"text_attention_mask_shape":list(text_attention_mask.shape),"text_attention_mask_was_none":text_mask_was_none,"spatial_shapes":shapes.cpu().tolist(),"level_start_index":starts.cpu().tolist(),"invalid_mask_shape":list(invalid.shape),"key_padding_mask_was_none":invalid_was_none,"visual_finite":bool(torch.isfinite(visual).all()),"text_finite":bool(torch.isfinite(text).all()),"unmasked_visual_count":int(torch.isfinite(visual).sum()),"invalid_token_count":int(invalid.sum()),"total_visual_tokens":total,"encoder_returned_visual_none":bool(captured.get("encoder_return_visual_is_none",False)),"encoder_returned_memory_text_none":bool(captured.get("encoder_memory_text_none",False)),"visual_source":"encoder.fusion_layers[-1] output after final SingleScaleBiAttentionBlock; no decoder queries","text_source":"encoder.fusion_layers[-1] output after final SingleScaleBiAttentionBlock; no decoder queries"},
      "image":{"img_shape":list(native.metainfo["img_shape"]),"ori_shape":list(native.metainfo["ori_shape"]),"scale_factor":list(native.metainfo["scale_factor"]),"candidate_coordinate_conversion":"original boxes * [sx,sy,sx,sy]"},
      "roi":{"tokens_shape":list(tokens.shape),"levels":len(level_shapes),"level_shapes":level_shapes,"valid_sample_count":int(valid_counts.sum()),"total_sample_count":int(valid_counts.numel()*16),"all_candidate_rows":len(boxes),"all_candidate_rows_finite":bool(torch.isfinite(tokens).all()),"coverage_fraction":float((valid_counts>0).float().mean()),"valid_samples_per_candidate":valid_counts.cpu().tolist()},
      "postprocessed_native_count":int(len(native.pred_instances))})
except Exception as e:
    out.update({"error_type":type(e).__name__,"error":str(e),"traceback":traceback.format_exc(limit=30)})
print(json.dumps(out,default=str))
'''

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out-root",required=True)
    ap.add_argument("--unit-key",default="refer_kitti_v1|0001|14|106")
    a=ap.parse_args()
    if Path.cwd().resolve()!=ROOT: raise RuntimeError("wrong LocateMOT cwd")
    out=Path(a.out_root); out=out if out.is_absolute() else ROOT/out; out=out.resolve()
    if out.exists(): raise FileExistsError(out)
    for p in (MMDET,PYTHON,CONFIG,WEIGHT,BERT,UNITS):
        if not p.exists(): raise FileNotFoundError(p)
    units=[json.loads(x) for x in UNITS.read_text().splitlines() if x.strip() and json.loads(x).get("split")=="fit"]
    unit=next((x for x in units if x.get("unit_key")==a.unit_key),None)
    if unit is None: raise KeyError(a.unit_key)
    image=str(Path("/data1/LWR/vranlee/SERVER_ONLY/avis/KITTI_tracking/training/image_02")/str(unit["video"])/f'{int(unit["frame_id"]):06d}.png')
    import torch
    b=torch.load(unit["bank_path"],map_location="cpu")
    job={"unit_key":unit["unit_key"],"dataset":unit["dataset"],"video":unit["video"],"query_id":unit["query_id"],"frame_id":unit["frame_id"],"expression":unit["sentence"],"image_path":image,"bank_path":unit["bank_path"],"begin":unit["begin"],"end":unit["end"],"candidate_count":unit["candidate_count"],"candidate_boxes":b["tensors"]["box"][int(unit["begin"]):int(unit["end"])].tolist()}
    del b
    out.mkdir(parents=True)
    (out/"job_no_labels.json").write_text(json.dumps(job,indent=2)+"\n")
    env=os.environ.copy()
    env.update({"PYTHONPATH":str(MMDET),"L59_JOB":str(out/"job_no_labels.json"),"L59_CONFIG":str(CONFIG),"L59_WEIGHT":str(WEIGHT),"L59_BERT":str(BERT),"CUDA_VISIBLE_DEVICES":"0","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","TRANSFORMERS_OFFLINE":"1","HF_HUB_OFFLINE":"1","HF_DATASETS_OFFLINE":"1"})
    run=subprocess.run([str(PYTHON),"-c",CHILD],cwd=str(MMDET),env=env,text=True,capture_output=True,timeout=900)
    lines=[x for x in run.stdout.splitlines() if x.strip()]
    child=json.loads(lines[-1]) if lines else {"status":"fail","error":"no child JSON","stderr":run.stderr[-6000:]}
    payload={"format":"locatemot-l59-fused-roi-audit-v1","status":child.get("status"),"project_root":str(ROOT),"cwd":str(Path.cwd().resolve()),"isolated_interpreter":str(PYTHON),"source_root":str(MMDET),"config":str(CONFIG),"config_sha256":sha256(CONFIG),"weight":str(WEIGHT),"weight_sha256":sha256(WEIGHT),"bert_snapshot":str(BERT),"manifest_sha256":"06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa","labels_read":False,"screening_gt_used":False,"official_test_labels_read":False,"ordinary_mot_ovmot_touched":False,"persistent_raw_dense_cache_written":False,"candidate_rows_retained":True,"isolated_returncode":run.returncode,"child":child,"stderr_tail":run.stderr[-6000:]}
    (out/"fused_roi_representation.json").write_text(json.dumps(payload,indent=2,default=str)+"\n")
    if payload["status"]!="pass":
        (out/"INCOMPLETE.md").write_text("L59-A fused encoder-memory capture failed; decoder representation was not substituted.\n"+json.dumps(child,indent=2,default=str)+"\n")
    print(json.dumps({"status":payload["status"],"output":str(out/"fused_roi_representation.json")}))
if __name__=="__main__": main()
