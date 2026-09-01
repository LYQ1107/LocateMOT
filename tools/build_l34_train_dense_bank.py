#!/usr/bin/env python3
"""Build an independent train-only CLIP ViT-B/16 region-token bank for L34."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT=Path('/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT'); sys.path.insert(0,str(ROOT))
from tools.build_l23_candidate_bank_v3 import dense_clip_map, fixed_point_set, grid_sample_points, region_points, sha256_file
from tools.train_rmot_candidate_scorer import load_bank

L19=ROOT/'outputs/l19/dual_banks_features/kitti'; SPLIT=ROOT/'outputs/l16/data/protocol/split_manifest.json'; RAW=ROOT/'data/kitti_tracking_training/image_02'; WEIGHTS=Path('/home/lwr/.cache/clip/ViT-B-16.pt')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out-root',required=True); ap.add_argument('--device',default='cuda:0'); args=ap.parse_args()
    out=Path(args.out_root); out=out if out.is_absolute() else ROOT/out
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True); (out/'kitti').mkdir(); (out/'dense_maps').mkdir()
    train=sorted(str(x) for x in json.loads(SPLIT.read_text())['kitti_v2']['train']); device=torch.device(args.device)
    if not WEIGHTS.exists(): raise FileNotFoundError(WEIGHTS)
    import clip
    model,_=clip.load(str(WEIGHTS),device=device); model.eval(); summary={'format':'locatemot-l34-train-dense-bank-v1','train_only':True,'videos':train,'split_manifest':str(SPLIT.resolve()),'split_manifest_sha256':sha256_file(SPLIT),'source_bank':str(L19.resolve()),'weights':str(WEIGHTS),'weights_sha256':sha256_file(WEIGHTS),'raw_root':str(RAW.resolve()),'dense_map_shape':[512,14,14],'stride_input_pixels':16,'gt_used_for_features':False,'tracker_modified':False,'banks':{}}
    try:
      for video in train:
        start=time.time(); src=load_bank(L19/f'{video}.pt'); t=src['tensors']; frames=t['frame_ids'].numpy().astype(np.int32); ptr=t['frame_ptr'].numpy().astype(np.int64); boxes=t['box'].numpy().astype(np.float32); tracks=t['track_id'].numpy().astype(np.int64); pools=t['pool_id'].numpy().astype(np.int64)
        maps={}; map_meta=[]
        for frame in frames.tolist():
          path=RAW/video/f'{int(frame):06d}.png'; image=cv2.imread(str(path),cv2.IMREAD_COLOR)
          if image is None: raise FileNotFoundError(path)
          fmap=dense_clip_map(model,image,device).cpu().float().contiguous()
          if not torch.isfinite(fmap).all(): raise FloatingPointError(f'nonfinite map {video}/{frame}')
          mp=out/'dense_maps'/f'{video}_{int(frame):06d}.pt'; torch.save({'video':video,'frame_id':int(frame),'feature_map':fmap,'shape':list(fmap.shape),'stride_input_pixels':16,'raw_image':str(path),'raw_image_sha256':sha256_file(path)},str(mp)+'.tmp'); os.replace(str(mp)+'.tmp',mp); maps[int(frame)]=fmap; map_meta.append({'frame_id':int(frame),'path':str(mp),'raw_image':str(path),'raw_image_sha256':sha256_file(path)})
        fields={k:[] for k in ('dense_roi_tokens_v4','dense_points_v4','dense_context_1p5_tokens_v4','dense_context_3_tokens_v4','dense_prev_roi_tokens_v4','candidate_points_v4','roi_sample_points_v4','context_1p5_sample_points_v4','context_3_sample_points_v4')}; previous={}
        for fi,frame in enumerate(frames.tolist()):
          begin,end=int(ptr[fi]),int(ptr[fi+1]); image=cv2.imread(str(RAW/video/f'{int(frame):06d}.png'),cv2.IMREAD_COLOR); h,w=image.shape[:2]; fmap=maps[int(frame)].to(device); current=[]
          for row in range(begin,end):
            box=boxes[row]; rp=region_points(box,w,h,1.0); c15=region_points(box,w,h,1.5); c3=region_points(box,w,h,3.0); pp=fixed_point_set(box,w,h); roi=grid_sample_points(fmap,rp); ctx15=grid_sample_points(fmap,c15); ctx3=grid_sample_points(fmap,c3); points=grid_sample_points(fmap,pp); ns=(int(pools[row]),int(tracks[row])); prev=previous.get(ns,np.zeros((9,512),np.float32))
            fields['dense_roi_tokens_v4'].append(roi); fields['dense_points_v4'].append(points); fields['dense_context_1p5_tokens_v4'].append(ctx15); fields['dense_context_3_tokens_v4'].append(ctx3); fields['dense_prev_roi_tokens_v4'].append(prev); fields['candidate_points_v4'].append(pp); fields['roi_sample_points_v4'].append(rp); fields['context_1p5_sample_points_v4'].append(c15); fields['context_3_sample_points_v4'].append(c3); current.append((ns,roi))
          for ns,roi in current: previous[ns]=roi
        keep=('frame','candidate_index','track_id','box','pool_id','frame_ptr','frame_ids','objectness','geometry','motion','lifecycle','clip','history_clip','uidm_h')
        tensors={k:t[k].clone() for k in keep};
        for k,v in fields.items(): tensors[k]=torch.from_numpy(np.asarray(v,np.float32))
        tensors['dense_map_frame_index']=torch.repeat_interleave(torch.arange(len(frames),dtype=torch.int64),torch.diff(t['frame_ptr']))
        if not all(torch.isfinite(v.float()).all().item() for k,v in tensors.items() if torch.is_floating_point(v)): raise FloatingPointError(f'nonfinite candidate field {video}')
        output=out/'kitti'/f'{video}.pt'; torch.save({'metadata':{'format':'locatemot-l34-train-dense-bank','video':video,'train_only':True,'weights_sha256':summary['weights_sha256'],'dense_map_shape':[512,14,14],'row_order_preserved':True,'gt_used_for_features':False,'dense_map_files':map_meta},'tensors':t},str(output)+'.source.tmp')
        # Preserve source and append only the newly sampled dense fields.
        torch.save({'metadata':{'format':'locatemot-l34-train-dense-bank','video':video,'train_only':True,'weights_sha256':summary['weights_sha256'],'dense_map_shape':[512,14,14],'row_order_preserved':True,'gt_used_for_features':False,'dense_map_files':map_meta},'tensors':tensors},str(output)+'.tmp'); os.replace(str(output)+'.tmp',output)
        labels=src['candidate_gt']; output.with_suffix('.labels.json').write_text(json.dumps({'candidate_gt':labels},separators=(',',':'))+'\n'); output.with_suffix('.complete').write_text('ok\n'); summary['banks'][video]={'path':str(output),'rows':len(t['frame']),'frames':len(frames),'dense_map_files':len(map_meta),'elapsed_sec':time.time()-start}
      (out/'build_summary.json').write_text(json.dumps(summary,indent=2)+'\n'); (out/'BUILD_COMPLETE').write_text('ok\n'); print(json.dumps(summary,indent=2),flush=True)
    except Exception as exc:
      (out/'INCOMPLETE.md').write_text(f'# INCOMPLETE\n\nStopped at first error: `{type(exc).__name__}: {exc}`\n'); raise

if __name__=='__main__': main()
