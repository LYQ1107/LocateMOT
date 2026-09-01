"""Build L25's independent high-resolution token bank.

This is a frozen CLIP ViT-B/16 structure adaptation: one 224px forward per
raw frame yields a 14x14 projected patch map. Candidate ROI/context token sets
are sampled only from candidate boxes; GT is copied only as an alignment
sidecar and never used to choose a sample.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path('/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT');sys.path.insert(0,str(ROOT))
from tools.build_l23_candidate_bank_v3 import (MEAN,STD,box_clip,fixed_point_set,grid_sample_points,region_points,sha256_file,check_old_v2_alignment,dense_clip_map)
from tools.train_rmot_candidate_scorer import load_bank

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',default='outputs/l19/protocol/kitti_fast_eval_manifest.json');ap.add_argument('--old-bank-root',default='outputs/l19/dual_banks_features');ap.add_argument('--v2-bank-root',default='outputs/l22/candidate_bank_v2');ap.add_argument('--v3-root',default='outputs/l23/candidate_bank_v3');ap.add_argument('--raw-root',default='data/kitti_tracking_training/image_02');ap.add_argument('--weights',default='/home/lwr/.cache/clip/ViT-B-16.pt');ap.add_argument('--out-root',default='outputs/l25/candidate_bank_v4');ap.add_argument('--device',default='cuda:0');ap.add_argument('--map-dtype',choices=('float16','float32'),default='float16');args=ap.parse_args()
    def p(x):x=Path(x);return x if x.is_absolute() else ROOT/x
    manifest,old_root,v2_root,v3_root,raw_root,weights,out=map(p,(args.manifest,args.old_bank_root,args.v2_bank_root,args.v3_root,args.raw_root,args.weights,args.out_root))
    if out.exists():raise FileExistsError(f'refusing to overwrite {out}')
    out.mkdir(parents=True);(out/'kitti').mkdir();(out/'dense_maps').mkdir();data=json.loads(manifest.read_text());qs=data['queries'];
    if len(qs)!=160 or sum(q['split']=='calibration' for q in qs)!=64 or sum(q['split']=='screening' for q in qs)!=96:raise ValueError('fixed manifest must be 160=64+96')
    vids=sorted({str(q['video']) for q in qs});device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    import clip
    if not weights.exists():raise FileNotFoundError(weights)
    model,_=clip.load(str(weights),device=device);model.eval();dtype=torch.float16 if args.map_dtype=='float16' else torch.float32
    summary={'format':'locatemot-l25-candidate-bank-v4-build-v1','stage':'L25','manifest':str(manifest),'manifest_sha256':sha256_file(manifest),'query_count':160,'calibration_queries':64,'screening_queries':96,'old_bank_root':str(old_root),'v2_bank_root':str(v2_root),'v3_root':str(v3_root),'raw_root':str(raw_root),'weights':str(weights),'weights_sha256':sha256_file(weights),'device':str(device),'backbone':'frozen OpenAI CLIP ViT-B/16 projected patch tokens','dense_map_shape':[512,14,14],'dense_map_stride_input_pixels':16,'input_policy':'fixed full-frame resize to 224x224','gt_used_for_features':False,'tracker_modified':False,'official_flexhook_reproduction':False,'structure_note':'high-resolution CLIP patch-token and coordinate-sampling adaptation; ROPE-Swin unavailable/not substituted','candidate_features':{'roi_tokens':'3x3=9 bilinear patch tokens inside bbox','point_tokens':'center plus four 20%-inset points','context_1p5_tokens':'3x3=9 tokens in 1.5x bbox','context_3_tokens':'3x3=9 tokens in 3x bbox','previous_roi_tokens':'causal previous same pool/track namespace, 9 tokens','coordinates':'normalized x/y for five points and ROI/context sample points'},'banks':{}}
    try:
      for video in vids:
        start=time.time();v3=load_bank(v3_root/'kitti'/f'{video}.pt');v2=torch.load(v2_root/'kitti'/f'{video}.pt',map_location='cpu',weights_only=False);old=torch.load(old_root/'kitti'/f'{video}.pt',map_location='cpu',weights_only=False);v3t=v3['tensors'];
        oldlab=json.loads((old_root/'kitti'/f'{video}.labels.json').read_text())['candidate_gt'];v2lab=json.loads((v2_root/'kitti'/f'{video}.labels.json').read_text())['candidate_gt'];v3lab=json.loads((v3_root/'kitti'/f'{video}.labels.json').read_text())['candidate_gt'];check_old_v2_alignment(old,v2,oldlab,v2lab,video)
        if oldlab!=v3lab:raise AssertionError(f'v2/v3 labels mismatch {video}')
        frames=v3t['frame_ids'].numpy().astype(np.int32);ptr=v3t['frame_ptr'].numpy().astype(np.int64);boxes=v3t['box'].numpy().astype(np.float32);tracks=v3t['track_id'].numpy().astype(np.int64);pools=v3t['pool_id'].numpy().astype(np.int64);maps={};map_meta=[]
        for frame in frames.tolist():
          image_path=raw_root/video/f'{int(frame):06d}.png';image=cv2.imread(str(image_path),cv2.IMREAD_COLOR)
          if image is None:raise FileNotFoundError(image_path)
          fmap=dense_clip_map(model,image,device).cpu().to(dtype).contiguous();
          if not bool(torch.isfinite(fmap.float()).all()):raise FloatingPointError(f'nonfinite map {video}/{frame}')
          mp=out/'dense_maps'/f'{video}_{int(frame):06d}.pt';torch.save({'video':video,'frame_id':int(frame),'feature_map':fmap,'shape':list(fmap.shape),'stride_input_pixels':16,'normalization':'CLIP mean/std; projected patch tokens L2-normalized','raw_image':str(image_path),'raw_image_sha256':sha256_file(image_path)},str(mp)+'.tmp');os.replace(str(mp)+'.tmp',mp);maps[int(frame)]=fmap;map_meta.append({'frame_id':int(frame),'path':str(mp),'raw_image':str(image_path),'raw_image_sha256':sha256_file(image_path),'shape':list(fmap.shape)})
        fields={k:[] for k in ('dense_roi_tokens_v4','dense_points_v4','dense_context_1p5_tokens_v4','dense_context_3_tokens_v4','dense_prev_roi_tokens_v4','dense_roi_v4','dense_context_1p5_v4','dense_context_3_v4','dense_prev_roi_v4','candidate_points_v4','roi_sample_points_v4','context_1p5_sample_points_v4','context_3_sample_points_v4')};prev={}
        for fi,frame in enumerate(frames.tolist()):
          begin,end=int(ptr[fi]),int(ptr[fi+1]);image=cv2.imread(str(raw_root/video/f'{int(frame):06d}.png'),cv2.IMREAD_COLOR);h,w=image.shape[:2];fmap=maps[int(frame)].float().to(device);cur=[]
          for row in range(begin,end):
            box=boxes[row];rp=region_points(box,w,h,1.0);c15p=region_points(box,w,h,1.5);c3p=region_points(box,w,h,3.0);pp=fixed_point_set(box,w,h);roi=grid_sample_points(fmap,rp);c15=grid_sample_points(fmap,c15p);c3=grid_sample_points(fmap,c3p);points=grid_sample_points(fmap,pp);namespace=(int(pools[row]),int(tracks[row]));previous=prev.get(namespace,np.zeros((9,512),np.float32));
            fields['dense_roi_tokens_v4'].append(roi.astype(np.float32));fields['dense_points_v4'].append(points.astype(np.float32));fields['dense_context_1p5_tokens_v4'].append(c15.astype(np.float32));fields['dense_context_3_tokens_v4'].append(c3.astype(np.float32));fields['dense_prev_roi_tokens_v4'].append(previous.astype(np.float32));fields['dense_roi_v4'].append(roi.mean(0).astype(np.float32));fields['dense_context_1p5_v4'].append(c15.mean(0).astype(np.float32));fields['dense_context_3_v4'].append(c3.mean(0).astype(np.float32));fields['dense_prev_roi_v4'].append(previous.mean(0).astype(np.float32));fields['candidate_points_v4'].append(pp.astype(np.float32));fields['roi_sample_points_v4'].append(rp.astype(np.float32));fields['context_1p5_sample_points_v4'].append(c15p.astype(np.float32));fields['context_3_sample_points_v4'].append(c3p.astype(np.float32));cur.append((namespace,roi.astype(np.float32)))
          for namespace,roi in cur:prev[namespace]=roi
        tensors={k:v.clone() for k,v in v3t.items()};
        for k,vals in fields.items():tensors[k]=torch.from_numpy(np.asarray(vals,dtype=np.float16 if args.map_dtype=='float16' and 'points_v4' not in k else np.float32))
        tensors['dense_map_frame_index']=torch.repeat_interleave(torch.arange(len(frames),dtype=torch.int64),torch.diff(v3t['frame_ptr']))
        for k,val in tensors.items():
          if torch.is_floating_point(val) and not bool(torch.isfinite(val.float()).all()):raise FloatingPointError(f'nonfinite candidate field {video}/{k}')
        if not torch.equal(tensors['frame_ptr'],v3t['frame_ptr']) or not torch.equal(tensors['frame_ids'],v3t['frame_ids']):raise AssertionError(f'v3 alignment changed {video}')
        meta={**v3.get('metadata',{}),'format':'locatemot-l25-candidate-bank-v4','stage':'L25','v3_bank_sha256':sha256_file(v3_root/'kitti'/f'{video}.pt'),'v3_labels_sha256':sha256_file(v3_root/'kitti'/f'{video}.labels.json'),'manifest_sha256':summary['manifest_sha256'],'weights':str(weights),'weights_sha256':summary['weights_sha256'],'visual_backbone':'frozen OpenAI CLIP ViT-B/16 projected patch tokens','dense_backbone':'frozen OpenAI CLIP ViT-B/16 projected visual patch tokens','dense_map_shape':[512,14,14],'dense_map_stride_input_pixels':16,'dense_map_dtype':args.map_dtype,'dense_map_files':map_meta,'gt_used_for_features':False,'row_order_preserved':True,'v2_v3_alignment_verified':True,'new_feature_dims':{k:list(tensors[k].shape[1:]) for k in fields}}
        output=out/'kitti'/f'{video}.pt';torch.save({'metadata':meta,'tensors':tensors,'row_keys':v3['row_keys']},str(output)+'.tmp');os.replace(str(output)+'.tmp',output);output.with_suffix('.labels.json').write_text(json.dumps({'candidate_gt':v3lab},separators=(',',':'))+'\n');output.with_suffix('.audit.json').write_text(json.dumps(meta,indent=2)+'\n');output.with_suffix('.complete').write_text('ok\n');summary['banks'][video]={'path':str(output),'rows':len(tracks),'frames':len(frames),'dense_map_files':len(map_meta),'elapsed_sec':time.time()-start}
      (out/'build_summary.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'BUILD_COMPLETE').write_text('ok\n');print(json.dumps({'out_root':str(out),'banks':summary['banks'],'dense_map_shape':[512,14,14]},indent=2))
    except Exception as exc:
      (out/'INCOMPLETE.md').write_text(f'# INCOMPLETE\n\nL25 v4 build stopped at first error: `{type(exc).__name__}: {exc}`\n');raise
if __name__=='__main__':main()
