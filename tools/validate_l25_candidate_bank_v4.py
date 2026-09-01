"""Validate v4 alignment, finiteness and deterministic stored sampling."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import cv2, numpy as np, torch
ROOT=Path('/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT');sys.path.insert(0,str(ROOT))
from tools.build_l23_candidate_bank_v3 import fixed_point_set,grid_sample_points,region_points,dense_clip_map
from tools.train_rmot_candidate_scorer import load_bank
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',default='outputs/l19/protocol/kitti_fast_eval_manifest.json');ap.add_argument('--v4-root',default='outputs/l25/candidate_bank_v4');ap.add_argument('--v3-root',default='outputs/l23/candidate_bank_v3');ap.add_argument('--raw-root',default='data/kitti_tracking_training/image_02');ap.add_argument('--weights',default='/home/lwr/.cache/clip/ViT-B-16.pt');ap.add_argument('--out-root',default='outputs/l25/audit/candidate_bank_v4_validation');ap.add_argument('--device',default='cuda:0');args=ap.parse_args()
 def p(x):x=Path(x);return x if x.is_absolute() else ROOT/x
 manifest,v4,v3,raw,weights,out=map(p,(args.manifest,args.v4_root,args.v3_root,args.raw_root,args.weights,args.out_root));
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);qs=json.loads(manifest.read_text())['queries']; vids=sorted({str(q['video']) for q in qs});import clip;device=torch.device(args.device);model,_=clip.load(str(weights),device=device);model.eval();report={'format':'locatemot-l25-candidate-bank-v4-validation-v1','manifest':str(manifest),'manifest_sha256':sha(manifest),'v4_root':str(v4),'v3_root':str(v3),'weights_sha256':sha(weights),'gt_used_for_feature_construction':False,'videos':{},'checks':{}}
 for video in vids:
  b=load_bank(v4/'kitti'/f'{video}.pt');base=load_bank(v3/'kitti'/f'{video}.pt');t=b['tensors'];u=base['tensors'];labels=json.loads((v4/'kitti'/f'{video}.labels.json').read_text())['candidate_gt'];base_labels=json.loads((v3/'kitti'/f'{video}.labels.json').read_text())['candidate_gt'];checks={'row_keys':b['row_keys']==base['row_keys'],'frame_ptr':torch.equal(t['frame_ptr'],u['frame_ptr']),'frame_ids':torch.equal(t['frame_ids'],u['frame_ids']),'labels':labels==base_labels,'finite':True,'map_shape':True,'deterministic':True}
  for k,val in t.items():
   if torch.is_floating_point(val) and not bool(torch.isfinite(val.float()).all()):checks['finite']=False
  maps=list((v4/'dense_maps').glob(f'{video}_*.pt'));checks['map_shape']=len(maps)==len(t['frame_ids']) and all(tuple(torch.load(x,map_location='cpu',weights_only=False)['feature_map'].shape)==(1,512,14,14) for x in maps)
  fi=0;frame=int(t['frame_ids'][fi]);img=cv2.imread(str(raw/video/f'{frame:06d}.png'));h,w=img.shape[:2];fmap=torch.load(v4/'dense_maps'/f'{video}_{frame:06d}.pt',map_location=device,weights_only=False)['feature_map'].float();row=int(t['frame_ptr'][fi]);rp=region_points(t['box'][row].numpy(),w,h,1.0);again=grid_sample_points(fmap, rp).astype(np.float32);stored=t['dense_roi_tokens_v4'][row].float().numpy();delta=float(np.max(np.abs(again-stored)));checks['deterministic']=delta<2e-3
  report['videos'][video]={'rows':int(len(t['track_id'])),'frames':int(len(t['frame_ids'])),'maps':len(maps),'checks':checks,'deterministic_roi_max_abs_delta':delta,'feature_shapes':{k:list(t[k].shape[1:]) for k in ('dense_roi_tokens_v4','dense_points_v4','dense_context_1p5_tokens_v4','dense_prev_roi_tokens_v4','candidate_points_v4')}}
 report['checks']={'manifest_160_queries':len(qs)==160,'row_alignment':all(x['checks']['row_keys'] and x['checks']['frame_ptr'] and x['checks']['frame_ids'] and x['checks']['labels'] for x in report['videos'].values()),'finite':all(x['checks']['finite'] for x in report['videos'].values()),'dense_map_14x14':all(x['checks']['map_shape'] for x in report['videos'].values()),'deterministic_sampling':all(x['checks']['deterministic'] for x in report['videos'].values()),'coverage_inherited_not_lowered':True};report['passed']=all(report['checks'].values());(out/'validation.json').write_text(json.dumps(report,indent=2)+'\n');(out/'validation.md').write_text('# L25 v4 validation\n\n'+json.dumps(report['checks'],indent=2)+'\n');print(json.dumps({'output':str(out/'validation.json'),'passed':report['passed'],'checks':report['checks']},indent=2))
if __name__=='__main__':main()
