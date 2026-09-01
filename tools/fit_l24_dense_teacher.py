"""Fit and freeze the L23 C5 linear teacher using calibration only."""
from __future__ import annotations
import argparse, hashlib, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path('/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT'); sys.path.insert(0,str(ROOT))
from tools.train_l23_dense_correspondence import fixed_refs, arrays_for
from tools.train_rmot_candidate_scorer import load_bank, load_metadata, make_refs, auc, average_precision, scalar_stats
FIELDS=('dense_roi','dense_points','dense_context_1p5','dense_context_3','dense_prev_roi','geometry_v2','neighbor_v2','motion_v2','lifecycle_v2','objectness')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def feat(ref,bank,rows):
    a=arrays_for(ref,bank,rows); return np.concatenate([a['query'],a['dense_roi'],a['dense_points'].reshape(len(rows),-1),a['dense_context_1p5'],a['dense_context_3'],a['dense_prev_roi'],a['geometry'],bank['tensors']['neighbor_v2'][ref['begin']+rows].float().numpy(),a['motion'],bank['tensors']['lifecycle_v2'][ref['begin']+rows].float().numpy(),a['objectness']],axis=1).astype('float32')
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',default='outputs/l19/protocol/kitti_fast_eval_manifest.json'); ap.add_argument('--v3-root',default='outputs/l23/candidate_bank_v3'); ap.add_argument('--out-root',default='outputs/l24/teacher_c5'); ap.add_argument('--steps',type=int,default=100); ap.add_argument('--frames-per-split',type=int,default=6000); ap.add_argument('--seed',type=int,default=17); ap.add_argument('--device',default='cuda:0'); args=ap.parse_args()
    def p(x): x=Path(x); return x if x.is_absolute() else ROOT/x
    manifest,root,out=map(p,(args.manifest,args.v3_root,args.out_root));
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    data=json.loads(manifest.read_text()); queries=sorted(data['queries'],key=lambda x:int(x['query_index'])); meta=load_metadata(); vids=sorted({str(q['video']) for q in queries}); banks={v:load_bank(root/'kitti'/f'{v}.pt') for v in vids}; refs=make_refs(queries,meta,banks); cal=fixed_refs([r for r in refs if r['split']=='calibration'],args.frames_per_split,args.seed); val=fixed_refs([r for r in refs if r['split']=='screening'],args.frames_per_split,args.seed+1); device=torch.device(args.device); torch.manual_seed(args.seed); rng=random.Random(args.seed)
    sample=feat(cal[0],banks[cal[0]['video']],np.arange(cal[0]['end']-cal[0]['begin'])); model=nn.Linear(sample.shape[1],1,device=device); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4); loss=[]; grads=[]; start=time.time(); model.train()
    for _ in range(args.steps):
        chosen=[cal[rng.randrange(len(cal))] for _ in range(8)]; xs=[]; ys=[]; terms=[]
        for ref in chosen:
            n=ref['end']-ref['begin']; rows=np.arange(n); x=feat(ref,banks[ref['video']],rows); y=ref['positive'].astype('float32'); xs.append(x);ys.append(y)
        x=torch.as_tensor(np.concatenate(xs),device=device); y=torch.as_tensor(np.concatenate(ys),device=device); z=model(x).squeeze(1); off=0
        for ref in chosen:
            n=ref['end']-ref['begin']; terms.append(nn.functional.binary_cross_entropy_with_logits(z[off:off+n],y[off:off+n]));off+=n
        l=torch.stack(terms).mean(); opt.zero_grad(set_to_none=True);l.backward();g=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5));opt.step();loss.append(float(l));grads.append(g)
    def evaluate(rs):
        ss=[]; yy=[]; margins=[];top1=top5=pf=0
        model.eval()
        with torch.inference_mode():
            for ref in rs:
                n=ref['end']-ref['begin']; z=model(torch.as_tensor(feat(ref,banks[ref['video']],np.arange(n)),device=device)).squeeze(1).cpu().numpy(); y=ref['positive'].astype(bool);ss.append(z);yy.append(y);pos=np.flatnonzero(y)
                if len(pos):
                    pf+=1;o=np.argsort(-z);top1+=int(y[o[:1]].any());top5+=int(y[o[:5]].any()); neg=np.flatnonzero(~y);margins.append(float(z[pos].min()-z[neg].max()) if len(neg) else 0)
        s=np.concatenate(ss);y=np.concatenate(yy);return {'roc_auc':auc(s,y),'pr_auc':average_precision(s,y),'top1_frame_recall':top1/max(1,pf),'top5_frame_recall':top5/max(1,pf),'positive_model_hard_margin':scalar_stats(margins),'candidate_count':int(len(y)),'positive_count':int(y.sum())}
    w=model.weight.detach().cpu().reshape(-1); b=model.bias.detach().cpu(); ck=out/'teacher_c5_linear.pt'; torch.save({'weight':w,'bias':b,'input_dim':int(w.numel()),'manifest_sha256':sha(manifest),'v3_root':str(root),'fields':FIELDS},ck); report={'format':'locatemot-l24-frozen-teacher-v1','manifest':str(manifest),'manifest_sha256':sha(manifest),'v3_root':str(root),'calibration_only_fit':True,'screening_gt_used_for_selection':False,'fields':list(FIELDS),'input_dim':int(w.numel()),'steps':args.steps,'calibration_metrics':evaluate(cal),'screening_metrics':evaluate(val),'loss':scalar_stats(loss),'gradient_norm':scalar_stats(grads),'checkpoint':str(ck),'elapsed_sec':time.time()-start}; (out/'teacher_metrics.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__':main()
