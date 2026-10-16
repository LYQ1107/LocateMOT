"""Fixed DT posthoc discrimination/calibration audit; never optimize thresholds."""
import json
from pathlib import Path
from collections import Counter
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def main():
    import numpy as np
    import torch
    from PIL import Image
    from locatemot.groundjev.compact import CompactCore
    from locatemot.groundjev.global_identity_core import GlobalIdentityCore
    from locatemot.groundjev.global_competition import sparse_competition
    from tools.groundjev.run_video import ReplayFrames
    from tools.groundjev.analyze_tracking_events import load_truth
    from tools.groundjev.audit_global_identity_labels import frame_support
    torch.set_num_threads(4)
    plan=json.loads((ROOT/'configs/groundjev/global_identity_v1.json').read_text());source=Path(plan['immutable_source_root']);out=ROOT/plan['output']
    job=json.loads((source/plan['evaluation_plan']).read_text())['scopes']['mot_full1203'];manifest=json.loads((source/job['manifest']).read_text());truth,_=load_truth(manifest)
    models={};counts={};hist={};sums={};streams={}
    for arm,run in [('A2',out/'controls/A2_mot_full1203'),('A4',out/'own_state_controls/A4_own_state_v2_mot_full1203')]:
        checkpoint=Path(plan['model_output'])/('A4_own_state_v2' if arm=='A4' else arm)/'checkpoint.pt'
        data=torch.load(checkpoint,map_location='cpu',weights_only=True);base_data=torch.load(data['initial_checkpoint'],map_location='cpu',weights_only=True)
        base=CompactCore(base_data['architecture'],semantic=base_data['semantic']);base.load_state_dict(base_data['state_dict'])
        model=GlobalIdentityCore(base,competition=True);model.load_state_dict(data['state_dict']);model.eval();models[arm]=model
        counts[arm]=Counter();hist[arm]=[np.zeros(800,dtype=np.int64),np.zeros(800,dtype=np.int64)];sums[arm]=[0.,0.];streams[arm]=(run/'frame_decisions.jsonl').open()
    replay=ReplayFrames(source/job['candidates'])
    try:
        for number,path in zip(manifest['frame_numbers'],manifest['images']):
            raw=replay.frame(number);candidates=raw['candidates'];_,_,support=frame_support(candidates,truth[number])
            gt=np.array([truth[number][s[0]]['identity'] if len(s)==1 else -1 for s in support]);boxes=np.asarray([c['box'] for c in candidates],np.float32).reshape(-1,4)
            features=np.asarray([c['native_region_feature'] for c in candidates],np.float32).reshape(-1,2560)
            edges,geometry=sparse_competition(boxes,features,32)
            with Image.open(path) as image:w,h=image.size
            known=(gt[edges[:,0]]>=0)&(gt[edges[:,1]]>=0) if len(edges) else np.zeros(0,dtype=bool)
            target=(gt[edges[:,0]]==gt[edges[:,1]]) if len(edges) else np.zeros(0,dtype=bool)
            for arm,model in models.items():
                row=json.loads(next(streams[arm]));assert row['frame']==number
                with torch.no_grad():
                    c=model.visual(torch.from_numpy(features))+model.candidate_geo(torch.from_numpy(boxes/np.asarray([w,h,w,h],np.float32)))
                    scores=[]
                    for start in range(0,len(edges),4096):
                        block=edges[start:start+4096];a,b=c[block[:,0]],c[block[:,1]]
                        scores.append(model.edge(torch.cat([a,b,a*b,torch.from_numpy(geometry[start:start+4096])],-1))[:,1])
                    logits=torch.cat(scores).numpy() if scores else np.empty(0);prob=1/(1+np.exp(-np.clip(logits,-80,80)))
                parent=list(range(len(candidates)))
                def find(i):
                    while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
                    return i
                for (i,j),q,g in zip(edges,prob,geometry):
                    if q>=.5 and g[0]>0:parent[find(int(j))]=find(int(i))
                components={}
                for i in range(len(candidates)):components.setdefault(find(i),[]).append(i)
                actual={tuple(g) for g in components.values() if len(g)>1};expected={tuple(g['candidates']) for g in row['state']['decision_metadata']['competition_components']}
                assert actual==expected,(arm,number,'frozen edge recomputation differs from actual runtime')
                counts[arm]['frames_exact_component_replay']+=1
                for positive in [0,1]:
                    mask=known&(target==bool(positive));n=int(mask.sum());counts[arm]['positive_edges' if positive else 'negative_edges']+=n
                    sums[arm][positive]+=float(prob[mask].sum());buckets=np.clip(((logits[mask]+40)*10).astype(int),0,799)
                    hist[arm][positive]+=np.bincount(buckets,minlength=800)
                counts[arm]['TP']+=int((known&target&(prob>=.5)).sum());counts[arm]['FN']+=int((known&target&(prob<.5)).sum())
                counts[arm]['FP']+=int((known&~target&(prob>=.5)).sum());counts[arm]['TN']+=int((known&~target&(prob<.5)).sum())
    finally:
        replay.close()
        for f in streams.values():f.close()
    report={}
    for arm,c in counts.items():
        neg,pos=hist[arm];den=float(pos.sum())*float(neg.sum());auc=float((pos*(np.cumsum(neg)-.5*neg)).sum())/den if den else None
        assert c['frames_exact_component_replay']==1203
        report[arm]=dict(counts=dict(c),mean_duplicate_probability_positive=sums[arm][1]/c['positive_edges'],mean_duplicate_probability_negative=sums[arm][0]/c['negative_edges'],approximate_logit_histogram_AUROC=auc)
    result=dict(status='FROZEN_EDGE_SCORES_AND_ALL1203_ACTUAL_COMPONENTS_REPRODUCED',arms=report,
        threshold=.5,threshold_optimized=False,optimizer_updates=0,new_GPU_processes=0,new_feature_bank_bytes=0,
        scope='Original DT1203 fixed raw proposals/FP32 vectors. GT used posthoc only. AUROC uses0.1-logit bins over[-40,40] with tail clamping/ties; approximate, not an exact rank statistic.',full_goal_complete=False)
    (out/'DT_duplicate_edge_discrimination.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
