"""Scored class-agnostic detection protocol; never fabricate A0 ranking AP."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from locatemot.groundjev.dense_head.model import boxes_to_original,valid_ratio
from tools.groundjev.eval_no_mot20.evaluate_g2_detection import iou_matrix
from tools.groundjev.dense_head.prepare_train_data import write_json


def load_rows(path):
    return [json.loads(line) for line in Path(path).open()]


def average_precision(rows,predictions,iou_threshold):
    """101-point interpolated, one-class pooled AP, maxDets=all object queries.

    This is a documented class-agnostic protocol, not official COCO AP.
    Ignored GT objects suppress unmatched overlapping predictions.
    """
    entries=[];matrices=[];ignore_matrices=[];matched=[];total=0
    for i,(row,pred) in enumerate(zip(rows,predictions)):
        total+=len(row['boxes']);matrices.append(iou_matrix(pred['boxes'],row['boxes']))
        ignore_matrices.append(iou_matrix(pred['boxes'],row.get('ignored',[])))
        matched.append(set())
        entries.extend((float(s),i,j) for j,s in enumerate(pred['scores']))
    entries.sort(key=lambda e:-e[0]);tp=[];fp=[]
    for _,i,j in entries:
        m=matrices[i]
        candidates=[k for k in range(m.shape[1]) if k not in matched[i] and m[j,k]>=iou_threshold]
        if candidates:
            k=max(candidates,key=lambda k:m[j,k]);matched[i].add(k);tp.append(1);fp.append(0)
        elif ignore_matrices[i].shape[1] and ignore_matrices[i][j].max()>=iou_threshold:
            continue
        else:tp.append(0);fp.append(1)
    if not total:return None
    tp=np.cumsum(tp);fp=np.cumsum(fp);recall=tp/total;precision=tp/np.maximum(tp+fp,1)
    return float(np.mean([precision[recall>=r].max(initial=0) for r in np.linspace(0,1,101)]))


def summarize(rows,predictions,threshold=.5,compute_map=True):
    n_gt=n_pred=matched50=matched75=covered50=covered75=duplicates=ignored_count=0
    iou_sum=0.;groups=defaultdict(lambda:{'GT':0,'covered50':0,'covered75':0})
    for row,pred in zip(rows,predictions):
        boxes=[b for b,s in zip(pred['boxes'],pred['scores']) if s>=threshold]
        m=iou_matrix(row['boxes'],boxes);best=m.max(1) if boxes else np.zeros(len(row['boxes']))
        local_matches=0;matched_indices=set()
        if m.size:
            a,b=linear_sum_assignment(-(m>=.5).astype(float)-m*1e-6)
            matched_indices={int(j) for i,j in zip(a,b) if m[i,j]>=.5};local_matches=len(matched_indices)
            a,b=linear_sum_assignment(-(m>=.75).astype(float)-m*1e-6)
            matched75+=sum(m[i,j]>=.75 for i,j in zip(a,b))
        ignored=iou_matrix(row.get('ignored',[]),boxes)
        ignored_count+=sum(j not in matched_indices and ignored.shape[0]>0 and ignored[:,j].max()>=.5 for j in range(len(boxes)))
        matched50+=local_matches;n_gt+=len(row['boxes']);n_pred+=len(boxes)
        covered50+=int((best>=.5).sum());covered75+=int((best>=.75).sum());iou_sum+=float(best.sum())
        duplicates+=max(0,int((m.max(0)>=.5).sum())-local_matches) if m.size else 0
        for box,value in zip(row['boxes'],best):
            area=(box[2]-box[0])*(box[3]-box[1])
            labels=[row['category'],'small' if area<32**2 else 'medium' if area<96**2 else 'large',
                    'dense_ge20' if len(row['boxes'])>=20 else 'non_dense']
            for group in set(labels):
                groups[group]['GT']+=1;groups[group]['covered50']+=int(value>=.5);groups[group]['covered75']+=int(value>=.75)
    fp=n_pred-matched50-ignored_count
    result={'frames':len(rows),'videos':len({r['sequence'] for r in rows}),'GT':n_gt,'predictions':n_pred,
            'threshold':threshold,'Recall50':covered50/max(n_gt,1),'Recall75':covered75/max(n_gt,1),
            'one_to_one_Recall50':matched50/max(n_gt,1),'one_to_one_Recall75':matched75/max(n_gt,1),
            'Precision':matched50/max(n_pred-ignored_count,1),'FP_per_frame':fp/max(len(rows),1),
            'FP':fp,'FN':n_gt-matched50,'duplicates':duplicates,'mean_best_IoU':iou_sum/max(n_gt,1),
            'groups':{g:{**v,'Recall50':v['covered50']/v['GT'],'Recall75':v['covered75']/v['GT']} for g,v in groups.items()}}
    p,r=result['Precision'],result['one_to_one_Recall50'];result['F1']=2*p*r/max(p+r,1e-12)
    if compute_map:
        ap=[average_precision(rows,predictions,t) for t in np.linspace(.5,.95,10)]
        result.update(AP50=ap[0],AP75=ap[5],mAP=float(np.mean(ap)),AP_protocol='class-agnostic 101-point, all queries; not official COCO')
    return result


def evaluate_predictions(rows,predictions,threshold=.5,compute_map=True,per_video=False):
    output={}
    for dataset in sorted({r['dataset'] for r in rows}):
        indices=[i for i,r in enumerate(rows) if r['dataset']==dataset]
        output[dataset]=summarize([rows[i] for i in indices],[predictions[i] for i in indices],threshold,compute_map)
    if per_video:
        output['per_video']={}
        for sequence in sorted({(r['dataset'],r['sequence']) for r in rows}):
            idx=[i for i,r in enumerate(rows) if (r['dataset'],r['sequence'])==sequence]
            output['per_video']['/'.join(sequence)]=summarize([rows[i] for i in idx],[predictions[i] for i in idx],threshold,compute_map)
    return output


def select_threshold(rows,predictions):
    choices=[]
    for threshold in [.05,.1,.2,.3,.5,.7]:
        m=evaluate_predictions(rows,predictions,threshold,compute_map=False)
        choices.append({'threshold':threshold,'macro_domain_F1':float(np.mean([v['F1'] for v in m.values()]))})
    return max(choices,key=lambda r:(r['macro_domain_F1'],r['threshold']))['threshold'],choices


@torch.no_grad()
def predict(head,store,rows,use_cache=True):
    head.eval();predictions=[];vision_seconds=decoder_seconds=0.;started=time.perf_counter()
    for index,row in enumerate(rows):
        spatial,meta=store.get(row['image_path'],use_cache)
        torch.cuda.synchronize();began=time.perf_counter()
        pred=head(spatial,valid_ratio(meta));boxes=boxes_to_original(pred['boxes'][0],meta)
        scores=pred['logits'][0].sigmoid();valid=(boxes[:,2]>boxes[:,0]) & (boxes[:,3]>boxes[:,1])
        torch.cuda.synchronize();elapsed=time.perf_counter()-began
        predictions.append({'image_path':row['image_path'],'boxes':boxes[valid].cpu().tolist(),
                            'scores':scores[valid].cpu().tolist(),'vision_seconds':meta['seconds'],
                            'decoder_seconds':elapsed})
        vision_seconds+=meta['seconds'];decoder_seconds+=elapsed
        if (index+1)%250==0:
            print(json.dumps({'event':'DENSE_EVAL_PROGRESS','frames':index+1,'total':len(rows)}),flush=True)
    wall=time.perf_counter()-started
    return predictions,{'frames':len(rows),'summed_actual_extraction_seconds':vision_seconds,
                        'decoder_seconds':decoder_seconds,'vision_FPS':len(rows)/max(vision_seconds,1e-9),
                        'decoder_FPS':len(rows)/max(decoder_seconds,1e-9),
                        'vision_plus_decoder_FPS':len(rows)/max(vision_seconds+decoder_seconds,1e-9),
                        'evaluation_wall_seconds':wall,'evaluation_wall_FPS':len(rows)/max(wall,1e-9),
                        'cache_used':use_cache,'timing_note':'Cached extraction times are original GPU forward timings; fresh final evaluation disables feature cache. Includes native preprocessing; image-open IO is included in fresh wall, not vision timer.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--index',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--threshold',type=float,required=True);a=p.parse_args()
    import yaml
    from locatemot.groundjev.dense_head.spatial_backbone import SpatialBackbone,BoundedFeatureStore
    from locatemot.groundjev.dense_head.object_decoder import ObjectDecoder
    c=yaml.safe_load(a.config.read_text());torch.set_num_threads(4);b=SpatialBackbone(c['model']);h=ObjectDecoder(queries=c['queries']).cuda()
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);h.load_state_dict(saved['head'])
    store=BoundedFeatureStore(b,a.out.parent/'temporary_features',c['feature_cache_bytes'])
    rows=load_rows(a.index);pred,timing=predict(h,store,rows,use_cache=False)
    write_json(a.out,{'metrics':evaluate_predictions(rows,pred,a.threshold,per_video=True),'timing':timing,'checkpoint_step':saved['step']})
    write_json(a.out.with_suffix('.predictions.json'),pred)


if __name__=='__main__':main()
