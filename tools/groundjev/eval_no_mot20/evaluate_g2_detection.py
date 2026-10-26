"""Score completed native candidate sets; GT is loaded only in this module.

No ranking AP is invented for unscored GroundAnything outputs. TAO unmatched
boxes are reported as unverified, because its annotation is non-exhaustive.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tools.groundjev.eval_no_mot20.common import load,save,merged_eval_manifest


def ground_truth(manifest):
    result=defaultdict(list);ignored=defaultdict(list)
    if manifest['task']=='mot':
        for line in Path(manifest['gt_path_for_evaluation_only']).read_text().splitlines():
            f=line.split(',')
            if len(f)<7:continue
            frame,identity=int(float(f[0])),int(float(f[1]));x,y,w,h=map(float,f[2:6])
            if (manifest['dataset']!='animaltrack' and len(f)>7 and int(float(f[7]))!=1) or float(f[6])<=0:
                ignored[frame].append([x,y,x+w,y+h]);continue
            result[frame].append({'identity':identity,'box':[x,y,x+w,y+h],
                                  'visibility':float(f[8]) if len(f)>8 and float(f[8])>=0 else None})
    elif manifest['task']=='rmot':
        query=load(Path(manifest['expression_file_for_evaluation_only']))
        width,height=manifest['image_size']
        for frame in manifest['frame_numbers']:
            ids=set(query['label'].get(str(frame),[]))
            if not ids:continue
            p=Path(manifest['label_dir_for_evaluation_only'])/f'{frame:06d}.txt'
            for line in p.read_text().splitlines():
                f=line.split();identity=int(f[1])
                if identity in ids:
                    x,y,w,h=map(float,f[2:6]);result[frame].append({'identity':identity,
                        'box':[x*width,y*height,(x+w)*width,(y+h)*height]})
            if {g['identity'] for g in result[frame]}!=ids:
                raise ValueError(f'Missing expression target GT: {frame}')
    else:
        annotation=load(Path(manifest['annotation_path_for_evaluation_only']))
        by_image=dict(zip(manifest['image_ids'],manifest['frame_numbers']))
        categories={c['id']:c for c in annotation['categories']}
        for row in annotation['annotations']:
            if row['image_id'] not in by_image:continue
            x,y,w,h=row['bbox'];category=categories[row['category_id']]
            result[by_image[row['image_id']]].append({'identity':row['track_id'],
                'box':[x,y,x+w,y+h],'category_id':category['id'],
                'group':'novel' if category['frequency']=='r' else 'base'})
    return result,ignored


def iou_matrix(first,second):
    import numpy as np
    a=np.asarray(first,dtype=float).reshape(-1,4);b=np.asarray(second,dtype=float).reshape(-1,4)
    inter=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(-1)
    aa=np.maximum(0,a[:,2:]-a[:,:2]).prod(-1);bb=np.maximum(0,b[:,2:]-b[:,:2]).prod(-1)
    return inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-12)


def measure(job):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    manifest=merged_eval_manifest(job);receipt=load(job/'cache_receipt.json')
    gt,ignored=ground_truth(manifest)
    count=covered50=covered75=matched50=unmatched=duplicates=predictions=empty_fp=empty_frames=multi=0
    best_values=[];groups=defaultdict(lambda:{'GT':0,'covered50':0,'covered75':0})
    frame_rows=[];unmapped=0;latencies=[]
    if manifest['task']=='ovmot':
        annotation=load(Path(manifest['annotation_path_for_evaluation_only']))
        aliases={}
        def normal(s):return ' '.join(s.lower().replace('_',' ').split())
        for c in annotation['categories']:
            for s in [c['name']]+c.get('synonyms',[]):aliases[normal(s)]=c['id']
    for line in (job/'detector_candidates.jsonl').open():
        row=json.loads(line);frame=row['frame'];truth=gt[frame];candidates=row['candidates']
        matrix=iou_matrix([g['box'] for g in truth],[c['box'] for c in candidates])
        best=matrix.max(1) if len(candidates) else np.zeros(len(truth))
        c50=int((best>=.5).sum());c75=int((best>=.75).sum())
        count+=len(truth);predictions+=len(candidates);covered50+=c50;covered75+=c75
        best_values.extend(best.tolist());latencies.append(row['seconds'])
        if len(truth)>1:multi+=len(truth)
        if matrix.size:
            rr,cc=linear_sum_assignment(-(matrix>=.5).astype(float)-matrix*1e-6)
            matched=int(sum(matrix[i,j]>=.5 for i,j in zip(rr,cc)))
        else:matched=0
        matched50+=matched
        fp=len(candidates)-matched
        if ignored[frame] and candidates:
            overlap_ignore=iou_matrix([g['box'] for g in truth]+ignored[frame],[c['box'] for c in candidates])
            # Count as FP only unmatched boxes that overlap neither a target nor
            # any ignored object. Duplicate target boxes are counted separately.
            verified_fp=int(sum(v<.5 for v in overlap_ignore.max(0)))
        else:verified_fp=fp
        unmatched+=verified_fp
        duplicate_count=max(0,int((matrix.max(0)>=.5).sum())-matched) if matrix.size else 0
        duplicates+=duplicate_count
        if not truth:empty_frames+=1;empty_fp+=len(candidates)
        for g,value in zip(truth,best):
            box=g['box'];area=max(0,box[2]-box[0])*max(0,box[3]-box[1])
            keys=[g.get('group','all'),'small' if area<32**2 else 'medium' if area<96**2 else 'large',
                  'dense_ge20' if len(truth)>=20 else 'non_dense']
            if g.get('visibility') is not None:keys.append('occluded_visibility_lt0.5' if g['visibility']<.5 else 'visible_ge0.5')
            for key in set(keys):
                groups[key]['GT']+=1;groups[key]['covered50']+=int(value>=.5);groups[key]['covered75']+=int(value>=.75)
        if manifest['task']=='ovmot':
            mapped=[aliases.get(normal(c['label'])) for c in candidates]
            unmapped+=sum(v is None for v in mapped)
            for g,values in zip(truth,matrix):
                correct=max((v for v,category in zip(values,mapped) if category==g['category_id']),default=0.)
                key=g['group']+'_class_correct';groups[key]['GT']+=1
                groups[key]['covered50']+=int(correct>=.5);groups[key]['covered75']+=int(correct>=.75)
        frame_rows.append({'frame':frame,'GT':len(truth),'candidates':len(candidates),'covered50':c50,'covered75':c75,
                           'matched50':matched,'unmatched':fp,'duplicates':duplicate_count})
    report={'status':'G2_DETECTION_EVALUATION_COMPLETE','dataset':manifest['dataset'],
            'sequence':manifest['sequence'],'expression_id':manifest.get('expression_id'),
            'scope':manifest['result_scope'],'frames':len(frame_rows),'GT_boxes':count,'candidates':predictions,
            'covered50':covered50,'covered75':covered75,'missed50':count-covered50,'matched50':matched50,
            'Recall@0.50':covered50/count if count else None,'Recall@0.75':covered75/count if count else None,
            'one_to_one_recall50':matched50/count if count else None,
            'FP_per_frame':None if manifest['task']=='ovmot' else unmatched/len(frame_rows),
            'unmatched_candidates':unmatched,'unmatched_TAO_are_not_verified_false_positives':manifest['task']=='ovmot',
            'duplicates':duplicates,'empty_target_frames':empty_frames,'empty_frame_predictions':empty_fp,
            'multi_instance_GT':multi,'rejected':receipt['rejected'],'groups':dict(groups),
            'IoU_quantiles':{str(q):float(np.quantile(best_values,q)) for q in (.1,.5,.9)} if best_values else {},
            'AP50':None,'AP75':None,'mAP':None,'AP_unavailable_reason':'native per-box confidence is absent; no fabricated ranking',
            'detector_only_fps':receipt['detector_only_fps'],'mean_latency_seconds':float(np.mean(latencies)),
            'p95_latency_seconds':float(np.quantile(latencies,.95)),
            'physical_video_FPS_available':manifest['physical_frames_complete'],
            'unmapped_labels':unmapped,'model_sha256':receipt['model_sha256'],'model_step':receipt['model_step'],
            'candidate_sha256':receipt['candidate_sha256'],'GT_access':'post-inference evaluation only'}
    save(job/'detection_metrics.json',report);save(job/'detection_per_frame.json',frame_rows)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(measure(args.job)),flush=True)


if __name__=='__main__':main()
