"""Read a real GPU-trained head on its image-only cached features (CPU diagnostic)."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import torch
from locatemot.groundjev.dense_head.object_decoder import ObjectDecoder
from locatemot.groundjev.dense_head.model import valid_ratio,boxes_to_original,targets_on_grid
from locatemot.groundjev.dense_head.hungarian_criterion import HungarianCriterion
from tools.groundjev.dense_head.prepare_train_data import RUN,write_json
from tools.groundjev.eval_no_mot20.evaluate_g2_detection import iou_matrix


def main():
    torch.set_num_threads(4);results={}
    dev=[json.loads(line) for line in (RUN/'data/dev.jsonl').open()]
    examples=[next(r for r in dev if r['dataset']==d) for d in ['sportsmot','animaltrack']]
    for arm in ['g0','g2']:
        checkpoint=RUN/arm/'best_dev.pt'
        if not checkpoint.exists():continue
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False);c=saved['config']
        head=ObjectDecoder(queries=c['queries'],spatial_attention_sigma=c.get('spatial_attention_sigma'));head.load_state_dict(saved['head']);head.eval()
        model=Path(c['model']).resolve()/'model.safetensors';identity=[str(model.resolve()),model.stat().st_size,model.stat().st_mtime_ns,c['max_tokens']]
        rows=[]
        for row in examples:
            p=Path(row['image_path']);stat=p.stat()
            key=hashlib.sha256(json.dumps([identity,str(p.resolve()),stat.st_size,stat.st_mtime_ns]).encode()).hexdigest()
            cached=RUN/arm/'features'/(key+'.pt')
            if not cached.exists():continue
            data=torch.load(cached,map_location='cpu',weights_only=False);feature=data['spatial'];meta=data['metadata']
            with torch.no_grad():
                pred=head(feature,valid_ratio(meta));boxes=boxes_to_original(pred['boxes'][0],meta)
                loss=HungarianCriterion()(pred,targets_on_grid(row['boxes'],meta,'cpu'))
            matrix=iou_matrix(row['boxes'],boxes.tolist());best=matrix.max(1)
            probability=pred['logits'][0].sigmoid()
            rows.append({'dataset':row['dataset'],'sequence':row['sequence'],'frame':row['frame'],
                'GT_count':len(row['boxes']),'queries':len(boxes),'GT_boxes':row['boxes'],
                'predicted_boxes':boxes.tolist(),'predicted_scores':probability.tolist(),
                'objectness_quantiles':torch.quantile(probability,torch.tensor([0.,.25,.5,.75,1.])).tolist(),
                'coverage50_all_queries':float((best>=.5).mean()),'mean_best_IoU_all_queries':float(best.mean()),
                'Hungarian_losses':{k:float(v) for k,v in loss.items()},
                'feature_shape':list(feature.shape),'feature_finite':bool(torch.isfinite(feature).all()),
                'pixel_box_finite':bool(torch.isfinite(boxes).all())})
        results[arm]={'actual_optimizer_step':saved['step'],'examples':rows}
    write_json(RUN/'audits/checkpoint_readout_diagnostic.json',{'status':'REAL_CHECKPOINT_CACHED_FEATURE_CPU_DIAGNOSTIC',
        'CPU_training':False,'mock_data':False,'feature_source':'native real GPU cache','results':results})
    print(json.dumps({a:{'step':v['actual_optimizer_step'],'examples':[{k:r[k] for k in ['dataset','GT_count','objectness_quantiles','coverage50_all_queries','mean_best_IoU_all_queries']} for r in v['examples']]} for a,v in results.items()},indent=2))


if __name__=='__main__':main()
