"""Two existing TRAIN images through the trained direct numeric decoder, no fit."""
import json
import argparse
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import torch
from PIL import Image

from locatemot.groundjev.dense_head.spatial_backbone import SpatialBackbone
from locatemot.groundjev.dense_head.object_decoder import ObjectDecoder
from locatemot.groundjev.dense_head.model import targets_on_grid,boxes_to_original,valid_ratio
from locatemot.groundjev.dense_head.hungarian_criterion import HungarianCriterion
from tools.groundjev.eval_no_mot20.evaluate_g2_detection import iou_matrix
from tools.groundjev.dense_head.prepare_train_data import RUN,write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,default=RUN)
    args=parser.parse_args();run=args.run_root
    started=time.perf_counter();torch.set_num_threads(4)
    assert json.loads((run/'g2/receipt.json').read_text())['status']=='REAL_GPU_PILOT_AND_FULL_DEV_COMPLETE'
    saved=torch.load(run/'g2/best_dev.pt',map_location='cpu',weights_only=False)
    selection=json.loads((run/'g2/frozen_selection.json').read_text());threshold=selection['threshold']
    backbone=SpatialBackbone(saved['config']['model']);head=ObjectDecoder(queries=saved['config']['queries'],
        spatial_attention_sigma=saved['config'].get('spatial_attention_sigma')).cuda()
    head.load_state_dict(saved['head']);head.eval();out=[]
    for row in json.loads((RUN/'data/auxiliary_rows.json').read_text()):
        with Image.open(row['image_path']) as im:features,meta=backbone.encode(im.convert('RGB'))
        with torch.no_grad():
            pred=head(features,valid_ratio(meta));boxes=boxes_to_original(pred['boxes'][0],meta)
            scores=pred['logits'][0].sigmoid()
            positive_only=HungarianCriterion()(pred,targets_on_grid(row['boxes'],meta,'cuda:0'),exhaustive=False)
        assert torch.isfinite(boxes).all() and torch.isfinite(scores).all() and torch.isfinite(positive_only['loss'])
        keep=scores>=threshold;matrix=iou_matrix(row['boxes'],boxes[keep].cpu().tolist())
        coverage=float((matrix.max(1)>=.5).mean()) if keep.any() else 0.
        out.append({'dataset':row['dataset'],'image_path':row['image_path'],'source_split':row['source_split'],
                    'model_step':saved['step'],'raw_parallel_queries':len(boxes),'above_DEV_threshold':int(keep.sum()),
                    'threshold':threshold,'Base_or_physical_GT_count':len(row['boxes']),
                    'coverage50_on_this_TRAIN_image':coverage,'boxes':boxes[keep].cpu().tolist(),
                    'scores':scores[keep].cpu().tolist(),'positive_only_loss':float(positive_only['loss']),
                    'finite_numeric_coordinates':True,'native_text_decoder_invoked':False,
                    'AP_Precision_TETA':None,'reason':'Single TRAIN diagnostic, partial/category-limited annotation, class-agnostic decoder'} )
    write_json(RUN/'audits/auxiliary_GPU_diagnostic.json',{'status':'REAL_GPU_DIRECT_DECODER_INTERFACE_DIAGNOSTIC_COMPLETE',
        'trained_run':str(run),'spatial_attention_sigma':saved['config'].get('spatial_attention_sigma'),
        'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],'rows':out,'optimizer_updates':0,'Novel_labels_used':False,
        'elapsed_seconds':time.perf_counter()-started,'GPU_hours':(time.perf_counter()-started)/3600,
        'open_vocabulary_1203_class_claim':False,'RMOT_expression_satisfaction_claim':False})
    print(json.dumps({'status':'REAL_GPU_DIRECT_DECODER_INTERFACE_DIAGNOSTIC_COMPLETE',
                      'rows':[{k:r[k] for k in ['dataset','above_DEV_threshold','coverage50_on_this_TRAIN_image']} for r in out]}),flush=True)


if __name__=='__main__':main()
