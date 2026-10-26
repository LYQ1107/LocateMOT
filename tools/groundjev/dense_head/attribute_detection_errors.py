"""Post-inference target recovery/loss analysis on the frozen common scope."""
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from tools.groundjev.eval_no_mot20.evaluate_g2_detection import iou_matrix
from tools.groundjev.dense_head.prepare_train_data import RUN,write_json
from tools.groundjev.dense_head.evaluate_dense_head import load_rows


def main():
    rows=load_rows(RUN/'data/external_frozen.jsonl')
    native=json.loads((RUN/'data/A0_external_predictions.json').read_text());results={}
    for variant in ['', 'spatial_prior_v2']:
        for arm in ['g0','g2']:
            root=RUN/variant/arm;receipt=json.loads((root/'receipt.json').read_text())
            predictions=json.loads((root/'external_predictions.json').read_text());threshold=receipt['selected_threshold']
            assert [p['image_path'] for p in predictions]==[r['image_path'] for r in rows]
            groups=defaultdict(lambda:{'GT':0,'both_covered50':0,'recovered50_native_missed':0,
                'lost50_native_covered':0,'both_missed50':0,'recovered50_in_native_rejected_frames':0})
            examples=[]
            for row,a0,prediction in zip(rows,native,predictions):
                truth=row['boxes'];boxes=[b for b,s in zip(prediction['boxes'],prediction['scores']) if s>=threshold]
                first=iou_matrix(truth,a0['boxes']);second=iou_matrix(truth,boxes)
                ca=first.max(1)>=.5 if first.shape[1] else np.zeros(len(truth),dtype=bool)
                cb=second.max(1)>=.5 if second.shape[1] else np.zeros(len(truth),dtype=bool)
                for i,(box,before,after) in enumerate(zip(truth,ca,cb)):
                    area=(box[2]-box[0])*(box[3]-box[1]);key='both_covered50' if before and after else 'recovered50_native_missed' if after else 'lost50_native_covered' if before else 'both_missed50'
                    for group in [row['dataset'],row['dataset']+'/'+row['category'],
                                  row['dataset']+('/small' if area<32**2 else '/medium' if area<96**2 else '/large')]:
                        groups[group]['GT']+=1;groups[group][key]+=1
                        if after and not before and a0['rejected']:groups[group]['recovered50_in_native_rejected_frames']+=1
                    if key in ('recovered50_native_missed','lost50_native_covered') and len(examples)<20:
                        examples.append({'image_path':row['image_path'],'GT_box_index':i,'GT_box':box,'change':key})
            results[(variant or 'global')+'/'+arm]={'head_step':receipt['selected_step'],
                'head_checkpoint_SHA256':receipt['checkpoint_best_SHA256'],'DEV_selected_threshold':threshold,
                'target_coverage_transitions':dict(groups),'first20_changed_targets':examples,
                'not_a_hybrid_model':'Recovered counts are target-level coverage diagnostics; no union detector or tracking gain is claimed.'}
    out=ROOT/'docs/GROUNDJEV_DENSE_HEAD/DETECTION_ERROR_ATTRIBUTION.json';write_json(out,results)
    lines=['# Detection error attribution','',
           'Same1,607 external frames and GT, native final G2 cache vs each frozen readout. Thresholds were fixed on internal DEV. Coverage transitions below do not imply net improvement: consult FP/precision/duplicate and one-to-one metrics in PILOT_RESULTS.json.', '',
           '|Model / dataset|GT|Both covered|Newly recovered native misses|Lost native coverage|Both missed|Recovered after native parser rejection|',
           '|---|---:|---:|---:|---:|---:|---:|']
    for label,result in results.items():
        for dataset in ['animaltrack','sportsmot']:
            g=result['target_coverage_transitions'][dataset]
            lines.append(f"|{label}/{dataset}|{g['GT']}|{g['both_covered50']}|{g['recovered50_native_missed']}|{g['lost50_native_covered']}|{g['both_missed50']}|{g['recovered50_in_native_rejected_frames']}|")
    lines+=['','Per-category/size transitions and changed-target examples are in the JSON. Direct numeric box output has no language grammar/parser step; class/category understanding and detection accuracy remain separate requirements. No AR regeneration, GT candidate construction or external parameter tuning was performed.']
    (ROOT/'docs/GROUNDJEV_DENSE_HEAD/DETECTION_ERROR_ATTRIBUTION.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:{d:v['target_coverage_transitions'][d] for d in ['sportsmot','animaltrack']} for k,v in results.items()},indent=2))


if __name__=='__main__':main()
