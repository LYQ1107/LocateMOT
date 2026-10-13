"""Audit real class-query coverage and scored-chunk localization, post-hoc only."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'external/GroundAnything')]


def main():
    from locatemot.groundjev.decision import overlap
    parser=argparse.ArgumentParser()
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads(args.video.read_text())
    rows=[json.loads(line) for line in (args.run/'frame_decisions.jsonl').read_text().splitlines()]
    if [r['frame'] for r in rows]!=manifest['frame_numbers']:raise ValueError('Incomplete registered frame scope')
    source=ROOT/'external/GroundAnything/eval/utils/prompt_mode.py'
    spec=importlib.util.spec_from_file_location('official_parser',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    annotation=json.loads(Path(manifest['annotation_path_for_evaluation_only']).read_text())
    truth={identity:[a for a in annotation['annotations'] if a['image_id']==identity] for identity in manifest['image_ids']}
    frames=[];total=covered=0
    for row,image_id in zip(rows,manifest['image_ids']):
        outcomes=row['perception']['chunk_outcomes']
        size=len(manifest['vocabulary']);chunk=96
        assert [o['offset'] for o in outcomes]==list(range(0,size,chunk))
        assert sum(o['classes'] for o in outcomes)==size
        assert row['perception']['all_vocabulary_classes_requested'] is True
        boxes=[]
        for outcome in outcomes:
            if outcome['status']!='SCORED':continue
            result=outcome['metadata']['result']
            if result['valid']:
                parsed=[(p['label'],coordinates) for p in result['predictions'] for coordinates in p['coordinates'] or []]
            else:parsed=module._parse_gam_predictions_deduplicated(result['raw_output'],'bbox')
            boxes += [[v/999*(manifest['image_size'][i%2]-1) for i,v in enumerate(coordinates)] for label,coordinates in parsed]
        gt=truth[image_id]
        hit=sum(any(overlap([a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]],box)>=.5 for box in boxes) for a in gt)
        total+=len(gt);covered+=hit
        frames.append({'frame':row['frame'],'chunks':len(outcomes),'requested_classes':size,
            'rejected_chunks':sum(o['status']=='REJECTED' for o in outcomes),
            'scored_chunks':sum(o['status']=='SCORED' for o in outcomes),
            'valid_chunk_box_count':len(boxes),'valid_chunk_class_agnostic_GT_covered':hit,
            'GT_boxes':len(gt),'whole_frame_policy_predictions':len(row['outputs'])})
    report={'status':'ALL_REAL_NATIVE_CLASS_REQUESTS_AND_COVERAGE_VERIFIED','scope':manifest['scope'],
        'frames':frames,'all_class_requests':True,'official_TETA':json.loads((args.run/'official_metrics.json').read_text())['metrics'],
        'valid_chunk_localization_coverage':{'covered':covered,'GT_boxes':total,'recall':covered/total,
            'class_agnostic':True,'diagnostic_only':True,'runtime_GT_correction':False},
        'interpretation':'whole-frame rejection discards independently scored chunks; TETA0 does not prove every native class localization is absent',
        'end_to_end_fps':json.loads((args.run/'result.json').read_text())['end_to_end_fps'],
        'next_control':'register failed-class chunks as empty predictions while preserving independently scored chunks, every class query, every GT and frame; retain current whole-frame-rejection result'}
    args.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
