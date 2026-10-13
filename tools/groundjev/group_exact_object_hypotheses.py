"""GT-free class-agnostic grouping of identical native box geometries."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Preserve the previous object-hypothesis control')
    args.out.mkdir(parents=True)
    counts=[]
    with (args.out/'frame_candidates.jsonl').open('w') as output:
        for text in args.source.read_text().splitlines():
            row=json.loads(text);grouped={}
            for index,candidate in enumerate(row['candidates']):
                key=tuple(candidate['box'])
                if key not in grouped:
                    grouped[key]=dict(candidate,label_hypotheses=[],source_candidate_indices=[])
                group=grouped[key]
                if candidate['label'] not in group['label_hypotheses']:group['label_hypotheses'].append(candidate['label'])
                group['source_candidate_indices'].append(index)
            assert set(grouped)=={tuple(c['box']) for c in row['candidates']}
            assert sum(len(c['source_candidate_indices']) for c in grouped.values())==len(row['candidates'])
            perception={'mode':'exact_object_geometry_hypothesis_grouping','original_perception':row['perception'],
                'model_response_rejected':row['perception'].get('model_response_rejected',False),
                'candidate_policy':'merge only exactly equal native geometries; preserve all proposed labels and distinct boxes',
                'label_policy':'first native label is an explicit unlearned category baseline; all hypotheses retained for a future typed native class head',
                'GT_read':False,'new_detector_requests':False,'no_NMS_or_topK':True}
            output.write(json.dumps({'frame':row['frame'],'image_path':row['image_path'],'candidates':list(grouped.values()),'perception':perception})+'\n')
            counts.append({'frame':row['frame'],'native_label_box_rows':len(row['candidates']),'distinct_object_geometries':len(grouped)})
    report={'status':'EXACT_NATIVE_GEOMETRY_GROUPING_VALIDATED','frames':counts,'distinct_geometry_coverage_preserved':True,
        'all_original_class_labels_preserved':True,'classification_improvement_claim':False,'GT_read':False,
        'official_GT_metric_or_parser_changed':False,'source':str(args.source)}
    (args.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
