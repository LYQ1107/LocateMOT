"""GT-free registered candidate control retaining independently scored native queries."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    from PIL import Image
    from locatemot.groundjev.perception import GroundAnythingAdapter
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Preserve the existing scored-chunk control')
    args.out.mkdir(parents=True)
    frontend=GroundAnythingAdapter(parser_policy='official_scoring',decoder_policy='tracking_dense')
    summary=[]
    with (args.out/'frame_candidates.jsonl').open('w') as output:
        for text in (args.source/'frame_decisions.jsonl').read_text().splitlines():
            row=json.loads(text);original=row['perception'];outcomes=original['chunk_outcomes']
            assert len(outcomes)==13 and sum(c['classes'] for c in outcomes)==1203
            assert [c['offset'] for c in outcomes]==list(range(0,1203,96))
            candidates=[];reports=[]
            with Image.open(row['image_path']) as source:image=source.convert('RGB')
            for outcome in outcomes:
                if outcome['status']=='REJECTED':continue
                metadata=outcome['metadata'];result=metadata['result']
                response={'choices':[{'message':{'content':result['raw_output']},'finish_reason':result['finish_reason']}],
                          'usage':result['usage']}
                current,receipt=frontend.adapt_response(response,image,row['image_path'],metadata['source_encoding'],metadata['prompt'],metadata['seconds'])
                candidates.extend(current);reports.append(receipt)
            perception={'mode':'retained_independently_scored_native_queries','source':str(args.source),
                'query_failure_policy':'failed query chunks have empty predictions; retain other scored chunks; all classes/GT/frames remain',
                'original_perception':original,'model_response_rejected':any(c['status']=='REJECTED' for c in outcomes),
                'all_vocabulary_classes_requested':True,'scored_chunk_reports':reports,
                'candidate_filtering':'none; no NMS, topK, GT filtering, or label substitution',
                'runtime_GT_read':False,'native_HTTP_requests_repeated':False}
            output.write(json.dumps({'frame':row['frame'],'image_path':row['image_path'],'candidates':candidates,'perception':perception})+'\n')
            summary.append({'frame':row['frame'],'retained_candidates':len(candidates),
                'scored_chunks':len(reports),'failed_chunks':sum(c['status']=='REJECTED' for c in outcomes)})
    report={'status':'GT_FREE_NATIVE_QUERY_CONTROL_MATERIALIZED','frames':summary,
        'policy':'failed-query-empty, retain independently scored native queries',
        'original_whole_frame_rejection_preserved':True,'official_parser_changed':False,
        'raw_GT_read':False,'seed_changed':False,'end_to_end_speed_claim':False}
    (args.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
