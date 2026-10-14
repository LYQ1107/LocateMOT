"""Stream one full legal TAO video through every vocabulary query and real native ROIs."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    from PIL import Image
    from locatemot.groundjev.perception import GroundAnythingAdapter, GroundingProtocolError
    from locatemot.groundjev.feature_client import NativeFeatureClient
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(args.manifest.read_text())
    assert manifest['dataset']=='tao_amodal' and manifest['task']=='ovmot' and manifest['training_on_this_scope'] is False
    frontend=GroundAnythingAdapter('http://127.0.0.1:8104/v1',parser_policy='official_scoring',decoder_policy='tracking_dense')
    features=NativeFeatureClient('http://127.0.0.1:8112')
    start=time.perf_counter();counts={'frames':0,'native_query_requests':0,'scored_queries':0,'rejected_queries':0,'candidates':0}
    with (args.out/'frame_candidates.jsonl').open('w') as log:
        for frame,path in zip(manifest['frame_numbers'],manifest['images']):
            with Image.open(path) as source:image=source.convert('RGB')
            candidates=[];outcomes=[];vocabulary=manifest['vocabulary']
            for offset in range(0,len(vocabulary),96):
                chunk=vocabulary[offset:offset+96]
                try:
                    current,metadata=frontend._request(image,'ovmot',chunk,None,image_path=path)
                    candidates.extend(current);status='SCORED';counts['scored_queries']+=1
                except GroundingProtocolError as error:
                    metadata=error.metadata;status='REJECTED';counts['rejected_queries']+=1
                outcomes.append({'offset':offset,'classes':len(chunk),'status':status,'metadata':metadata})
                counts['native_query_requests']+=1
            candidates,feature_metadata=features(image,candidates,image_path=path)
            perception={'mode':'full_vocabulary_per_query_scoring','all_vocabulary_classes_requested':True,
                'requested_vocabulary_size':len(vocabulary),'chunk_outcomes':outcomes,
                'chunk_failure_policy':'each original-parser-rejected query contributes empty; all other native boxes retained',
                'candidate_merge':'concatenate, no NMS/topK/GT label or box replacement',
                'native_feature_extraction':feature_metadata,'GT_read':False}
            log.write(json.dumps({'frame':frame,'image_path':path,'candidates':candidates,'perception':perception})+'\n');log.flush()
            counts['frames']+=1;counts['candidates']+=len(candidates)
            print(json.dumps({'frame':frame,**counts}),flush=True)
            assert (args.out/'frame_candidates.jsonl').stat().st_size<25*2**30
    assert counts['frames']==len(manifest['images']) and counts['native_query_requests']==len(manifest['images'])*13
    result={'status':'COMPLETE_LEGAL_NATIVE_VOCABULARY_AND_ROI_STREAM','manifest':str(args.manifest),
        **counts,'all1203classes_requested_on_every_frame':True,'per_box_native_score':'absent, never fabricated',
        'GT_or_Novel_labels_read_by_model':False,'raw_images_reencoded_lossily':False,
        'stream_wall_seconds':time.perf_counter()-start,'end_to_end_tracking_FPS_claim':False,
        'feature_cache_bytes':(args.out/'frame_candidates.jsonl').stat().st_size}
    (args.out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
