"""Known video alias and representative-content audit, without changing splits."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tools.groundjev.dense_head.prepare_train_data import RUN,G2_TRAIN,write_json


def main():
    protocol=json.loads((RUN/'data/protocol.json').read_text())
    manifest=json.loads(G2_TRAIN.read_text());coverage=manifest['coverage_videos_or_sequences']
    rmot_images=set();counts=Counter()
    for line in Path(manifest['rows_path']).open():
        row=json.loads(line)
        if row['source'].startswith('RMOT26'):
            counts['RMOT26_G2_TRAIN_rows']+=1
            rmot_images.update(str(Path(image['path']).resolve()) for image in row['images'])
    hashes={}
    for path in sorted(rmot_images):
        p=Path(path)
        if not p.exists():counts['missing_RMOT26_source_images']+=1;continue
        sha=hashlib.sha256(p.read_bytes()).hexdigest();hashes.setdefault(sha,[]).append(path)
    collisions=[]
    for v in protocol['videos']:
        for path,sha in v['representative_image_sha256'].items():
            if sha in hashes:collisions.append({'head_split':v['head_split'],'video':v['sequence'],
                'image':path,'same_bytes_as_RMOT26_G2_TRAIN':hashes[sha]})
    external=json.loads((RUN/'data/external_protocol.json').read_text());external_matches=[]
    for job in external['jobs']:
        m=json.loads((Path(job['path'])/'manifest.json').read_text());images=m['images']
        for i in sorted({0,len(images)//2,len(images)-1}):
            p=Path(images[i]);sha=hashlib.sha256(p.read_bytes()).hexdigest()
            if sha in hashes:external_matches.append({'video':job['sequence'],'image':str(p),'RMOT26_sources':hashes[sha]})
    mot17={re.sub(r'-(DPM|FRCNN|SDP)$','',s) for s in coverage['MOT17']}
    rmot17={re.sub(r'-(DPM|FRCNN|SDP)$','',s.replace('MOT16-','MOT17-')) for s in coverage['RMOT26'] if s.startswith(('MOT16-','MOT17-'))}
    dance=set(coverage['DanceTrack']) & set(coverage['RMOT26'])
    head_sources={v['dataset'] for v in protocol['videos']}
    assert head_sources=={'sportsmot','animaltrack'}
    result={'status':'KNOWN_SOURCE_ALIAS_AND_REPRESENTATIVE_HASH_AUDIT',
        'head_optimizer_sources':sorted(head_sources),'MOT17_detector_variant_canonical_ids':sorted(mot17),
        'MOT17_RMOT26_G2_TRAIN_overlap':sorted(mot17 & rmot17),
        'DanceTrack_RMOT26_G2_TRAIN_overlap':sorted(dance),
        'these_sources_enter_head_optimizer':False,'RMOT26_rows_and_images':dict(counts),
        'RMOT26_unique_source_images_hashed':len(rmot_images),
        'head_video_representative_hash_matches_to_RMOT26':collisions,
        'external_representative_hash_matches_to_RMOT26':external_matches,
        'G2_TRAIN_rows_SHA256':manifest['rows_sha256'],
        'audit_boundary':'Exact bytes at representative frames plus known video/source aliases. Does not prove absence of re-encoded/resized/cropped copies or unseen-camera independence. Internal head DEV is already explicitly G2-TRAIN scope.'}
    write_json(RUN/'audits/cross_source_aliases.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
