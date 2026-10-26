"""Audit frozen evaluation images against actual G2 TRAIN rows, without GT."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tools.groundjev.eval_no_mot20.common import RUN,TRAIN_ROOT,load,save


def image_key(path):
    text=str(path)
    # TAO is distributed over four disks. Its official frame-relative path is
    # the image identity; a different disk prefix is not an independent image.
    marker='/TAO-Amodal/frames/'
    return 'tao:'+text.split(marker,1)[1] if marker in text else text


def audit(run=RUN):
    image_jobs=defaultdict(set);job_frames={};job_dataset={}
    for entry in load(run/'jobs.json')['jobs']:
        manifest=load(Path(entry['path'])/'manifest.json');job_dataset[entry['id']]=entry['dataset']
        for frame,path in zip(manifest['frame_numbers'],manifest['images']):
            key=image_key(path);image_jobs[key].add(entry['id']);job_frames[(entry['id'],key)]=frame
    overlaps=defaultdict(set);source_rows=0;train_images=0
    for line in (TRAIN_ROOT/'g2_clean_train_rows.jsonl').open():
        row=json.loads(line);source_rows+=1
        for image in row['images']:
            train_images+=1;key=image_key(image['path'])
            for job in image_jobs.get(key,()):overlaps[job].add((job_frames[(job,key)],key,row['source']))
    datasets={}
    for dataset in sorted(set(job_dataset.values())):
        selected=[j for j in job_dataset if job_dataset[j]==dataset]
        datasets[dataset]={'jobs':len(selected),'overlapping_jobs':sum(bool(overlaps[j]) for j in selected),
            'overlapping_query_frames':sum(len(overlaps[j]) for j in selected)}
    result={'status':'REGISTERED_G2_TRAIN_IMAGE_DISJOINT' if not any(overlaps.values()) else 'REGISTERED_TRAIN_OVERLAP_FOUND',
        'G2_rows_read':source_rows,'G2_image_references_read':train_images,
        'unique_registered_images':len(image_jobs),'datasets':datasets,
        'overlaps':{job:sorted(values) for job,values in overlaps.items() if values},
        'scope':'actual G2 fine-tuning TRAIN rows; exact paths with TAO disk aliases normalized; source-video audit is separate',
        'GT_read':False,'model_pretraining_overlap_claim':False}
    save(run/'registered_image_overlap_audit.json',result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=RUN);args=parser.parse_args()
    report=audit(args.root);print(json.dumps({k:v for k,v in report.items() if k!='overlaps'}),flush=True)


if __name__=='__main__':main()
