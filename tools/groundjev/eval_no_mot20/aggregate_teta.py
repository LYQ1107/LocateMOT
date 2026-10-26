"""Recompute official TETA jointly on identical completed video scopes.

Never average independently scored per-video TETA values.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tools.groundjev.eval_no_mot20.common import RUN,load,save,fingerprint,merged_eval_manifest,arm_directory


def aggregate(arm,run=RUN,job_ids=None):
    from tools.groundjev.evaluate_multitask import ovmot
    physical_arm=arm_directory(arm)
    jobs=[j for j in load(run/'jobs.json')['jobs'] if j['dataset']=='tao_amodal'
          and (job_ids is None or j['id'] in job_ids)
          and (Path(j['path'])/physical_arm/'frozen_receipt.json').exists()]
    if not jobs:return {'status':'INCOMPLETE','complete_jobs':0}
    signature=fingerprint([{'id':j['id'],'artifact_directory':physical_arm,
        'result':load(Path(j['path'])/physical_arm/'frozen_receipt.json')['prediction_sha256']} for j in jobs])
    out=run/'aggregates'/f'tao_{arm}_{signature[:16]}'
    if (out/'official_teta.json').exists():return load(out/'official_teta.json')
    out.mkdir(parents=True,exist_ok=True)
    manifests=[merged_eval_manifest(Path(j['path'])) for j in jobs]
    first=manifests[0]
    if len({m['annotation_path_for_evaluation_only'] for m in manifests})!=1:
        raise ValueError('Cannot combine different TAO annotation protocols')
    manifest={**first,'video_ids':[m['video_id'] for m in manifests],
              'image_ids':[image for m in manifests for image in m['image_ids']]}
    rows=[]
    for job in jobs:
        with (Path(job['path'])/physical_arm/'frame_decisions.jsonl').open() as handle:
            rows.extend(json.loads(line) for line in handle)
    if len(rows)!=len(manifest['image_ids']):raise ValueError('Merged TAO frame scope is incomplete')
    measured=ovmot(manifest,rows,out)
    report={**measured,'status':'OFFICIAL_JOINT_TETA_COMPLETE','arm':arm,'artifact_directory':physical_arm,'videos':len(jobs),'frames':len(rows),
            'job_ids':[j['id'] for j in jobs],'scope':'REGISTERED_COMPLETE_VIDEO',
            'aggregation':'unmodified official TETA over joint GT/predictions, never mean per-video TETA',
            'signature':signature}
    save(out/'official_teta.json',report);return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=RUN)
    parser.add_argument('--arm',choices=['simple','j1'],required=True);args=parser.parse_args()
    print(json.dumps(aggregate(args.arm,args.root)),flush=True)


if __name__=='__main__':main()
