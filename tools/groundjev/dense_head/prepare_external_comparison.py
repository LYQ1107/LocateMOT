"""Freeze small independent cached A0 scopes before fitting either head."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tools.groundjev.eval_no_mot20.evaluate_g2_detection import ground_truth
from tools.groundjev.dense_head.prepare_train_data import RUN,write_json,digest,physical_source


def main():
    if (RUN/'data/external_protocol.json').exists():
        receipt=json.loads((RUN/'data/external_protocol.json').read_text())
        assert digest(RUN/'data/external_frozen.jsonl')==receipt['index_SHA256']
        print(json.dumps({'status':'EXISTING_FROZEN_EXTERNAL_SCOPE_REUSED','frames':receipt['frames']}))
        return
    old=Path('/data3/user/LocateMOT-runs/GroundJEV_Eval_NoMOT20_4GPU/request_local_seed_v2')
    jobs=json.loads((old/'jobs.json').read_text())['jobs'];selected=[]
    for dataset in ['sportsmot','animaltrack']:
        complete=[j for j in jobs if j['dataset']==dataset and (Path(j['path'])/'cache_receipt.json').exists()]
        complete.sort(key=lambda j:hashlib.sha256(('dense-pilot42:'+j['id']).encode()).hexdigest())
        seen=set()
        for job in complete:
            group=physical_source(dataset,job['sequence']) if dataset=='sportsmot' else job['sequence'].rsplit('_',1)[0]
            if group in seen:continue
            seen.add(group);selected.append(job)
            if len(seen)==3:break
    rows=[];predictions=[]
    for job in selected:
        path=Path(job['path']);manifest=json.loads((path/'manifest.json').read_text())
        manifest.update(json.loads((path/'evaluation.json').read_text()))
        assert manifest['independent_video'] and not manifest['training_on_this_scope']
        gt,ignored=ground_truth(manifest)
        cache=[json.loads(x) for x in (path/'detector_candidates.jsonl').open()]
        assert len(cache)==job['frames']
        for prediction in cache:
            frame=prediction['frame'];boxes=[g['box'] for g in gt[frame]]
            rows.append({'dataset':job['dataset'],'sequence':job['sequence'],'source_group':physical_source(job['dataset'],job['sequence']),
                         'category':'person' if job['dataset']=='sportsmot' else job['sequence'].rsplit('_',1)[0],
                         'image_path':prediction['image_path'],'frame':frame,'image_size':prediction['image_size'],
                         'boxes':boxes,'ignored':ignored[frame],'split':'external_frozen','exhaustive':True})
            predictions.append({'image_path':prediction['image_path'],'boxes':[c['box'] for c in prediction['candidates']],
                                'scores':[1.]*len(prediction['candidates']),'native_per_box_scores_absent':True,
                                'native_seconds':prediction['seconds'],'rejected':prediction['perception']['model_response_rejected']})
    with (RUN/'data/external_frozen.jsonl').open('w') as f:
        for row in rows:f.write(json.dumps(row,separators=(',',':'))+'\n')
    write_json(RUN/'data/A0_external_predictions.json',predictions)
    write_json(RUN/'data/external_protocol.json',{'status':'FROZEN_BEFORE_HEAD_PILOT','selection':'first3 distinct raw-source Sports videos / distinct Animal categories in SHA256 order; completed-cache eligibility, never head metrics',
        'jobs':selected,'frames':len(rows),'index_SHA256':digest(RUN/'data/external_frozen.jsonl'),
        'A0_model_step':18227,'A0_model_SHA':'7c4a81349e333a304e32f42cf1999db289802d8c31eba415b85827038d14801f',
        'A0_AP_allowed':False,'do_not_tune_on_external_scope':True,
        'prior_research_use':'Previously measured for native G2/tracking; video-disjoint from G2/head TRAIN. This is a frozen comparative audit, not untouched final TEST.'})
    print(json.dumps({'selected':[(j['dataset'],j['sequence'],j['frames']) for j in selected],'frames':len(rows)}))


if __name__=='__main__':main()
