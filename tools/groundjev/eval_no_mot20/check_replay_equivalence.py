"""16 real cached-frame regression for visualization-free replay; no science claim."""
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tools.groundjev.eval_no_mot20.common import RUN,GROUND,save,load


def main():
    source=GROUND/'outputs/groundjev/heldout/g2_domain_dancetrack_gpu5_v1/G2_CLEAN_V2/dancetrack0004_val1203'
    rows=[]
    with (source/'frame_decisions.jsonl').open() as handle:
        for _ in range(16):rows.append(json.loads(next(handle)))
    out=RUN/'engineering_checks/simple_no_video16';out.mkdir(parents=True,exist_ok=True)
    manifest={'task':'mot','dataset':'dancetrack','sequence':'dancetrack0004',
        'images':[x['image_path'] for x in rows],'frame_numbers':[x['frame'] for x in rows],
        'fps':20.,'vocabulary':['person'],'expression':None,
        'scope':'FUNCTIONAL_SUBSET_16_FRAME_REGRESSION_NOT_INDEPENDENT_EVALUATION'}
    save(out/'manifest.json',manifest)
    (out/'candidates.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    for name,flags in [('with_video',[]),('no_video',['--no-video'])]:
        if (out/name/'result.json').exists():continue
        command=[sys.executable,str(ROOT/'tools/groundjev/run_video.py'),'--task','mot',
            '--video',str(out/'manifest.json'),'--backend','simple','--candidate-replay',str(out/'candidates.jsonl'),
            '--memory-export-interval','0','--memory-checkpoint-format','binary',
            '--grounding-parser','official_scoring','--out',str(out/name)]+flags
        with (out/(name+'.log')).open('w') as log:
            subprocess.run(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),
                           stdout=log,stderr=subprocess.STDOUT,check=True)
    a=[json.loads(l) for l in (out/'with_video/frame_decisions.jsonl').open()]
    b=[json.loads(l) for l in (out/'no_video/frame_decisions.jsonl').open()]
    assert len(a)==len(b)==16
    for x,y in zip(a,b):
        for key in ('frame','image_path','candidates','outputs','state'):assert x[key]==y[key],(x['frame'],key)
    assert (out/'with_video/predictions.txt').read_bytes()==(out/'no_video/predictions.txt').read_bytes()
    assert load(out/'no_video/result.json')['grounding_requests_in_this_run']==0
    save(out/'receipt.json',{'status':'REAL_CANDIDATE_16_FRAME_VISUALIZATION_EQUIVALENCE_PASSED',
        'frames':16,'outputs_ID_actions_states_exact':True,'G2_calls':0,'functional_only':True,'source':str(source)})
    print('REAL_16_FRAME_EQUIVALENCE_PASSED',flush=True)


if __name__=='__main__':main()
