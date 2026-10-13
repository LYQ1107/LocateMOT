"""Repair observed lifecycle targets without changing any actual model history."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'external/OneJev')]


def main():
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from tools.groundjev.prepare_tracking_decisions import annotations,PROTOCOL
    from locatemot.groundjev.decision import overlap
    from train import common
    from qev.prompt import render_question
    receipts=[]
    for name in ('student_v3','pretrained_student_v3'):
        directory=ROOT/'outputs/groundjev/training_data'/name
        out=directory.with_name(name+'_lifecycle_fixed')
        if out.exists():
            if (out/'summary.json').exists():
                raise RuntimeError('Preserve a completed label repair')
            if (out/'train.jsonl').exists():
                (out/'train.jsonl').rename(out/'train_failed_alias_parse.jsonl')
        else:
            out.mkdir()
        rows=list(map(json.loads,(directory/'native_rollout.jsonl').read_text().splitlines()))
        targets,previous,anchors={},None,{}
        for native in rows:
            sequence=native['sequence']
            if sequence!=previous:
                previous=sequence;anchors={}
                pname='dancetrack.json' if sequence.startswith('dance') else 'mot17_half.json' if sequence.startswith('MOT17') else 'mot20_half.json'
                seq=next(s for s in json.loads((PROTOCOL/pname).read_text())['sequences'] if s['sequence']==sequence)
                gt=annotations(seq,{int(Path(f).stem) for f in seq['train_frames']})
            current=gt[native['frame']];candidates=native['candidates'];matched={}
            if current and candidates:
                similarity=np.array([[overlap(c['box'],g['box']) for g in current] for c in candidates])
                rr,cc=linear_sum_assignment(-similarity)
                matched={int(i):current[int(j)] for i,j in zip(rr,cc) if similarity[i,j]>=.5}
            detected={g['gt_id'] for g in matched.values()}
            for batch,start in enumerate(range(0,len(candidates),16)):
                ids=next(e['decision']['shortlisted_ids'] for e in native['state']['events'] if e.get('candidate')==start)
                targets[(sequence,native['frame'],batch)]={'ids':ids,'anchor_gt':dict(anchors),'detected':detected}
            for event in native['state']['events']:
                if event.get('action')=='NEW':
                    g=matched.get(event['candidate']);anchors[event['track_id']]=g['gt_id'] if g else None
        changed,checked,counts=0,0,Counter()
        with (out/'train.jsonl').open('w') as output:
            for line in (directory/'train.jsonl').read_text().splitlines():
                row=json.loads(line)
                if row['task']=='lifecycle':
                    checked+=1
                    batch=int(row['id'].split('_batch')[1].split('_')[0])
                    source=targets[(row['source'],row['supervision']['frame'],batch)]
                    history=row['state']['historical_tracks']
                    assert len(history)==len(source['ids'])
                    mapping={str(t['id']):raw for t,raw in zip(history,source['ids'])}
                    alias=row['id'].rsplit('_life_',1)[1]
                    owner=source['anchor_gt'].get(mapping[alias])
                    old=next(iter(row['target']))
                    if old in ('ACTIVE','REACTIVATE') and owner not in source['detected']:
                        assert owner is not None
                        row['target']={'LOST':1.};changed+=1
                        row['supervision']['lifecycle_label_repair']='GT-visible identity absent from supplied current proposals -> LOST'
                    new=next(iter(row['target']))
                    assert new not in ('ACTIVE','REACTIVATE') or owner in source['detected']
                rendered=render_question('q',common.build_question(row['question']))
                common.target_vector(row['target'],rendered.labels)
                counts[row['task']+':'+next(iter(row['target']))]+=1
                output.write(json.dumps(row)+'\n')
        summary=json.loads((directory/'summary.json').read_text())
        summary.update(source_original_curriculum=str(directory),lifecycle_targets_repaired=changed,
                       all_actor_states_inputs_and_ID_aliases_unchanged=True,
                       lifecycle_detection_consistency_checked_rows=checked)
        counts['max_options']=summary['counts']['max_options']
        if 'contaminated_history_observations' in summary['counts']:
            counts['contaminated_history_observations']=summary['counts']['contaminated_history_observations']
        summary['counts']=dict(counts)
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        receipts.append({'source':name,'changed_lifecycle_targets':changed,'checked_lifecycle_rows':checked,
                         'original_curriculum_preserved':True,'all_states_and_non_lifecycle_targets_unchanged':True})
    assert [r['changed_lifecycle_targets'] for r in receipts]==[164,39]
    (ROOT/'outputs/groundjev/validation/lifecycle_current_proposal_regression.json').write_text(json.dumps(receipts,indent=2)+'\n')
    print(json.dumps(receipts),flush=True)


if __name__=='__main__': main()
