"""Real OneJev histories with legal train proposals and labeled box fragments.

This curriculum isolates decision learning: proposals are teacher GT plus
explicit synthetic negatives; histories are exclusively the model's states.
Actual GroundAnything-proposal rollout remains a separate next stage.
"""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'external/OneJev')]


def main():
    import numpy as np
    from PIL import Image
    from scipy.optimize import linear_sum_assignment
    from locatemot.groundjev.decision import OneJevDecision,overlap
    from locatemot.groundjev.tracker import OnlineTracker
    from tools.groundjev.prepare_tracking_decisions import annotations,mine_clips,PROTOCOL
    from train import common
    from qev.prompt import render_question
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/groundjev/training_data/student_v3')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'outputs/groundjev/training/onejev_teacher_check_100/final')
    args=parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('preserve prior model rollout; choose a fresh curriculum path')
    args.out.mkdir(parents=True,exist_ok=True)
    class Recorder(OneJevDecision):
        def build_request(self,image,group,ranked,timestamp,specification):
            request,media=super().build_request(image,group,ranked,timestamp,specification)
            self.captured.append((copy.deepcopy(group),[copy.copy(t) for t in ranked],timestamp,specification))
            return request,media
    model=Recorder(args.checkpoint,evidence_format='v3_crop_motion')
    builder=OneJevDecision.__new__(OneJevDecision)
    builder.evidence_format='v3_crop_motion'
    counts,clips=Counter(),[]
    total_rows=0
    with (args.out/'train.jsonl').open('w') as training,(args.out/'native_rollout.jsonl').open('w') as native:
        for clip_index,(protocol_file,name) in enumerate([('mot17_half.json','MOT17-02-FRCNN'),('dancetrack.json','dancetrack0001'),('mot20_half.json','MOT20-01')]):
            protocol=json.loads((PROTOCOL/protocol_file).read_text())
            sequence=next(s for s in protocol['sequences'] if s['sequence']==name)
            frame_names={int(Path(f).stem):f for f in sequence['train_frames']}
            gt=annotations(sequence,set(frame_names))
            score,stride,frames=mine_clips(sorted(frame_names),gt,2,16)[0]
            if name=='dancetrack0001':
                previous=json.loads((ROOT/'outputs/groundjev/training_data/teacher_v1/summary.json').read_text())
                frames=next(c['frames'] for c in previous['clips'] if c['sequence']==name and c['real_reappearances_after_missing_observation']>0)
            tracker=OnlineTracker(model)
            anchors,last_observation,aliases={},{},{}
            rng=random.Random(clip_index*1009)
            alias_pool=list(range(1,577));rng.shuffle(alias_pool)
            clip_rows=0
            for frame in frames:
                path=Path(sequence['image_dir'])/frame_names[frame]
                with Image.open(path) as source: image=source.convert('RGB')
                targets=gt[frame]
                ordered=sorted(targets,key=lambda c:(-max((overlap(c['box'],o['box']) for o in targets if o['gt_id']!=c['gt_id']),default=0.),c['visibility'],c['gt_id']))
                candidates=[{'box':list(c['box']),'label':'person','native_score':None,
                             'source':'legal_train_gt_proposal'} for c in ordered[:32]]
                for c in ordered[:2]:
                    x1,y1,x2,y2=c['box'];cx,cy=(x1+x2)/2,(y1+y2)/2
                    candidates.append({'box':[cx-(x2-x1)*.175,cy-(y2-y1)*.175,cx+(x2-x1)*.175,cy+(y2-y1)*.175],
                        'label':'person','native_score':None,'source':'synthetic_box_fragment_training_augmentation'})
                rng.shuffle(candidates)
                matched={}
                if candidates and targets:
                    similarities=np.array([[overlap(c['box'],g['box']) for g in targets] for c in candidates])
                    rr,cc=linear_sum_assignment(-similarities)
                    matched={int(i):targets[int(j)] for i,j in zip(rr,cc) if similarities[i,j]>=.5}
                prior_ids=set(tracker.tracks)
                before=dict(anchors)
                model.captured=[]
                outputs,state=tracker.step(image,candidates,frame/sequence['fps'],{'task':'mot','vocabulary':['person'],'expression':None})
                native.write(json.dumps({'sequence':name,'frame':frame,'image_path':str(path),'candidates':candidates,'outputs':outputs,'state':state})+'\n');native.flush()
                global_index=0
                for batch,(group,ranked,timestamp,specification) in enumerate(model.captured):
                    for t in ranked:
                        if t.track_id not in aliases:
                            aliases[t.track_id]=alias_pool[len(aliases)]
                    shadow=[]
                    for t in ranked:
                        clone=copy.copy(t);clone.track_id=aliases[t.track_id];shadow.append(clone)
                    request,media=builder.build_request(image,group,shadow,timestamp,specification)
                    files=[{'type':'image','path':str(path),'w':image.width,'h':image.height}]
                    for index,m in enumerate(media[1:],start=2):
                        directory=args.out/'evidence';directory.mkdir(exist_ok=True)
                        file=directory/f'{name}_{frame:06d}_batch{batch}_image{index}.png'
                        m['image'].save(file)
                        files.append({'type':'image','path':str(file),'w':m['image'].width,'h':m['image'].height})
                    labels={}
                    for local,candidate in enumerate(group):
                        truth=matched.get(global_index+local)
                        gid=truth['gt_id'] if truth else None
                        correct=[t for t in ranked if gid is not None and before.get(t.track_id)==gid]
                        known=[tid for tid in prior_ids if gid is not None and before.get(tid)==gid]
                        if truth is None: target='IGNORE'
                        elif correct: target=str(aliases[min(correct,key=lambda t:t.track_id).track_id])
                        elif known: target='NO_MATCH'
                        else: target='NEW'
                        labels[f'match_{local}']=(target,'identity')
                        reliable=truth is not None and truth['visibility']>=.6 and target!='NO_MATCH'
                        labels[f'update_{local}']=('yes' if reliable else 'no','memory_update')
                        level='0' if truth is None else '1' if not reliable else '3' if truth['visibility']>=.85 else '2'
                        labels[f'reliability_{local}']=(level,'reliability')
                        if correct and local==0:
                            question=request.questions[f'match_{local}'].model_copy(deep=True)
                            for t in correct: question.criteria.pop(str(aliases[t.track_id]))
                            request.questions[f'withheld_{local}']=question
                            labels[f'withheld_{local}']=('NO_MATCH','withheld_correct_options')
                    present={g['gt_id'] for g in targets}
                    for t in ranked:
                        gid=before.get(t.track_id)
                        action='TERMINATE' if gid is None else 'REACTIVATE' if gid in present and t.lost_age else 'ACTIVE' if gid in present else 'LOST'
                        labels[f'life_{aliases[t.track_id]}']=(action,'lifecycle')
                        if gid is not None and last_observation.get(t.track_id)!=gid:
                            counts['contaminated_history_observations']+=1
                    for key,(target,kind) in labels.items():
                        question=request.questions[key]
                        rendered=render_question('q',question)
                        common.target_vector({target:1.},rendered.labels)
                        row={'id':f'{name}_{frame}_batch{batch}_{key}','split':'train','source':name,'task':kind,
                            'state':request.state,'question':question.model_dump(exclude_none=True),'target':{target:1.},'media':files,
                            'supervision':{'frame':frame,'clip_length':16,'state_source':'actual_model_rollout_with_permuted_identity_labels',
                                'candidate_source':'legal_training_GT_proposals_and_explicit_synthetic_fragments',
                                'actor_state_corrected_by_GT':False,'GT_labels_outside_model_state':True,'evidence_format':'v3_crop_motion'}}
                        training.write(json.dumps(row)+'\n')
                        counts[kind+':'+target]+=1;clip_rows+=1;total_rows+=1
                        counts['max_options']=max(counts['max_options'],len(rendered.labels))
                    global_index+=len(group)
                training.flush()
                for event in state['events']:
                    if 'candidate' not in event or 'track_id' not in event: continue
                    truth=matched.get(event['candidate']);gid=truth['gt_id'] if truth else None
                    if event['action']=='NEW': anchors[event['track_id']]=gid
                    last_observation[event['track_id']]=gid
                print(json.dumps({'sequence':name,'frame':frame,'tracks':len(outputs),'training_rows':total_rows}),flush=True)
            clips.append({'sequence':name,'frames':frames,'rows':clip_rows,'last_runtime_id':tracker.next_id-1,'own_model_states':True})
    summary={'status':'REAL_MODEL_HISTORY_CURRICULUM_COMPLETE_NOT_GROUND_DETECTOR_ROLLOUT',
        'rows':total_rows,'seed':0,'checkpoint':str(args.checkpoint),'clips':clips,'counts':dict(counts),
        'evidence_png_bytes':sum(p.stat().st_size for p in (args.out/'evidence').glob('*.png')),
        'actor_GT_ID_inputs':False,'actor_state_corrected_by_GT':False,'validation_or_test_supervision_used':False,
        'next_stage':'actual GroundAnything proposals and joint teacher/model-history mixture under the same evidence protocol'}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
