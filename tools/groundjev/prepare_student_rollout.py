"""Real OneJev histories with legal train proposals and labeled box fragments.

This curriculum isolates decision learning: proposals are teacher GT plus
explicit synthetic negatives; histories are exclusively the model's states.
An explicit native_ground mode uses all actual DLM proposals, with no GT
proposal substitution or synthetic fragments and no corrected actor states.
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


def registered_native_candidates(path):
    rows = {}
    for line in Path(path).open():
        row = json.loads(line)
        key = row['sequence'],row['frame']
        assert key not in rows
        rows[key] = {k:row[k] for k in ('image_path','candidates','perception')}
    return rows


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
    parser.add_argument('--candidate-source', choices=['gt_fragments', 'native_ground'], default='gt_fragments')
    parser.add_argument('--native-candidate-replay',type=Path)
    parser.add_argument('--output-byte-budget',type=int)
    args=parser.parse_args()
    if args.native_candidate_replay and args.candidate_source!='native_ground':
        parser.error('Native replay requires the actual native-ground candidate mode')
    replay = registered_native_candidates(args.native_candidate_replay) if args.native_candidate_replay else None
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('preserve prior model rollout; choose a fresh curriculum path')
    args.out.mkdir(parents=True,exist_ok=True)
    class Recorder(OneJevDecision):
        def accepted_batch(self,image,group,ranked,timestamp,specification):
            self.captured.append((copy.deepcopy(group),[copy.copy(t) for t in ranked],timestamp,specification))
    native_ground = args.candidate_source == 'native_ground'
    model=Recorder(args.checkpoint,evidence_format='v3_crop_motion',adaptive_context=native_ground)
    if native_ground and replay is None:
        from locatemot.groundjev.perception import GroundAnythingAdapter, GroundingProtocolError
        frontend=GroundAnythingAdapter(parser_policy='official_scoring',decoder_policy='tracking_dense')
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
            if replay is not None:
                frames = [frame for sequence,frame in replay if sequence==name]
                assert len(frames)==16 and all(frame in frame_names for frame in frames)
            else:
                score,stride,frames=mine_clips(sorted(frame_names),gt,2,16)[0]
            if name=='dancetrack0001' and replay is None:
                previous=json.loads((ROOT/'outputs/groundjev/training_data/teacher_v1/summary.json').read_text())
                frames=next(c['frames'] for c in previous['clips'] if c['sequence']==name and c['real_reappearances_after_missing_observation']>0)
            tracker=OnlineTracker(model)
            anchors,last_observation,aliases={},{},{}
            rng=random.Random(clip_index*1009)
            alias_pool=list(range(1,100001 if native_ground else 577));rng.shuffle(alias_pool)
            clip_rows=0
            for frame in frames:
                path=Path(sequence['image_dir'])/frame_names[frame]
                with Image.open(path) as source: image=source.convert('RGB')
                targets=gt[frame]
                ordered=sorted(targets,key=lambda c:(-max((overlap(c['box'],o['box']) for o in targets if o['gt_id']!=c['gt_id']),default=0.),c['visibility'],c['gt_id']))
                perception=None
                if replay is not None:
                    saved = replay[name,frame]
                    assert Path(saved['image_path']).resolve()==path.resolve()
                    candidates,perception=copy.deepcopy(saved['candidates']),copy.deepcopy(saved['perception'])
                    counts['actual_native_candidates']+=len(candidates)
                    counts['replayed_native_frames']+=1
                    counts['rejected_perception_frames']+=bool(perception and perception.get('model_response_rejected'))
                elif native_ground:
                    try:
                        candidates,perception=frontend(image,'mot',['person'],image_path=path)
                    except GroundingProtocolError as error:
                        candidates=[];perception=dict(error.metadata,model_response_rejected=True,
                            rejected_output_policy='empty native candidates; training frame and all GT supervision retained')
                        counts['rejected_perception_frames']+=1
                    counts['actual_native_candidates']+=len(candidates)
                else:
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
                native.write(json.dumps({'sequence':name,'frame':frame,'image_path':str(path),'candidates':candidates,'outputs':outputs,'state':state,'perception':perception})+'\n');native.flush()
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
                    if 'empty' in request.questions:
                        labels['empty']=('yes' if targets else 'no','empty_image_target_presence')
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
                    present={g['gt_id'] for g in matched.values()}
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
                                'candidate_source':'actual_native_GroundAnything_DLM_proposals' if native_ground else 'legal_training_GT_proposals_and_explicit_synthetic_fragments',
                                'actor_state_corrected_by_GT':False,'GT_labels_outside_model_state':True,'evidence_format':'v3_crop_motion'}}
                        training.write(json.dumps(row)+'\n')
                        counts[kind+':'+target]+=1;clip_rows+=1;total_rows+=1
                        counts['max_options']=max(counts['max_options'],len(rendered.labels))
                    global_index+=len(group)
                assert global_index==len(candidates), 'Every actual candidate must have one accepted actor group'
                training.flush()
                if args.output_byte_budget:
                    exported_bytes = training.tell()+native.tell()+sum(p.stat().st_size for p in (args.out/'evidence').glob('*.png'))
                    if exported_bytes>args.output_byte_budget:
                        raise RuntimeError(f'Rollout export exceeded registered {args.output_byte_budget} byte budget; preserve completed prefix')
                for event in state['events']:
                    if 'candidate' not in event or 'track_id' not in event: continue
                    truth=matched.get(event['candidate']);gid=truth['gt_id'] if truth else None
                    if event['action']=='NEW': anchors[event['track_id']]=gid
                    last_observation[event['track_id']]=gid
                print(json.dumps({'sequence':name,'frame':frame,'tracks':len(outputs),'training_rows':total_rows}),flush=True)
            clips.append({'sequence':name,'frames':frames,'rows':clip_rows,'last_runtime_id':tracker.next_id-1,'own_model_states':True})
    summary={'status':'ACTUAL_NATIVE_GROUND_OWN_HISTORY_CURRICULUM_COMPLETE' if native_ground else 'REAL_MODEL_HISTORY_CURRICULUM_COMPLETE_NOT_GROUND_DETECTOR_ROLLOUT',
        'native_candidate_replay':str(args.native_candidate_replay) if replay is not None else None,
        'new_ground_requests':False if replay is not None else native_ground,
        'registered_output_byte_budget':args.output_byte_budget,
        'rows':total_rows,'seed':0,'checkpoint':str(args.checkpoint),'clips':clips,'counts':dict(counts),
        'evidence_png_bytes':sum(p.stat().st_size for p in (args.out/'evidence').glob('*.png')),
        'actor_GT_ID_inputs':False,'actor_state_corrected_by_GT':False,'validation_or_test_supervision_used':False,
        'actual_native_GroundAnything_proposals':native_ground,'synthetic_fragments':not native_ground,
        'next_stage':'train on joint teacher/model-history mixture; evaluate unchanged full legal protocols'}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
