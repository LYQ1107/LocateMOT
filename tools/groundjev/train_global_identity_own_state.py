"""Real short-sequence own states; gradient accumulation without temporal BPTT."""
import argparse
from collections import Counter
import gzip
import json
import math
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def main():
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from locatemot.groundjev.global_identity_runtime import GlobalIdentityDecision,GlobalIdentityTracker
    from locatemot.groundjev.global_identity_core import physical_inputs,physical_loss
    from tools.groundjev.global_identity_data import training_frames
    from tools.groundjev.global_identity_data_v2 import entity_options
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);args=parser.parse_args()
    settings=json.loads(args.plan.read_text());plan=json.loads((ROOT/settings['physical_plan']).read_text())
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(4);torch.manual_seed(0);torch.use_deterministic_algorithms(True)
    audit=json.loads((ROOT/settings['label_audit']).read_text());counts_in=audit['totals']
    birth_weight=math.sqrt(counts_in['MATCH_supported_proposal']/counts_in['NEW_first_effective_birth_proposal'])
    actor=GlobalIdentityDecision(settings['initial_checkpoint'],memory_policy='assigned');model=actor.model
    initial={k:v.clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    source=Path(plan['immutable_source_root']);units=json.loads((source/plan['source_index']).read_text())['units']
    calibration=set(sorted(units)[::5]);out=Path(settings['output']);out.mkdir(parents=True,exist_ok=False)
    counts=Counter();first=[];steps=0;phase_records=[]
    with (out/'log.jsonl').open('w') as log,gzip.open(out/'own_state_frame_proof.jsonl.gz','wt') as proof:
        for length in settings['episode_lengths']:
            tracker=None;unit=None;position=0;anchors={};ever_bound=set();episode_frames=0;episode_labels=0
            for frame in training_frames(plan):
                if unit!=frame['unit'] or episode_frames==0:
                    unit=frame['unit'];tracker=GlobalIdentityTracker(actor);anchors={};ever_bound=set()
                    position=frame['manifest']['frame_numbers'].index(frame['frame'])
                    actual_length=min(length,len(frame['manifest']['images'])-position)
                    optimizer.zero_grad();episode_loss=0.;episode_labels=0;episode_frames=0
                live=list(tracker.tracks.values());before=dict(anchors)
                e=physical_inputs(frame['image_size'],frame['candidates'],live,frame['timestamp'],32)
                options,label_counts=entity_options(frame['candidates'],frame['labels']['support'],frame['labels']['full_truth'],
                    live,anchors,ever_bound)
                labels=dict(frame['labels'],action_options=options)
                actor.model.eval()
                with Image.open(frame['source_row']['image_path']) as raw:
                    image=raw.convert('RGB')
                outputs,state=tracker.step(image,frame['candidates'],frame['timestamp'],dict(task='mot',vocabulary=['person'],expression=None))
                by_id={t.track_id:i for i,t in enumerate(live)};write_rows=[];write_cols=[];write_targets=[]
                for event in state['events']:
                    if 'candidate' not in event:continue
                    i=event['candidate'];tid=event.get('track_id');supported=labels['support'][i]
                    gt=labels['full_truth'][supported[0]] if len(supported)==1 else None
                    if event['action']=='NEW':
                        anchors[tid]=gt['identity'] if gt is not None else None
                        if gt is not None:ever_bound.add(gt['identity'])
                    elif tid in by_id and gt is not None and before.get(tid) is not None:
                        target=float(before[tid]==gt['identity'] and gt['visibility']>=.6)
                        counts['own_actual_assigned_write_labels']+=1
                        counts['own_actual_wrong_long_writes']+=event['memory_action']=='UPDATE' and before[tid]!=gt['identity']
                        if unit not in calibration:
                            write_rows.append(i);write_cols.append(by_id[tid]);write_targets.append(target)
                model.train();prediction=model(e);loss,metrics=physical_loss(prediction,e,labels)
                newborn=[i for i,choices in enumerate(options) if choices==[len(live)]]
                if newborn:
                    extra=sum(torch.logsumexp(prediction['logits'][i],0)-prediction['logits'][i,len(live)] for i in newborn)
                    loss=loss+(birth_weight-1)*extra/max(metrics['supervised_actions'],1)
                memory_loss=loss*0
                if write_rows:
                    confidence=[];margins=[];extra=[]
                    for i,c in zip(write_rows,write_cols):
                        scores=prediction['logits'][i];rivals=torch.cat([scores[:c],scores[c+1:]])
                        confidence.append(scores.softmax(0)[c]);margins.append(scores[c]-rivals.max())
                        decision=actor.cache['decisions'][i]
                        original_update=decision.get('observation_update_probability',decision['update_probability'])
                        extra.append(torch.stack([torch.tensor(original_update),
                            e['history_geo'][c,4],e['history_geo'][c,5],
                            F.cosine_similarity(e['candidates'][i],e['history'][c,2],dim=0)]))
                    logits=model.assigned_write(prediction,torch.tensor(write_rows),torch.tensor(write_cols),
                        torch.stack(confidence),torch.stack(margins),torch.stack(extra))
                    memory_loss=F.binary_cross_entropy_with_logits(logits,torch.tensor(write_targets))
                    loss=loss+settings['memory_loss_weight']*memory_loss
                assert torch.isfinite(loss);(loss/actual_length).backward()
                episode_loss+=float(loss.detach());episode_labels+=metrics['supervised_actions'];episode_frames+=1
                counts['frames']+=1;counts['candidate_exposures']+=len(frame['candidates']);counts.update(label_counts)
                counts['actual_own_state_new_births']+=sum(e.get('action')=='NEW' for e in state['events'])
                proof.write(json.dumps(dict(length=length,unit=unit,frame=frame['frame'],
                    prior_predicted_IDs=list(by_id),posterior_predicted_IDs=state['persistent_ids'],
                    actions=[{k:v for k,v in event.items() if k in ('candidate','track_id','action','memory_action')} for event in state['events']],
                    supervision_outside_model_input=True,teacher_history_replaced=False))+'\n')
                if episode_frames==actual_length:
                    gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip']))
                    assert math.isfinite(gradient)
                    if gradient>0:optimizer.step();steps+=1
                    counts['episodes:'+str(length)]+=1
                    record=dict(length=length,actual_episode_frames=episode_frames,step=steps,unit=unit,
                        mean_loss=episode_loss/episode_frames,gradient_norm=gradient,supervised_actions=episode_labels)
                    if len(first)<5:first.append(record)
                    if steps<=5 or steps%25==0:log.write(json.dumps(record)+'\n');log.flush();print(json.dumps(record),flush=True)
                    episode_frames=0
            assert episode_frames==0
            phase_records.append(dict(length=length,steps=steps,frames=counts['frames']))
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),steps=steps,torch_rng=torch.get_rng_state(),
                settings=settings,completed_length=length,counts=dict(counts)),out/'latest_training_state.pt')
    assert counts['frames']==3*1056
    model.eval();payload=torch.load(settings['initial_checkpoint'],map_location='cpu',weights_only=True)
    payload.update(arm='A4_own_state_v2',state_dict=model.state_dict(),own_state_settings=settings,
        own_state_actual_updates=steps,write_calibration_after_own_state='Not revalidated after joint own-state updates; sigmoid is uncalibrated until separately measured.')
    torch.save(payload,out/'checkpoint.pt')
    changed={k:int((v!=initial[k]).sum()) for k,v in model.state_dict().items()}
    assert changed['pair.0.weight'] and changed['actions.0.weight'] and changed['write.0.weight']
    report=dict(status='REAL_8_16_32_FRAME_MODEL_IN_LOOP_TRAINING_COMPLETE',actual_updates=steps,counts=dict(counts),
        phase_records=phase_records,first5=first,changed_elements=changed,birth_cost_weight=birth_weight,
        all_original_frames_processed_per_length=1056,model_own_predicted_histories_used=True,
        GT_corrected_predicted_history=False,temporal_BPTT=False,
        gradient_protocol='Per-frame detached gradients accumulated under unchanged episode weights; one optimizer update at episode end.',
        all_tasks_same_final_checkpoint=str(out/'checkpoint.pt'),new_GPU_processes=0,new_feature_bank_bytes=0,
        official_tracking_gain='pending fixed five-scope evaluation',full_goal_complete=False)
    (out/'training_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
