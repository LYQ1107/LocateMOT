"""Compare native/projected appearance and learned retrieval on frozen fit states."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import select
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def measure(dataset, checkpoint, output, arm):
    import torch
    import torch.nn.functional as F
    from locatemot.groundjev.compact import CompactCore
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = CompactCore(saved['architecture'], semantic=saved['semantic'],
        input_evidence=saved.get('input_evidence', 'geometry_appearance_language'))
    model.load_state_dict(saved['state_dict']); model.eval()
    rows = [r for r in dataset.rows if r['task']=='identity' and r['label']=='EXISTING']
    assert len(rows)==1250 and len({r['id'] for r in rows})==1250
    totals=defaultdict(Counter); cases=[]; actions=Counter()
    with torch.no_grad():
        for number, row in enumerate(rows, 1):
            e=dataset.inputs(row); slot=row['slot']; m=len(e['history'])
            correct=max(range(len(row['target'])),key=row['target'].__getitem__)
            assert correct<m
            native=F.normalize(e['candidates'][slot],dim=-1) @ F.normalize(e['history'],dim=-1).reshape(-1,2560).T
            native=native.reshape(m,3).mean(-1)
            current=model.visual(e['candidates'][slot:slot+1])[0]
            historical=model.visual(e['history'])
            projected=F.normalize(current,dim=-1) @ F.normalize(historical,dim=-1).reshape(-1,model.width).T
            projected=projected.reshape(m,3).mean(-1)
            y=model(e)['match'][slot]
            methods=dict(native_cosine=native,projected_cosine=projected,IoU=e['pair_geometry'][slot,:,0],learned_existing_logits=y[:m])
            sample=dict(id=row['id'],unit=row['unit'],group=row['group'],slot=slot,existing_options=m,correct_alias_index=correct)
            for name,scores in methods.items():
                assert torch.isfinite(scores).all()
                order=sorted(range(m),key=lambda j:(-float(scores[j]),j)); rank=order.index(correct)+1
                totals[name]['questions']+=1; totals[name]['top1_correct']+=order[0]==correct
                totals[name]['rank_sum']+=rank; totals[name]['reciprocal_rank_sum']+=1/rank
                totals[name]['uniform_chance_sum']+=1/m
                sample[name+'_target_rank']=rank
            predicted=int(y.argmax()); name='EXISTING' if predicted<m else ['NEW','IGNORE','NO_MATCH'][predicted-m]
            actions[name]+=1; sample['learned_full_action_correct']=predicted==correct
            sample['learned_full_target_probability']=float(y.softmax(-1)[correct])
            cases.append(sample)
            if number%250==0: print(json.dumps(dict(status='FROZEN_FIT_AFFINITY_DIAGNOSTIC',arm=arm,checked=number,total=len(rows))),flush=True)
    summary={}
    for name,values in totals.items():
        summary[name]=dict(questions=values['questions'],top1_correct=values['top1_correct'],
            top1_accuracy=values['top1_correct']/len(rows),mean_target_rank=values['rank_sum']/len(rows),
            mean_reciprocal_rank=values['reciprocal_rank_sum']/len(rows),mean_uniform_chance=values['uniform_chance_sum']/len(rows))
    result=dict(status='FROZEN_FIT_NATIVE_PROJECTED_AND_LEARNED_AFFINITY_MEASURED',arm=arm,
        checkpoint=str(checkpoint),existing_identity_questions=len(rows),methods=summary,
        learned_full_action_counts=dict(actions),learned_full_action_correct=sum(c['learned_full_action_correct'] for c in cases),
        learned_mean_correct_alias_probability=sum(c['learned_full_target_probability'] for c in cases)/len(cases),
        scope='Declared fit-source existing-identity questions on identical frozen original-actor histories; not held-out accuracy or a causal trajectory intervention.',
        runtime_or_training_changed=False,official_tracking_gain_claimed=False,full_goal_complete=False)
    (output/(arm+'_summary.json')).write_text(json.dumps(result,indent=2)+'\n')
    with (output/(arm+'_cases.jsonl')).open('w') as file:
        for case in cases: file.write(json.dumps(case)+'\n')
    print(json.dumps(result),flush=True)
    return result


def main():
    import torch
    from tools.groundjev.compact_question_data import CompactQuestionData
    parser=argparse.ArgumentParser(); parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,help='explicit completed checkpoint, without a training wait')
    parser.add_argument('--arm',default='after'); args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(4); dataset=CompactQuestionData(args.data)
    if args.checkpoint:
        args.out.mkdir(parents=True,exist_ok=False)
        measure(dataset,args.checkpoint,args.out,args.arm)
        assert not torch.cuda.is_initialized()
        return
    plan=dataset.plan; training=Path(plan['training_output'])
    registered=json.loads((training/'owner_state.json').read_text())
    proc=Path('/proc')/str(registered['owner_pid']); descriptor=None
    if proc.exists():
        assert (proc/'stat').read_text().rsplit(')',1)[1].split()[19]==registered['owner_starttick']
        assert (proc/'cwd').resolve()==ROOT
        descriptor=os.pidfd_open(registered['owner_pid'])
    else: assert registered['status']=='BROADER_SHARED_COMPACT_TYPED_QUESTION_TRAINING_COMPLETE'
    args.out.mkdir(parents=True,exist_ok=False)
    state=dict(status='FROZEN_FIT_AFFINITY_DIAGNOSIS_BEFORE_THEN_AFTER_ORIGINAL_TRAINING',
        owner_pid=os.getpid(),owner_starttick=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        original_training_PID=registered['owner_pid'],original_training_starttick=registered['owner_starttick'],
        physical_GPUs=[],completed=[],full_goal_complete=False)
    def save():
        (args.out/'owner_state.json').write_text(json.dumps(state,indent=2)+'\n'); print(json.dumps(state),flush=True)
    save()
    try:
        before=measure(dataset,ROOT/plan['initial_checkpoint'],args.out,'before'); state['completed'].append('before'); save()
        if descriptor is not None:
            state['status']='WAITING_EXACT_ORIGINAL_COMPACT_TRAINING_PIDFD'; save()
            poller=select.poll(); poller.register(descriptor,select.POLLIN); poller.poll(); os.close(descriptor)
        final=json.loads((training/'owner_state.json').read_text())
        assert final['status']=='BROADER_SHARED_COMPACT_TYPED_QUESTION_TRAINING_COMPLETE', final
        after=measure(dataset,training/'checkpoint.pt',args.out,'after')
        for method in ('native_cosine','IoU'): assert before['methods'][method]==after['methods'][method]
        state.update(status='FROZEN_FIT_AFFINITY_BEFORE_AFTER_COMPLETE',completed=['before','after'],
            learned_full_correct_before=before['learned_full_action_correct'],learned_full_correct_after=after['learned_full_action_correct'],
            frozen_native_cosine_and_geometry_identical=True,scope=after['scope'])
        save()
        assert sum(p.stat().st_size for p in args.out.iterdir() if p.is_file())<2*2**20
        assert not torch.cuda.is_initialized()
    except Exception as error:
        state.update(status='FROZEN_FIT_AFFINITY_DIAGNOSIS_FAILED',error=repr(error)); save(); raise


if __name__=='__main__': main()
