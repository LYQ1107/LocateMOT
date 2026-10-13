"""One blocking owner for finite, preregistered own-history decision adaptation."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]


def main():
    out=ROOT/'outputs/groundjev/training/onejev_own_history_v3_500'
    out.mkdir(parents=True,exist_ok=True)
    state_file=out/'owner_state.json'
    if state_file.exists():
        raise RuntimeError('Preserve an existing formal training owner and checkpoint')
    expected=ROOT/'outputs/groundjev/training_data/pretrained_student_v3'
    owners=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            argv=p.joinpath('cmdline').read_bytes().decode().split('\0')[:-1]
            if str(p.joinpath('cwd').resolve())!=str(ROOT) or 'tools/groundjev/prepare_student_rollout.py' not in argv: continue
            if '--out' in argv and (ROOT/argv[argv.index('--out')+1]).resolve()==expected:
                owners.append(int(p.name))
        except OSError: pass
    state={'status':'WAITING_REAL_PRETRAINED_STUDENT_ROLLOUT','producer_pids':owners,'owner_pid':os.getpid(),
           'physical_gpus':[4],'optimizer_step_budget':500,'seed':0,'identity_row_repeat_weight':4,
           'scope':'own-model-history_decision_adaptation_with_declared_GT_proposals; actual_DLM_proposal_stage_pending'}
    def save():
        state_file.write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps(state),flush=True)
    save()
    while any((Path('/proc')/str(pid)/'stat').exists() and (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z' for pid in owners):
        time.sleep(10)
    for directory in [expected,ROOT/'outputs/groundjev/training_data/student_v3']:
        receipt=directory/'summary.json'
        if not receipt.exists() or json.loads(receipt.read_text())['status']!='REAL_MODEL_HISTORY_CURRICULUM_COMPLETE_NOT_GROUND_DETECTOR_ROLLOUT':
            raise RuntimeError('A real rollout producer failed; do not train from incomplete units')
    curriculum=ROOT/'outputs/groundjev/training_data/mixed_student_v3'
    curriculum.mkdir(parents=True,exist_ok=True)
    counts,unique,total={},0,0
    with (curriculum/'train.jsonl').open('w') as output:
        for directory in [expected,ROOT/'outputs/groundjev/training_data/student_v3']:
            for line in (directory/'train.jsonl').read_text().splitlines():
                row=json.loads(line);unique+=1
                repeats=4 if row['task']=='identity' else 1
                for repeat in range(repeats):
                    copied=dict(row);copied['id']=directory.name+'_'+row['id']+f'_repeat{repeat}'
                    output.write(json.dumps(copied)+'\n');total+=1
                    counts[row['task']]=counts.get(row['task'],0)+1
    (curriculum/'summary.json').write_text(json.dumps({'unique_real_supervised_rows':unique,'training_rows_with_explicit_repeat_sampling':total,
        'family_counts':counts,'identity_repeat_weight':4,'seed':0,'actor_state_corrected_by_GT':False,
        'candidate_source':'legal_train_GT_plus_explicit_fragments','GroundAnything_prediction_claim':False},indent=2)+'\n')
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'external/OneJev')
    config=ROOT/'configs/groundjev/onejev_own_history_v3_500.json'
    settings={'model':str(ROOT/'outputs/groundjev/training/onejev_teacher_check_100/final'),
        'train':str(curriculum/'train.jsonl'),'media_root':str(ROOT),'name':out.name,'runs_dir':str(out.parent),
        'dtype':'bfloat16','multimodal':True,'full_finetune':True,'train_vision':False,'lr':5e-6,
        'epochs':1,'max_steps':500,'max_len':32768,'batch_tokens':32768,'max_rows':1,'accum_rows':4,
        'brier_weight':1.,'warmup_frac':.03,'weight_decay':0.,'max_grad_norm':1.,'augment':True,
        'pad_options':0.,'seed':0,'log_every':25,'save_every':125}
    # JSON is valid YAML for the unmodified official configuration reader.
    config.write_text(json.dumps(settings,indent=2)+'\n')
    command=[str(ROOT/'.venv-onejev-py310/bin/python'),'-m','train.sft','--config',str(config)]
    def run(stage,args):
        state['status']=stage;save()
        with (out/(stage.lower()+'.log')).open('w') as log:
            process=subprocess.Popen(args,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in process.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
            code=process.wait()
        if code: raise RuntimeError(f'{stage} failed with{code}; preserve all prior artifacts')
    state.update(unique_rows=unique,training_rows=total,initialization=settings['model'],
                 optimizer_reset='explicit new curriculum stage; weights continue from100-step checkpoint',
                 maximum_context='32768 measured-plan cap; no rows may be dropped')
    run('PLANNING_COMPLETE_CURRICULUM',command+['--plan-only'])
    text=(out/'planning_complete_curriculum.log').read_text()
    if '0 rows over max-len 32768' not in text:
        raise RuntimeError('Curriculum does not fit the registered context; resolve the first oversized unit without dropping rows')
    run('RUNNING_FINITE500_STEP_OFFICIAL_FULL_SFT',command)
    logs=[json.loads(x) for x in (out/'log.jsonl').read_text().splitlines()]
    done=next(x for x in logs if x.get('event')=='done')
    if done['steps']!=500: raise RuntimeError('Finite formal adaptation stopped short of500steps')
    state.update(status='FORMAL_OWN_HISTORY500_STEP_TRAINING_COMPLETE',steps=500,done=done,
                 before_after_official_metrics='pending checkpoint reload and same-candidate evaluation')
    save()


if __name__=='__main__': main()
