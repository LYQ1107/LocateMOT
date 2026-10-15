"""One blocking CPU owner validates both finite paired history-refresh plans."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import shutil

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--reuse-plans-from',type=Path)
    args=parser.parse_args()
    study=json.loads((args.study/'summary.json').read_text())
    assert study['status']=='MATCHED_HISTORY_REFRESH_CURRICULA_PREPARED_NOT_TRAINED'
    args.out.mkdir(parents=True,exist_ok=False)
    state={'status':'RUNNING_BOTH_HISTORY_REFRESH_OFFICIAL_CPU_PLANS',
        'owner_pid':os.getpid(),'model_or_GPU_loaded':False,'training_started':False,
        'study':str(args.study),'completed':{}}
    def save():
        (args.out/'owner_state.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps(state),flush=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PLAN_WORLD='1',WORLD_SIZE='1',
        PYTHONPATH=str(ROOT/'external/OneJev'))
    processes,logs={},{}
    if args.reuse_plans_from:
        state['reused_completed_CPU_plans_from']=str(args.reuse_plans_from)
        for name,config in study['configs'].items():
            prior=args.reuse_plans_from/(name+'.json')
            report=json.loads(prior.read_text())
            expected=json.loads(Path(config).read_text())
            assert report['status']=='OFFICIAL_CPU_PLAN_AND_FINITE_PREFIX_COMPLETE'
            assert report['config']['train_file']==expected['train']
            assert report['config']['model']==expected['model']
            assert report['config']['seed']==expected['seed']
            shutil.copy2(prior,args.out/prior.name)
        codes={name:0 for name in study['configs']}
    else:
        for name,config in study['configs'].items():
            settings=json.loads(Path(config).read_text())
            logs[name]=(args.out/(name+'.log')).open('w')
            command=[str(ROOT/'.venv-onejev-py310/bin/python'),
                'tools/groundjev/onejev_plan_with_exposure.py','--config',config,
                '--name',settings['name']+'_token_plan',
                '--plan-only','--expected-rows','2400','--paired-index-shuffle',
                '--receipt',str(args.out/(name+'.json'))]
            processes[name]=subprocess.Popen(command,cwd=ROOT,env=env,stdout=logs[name],stderr=subprocess.STDOUT)
        save()
        codes={name:child.wait() for name,child in processes.items()}
        for log in logs.values():
            log.close()
    state['return_codes']=codes
    if any(codes.values()):
        state['status']='HISTORY_REFRESH_CPU_PLAN_FAILED';save()
        raise RuntimeError('Preserve CPU logs and fix first failing unit without dropping rows')
    reports={name:json.loads((args.out/(name+'.json')).read_text()) for name in study['configs']}
    for name,report in reports.items():
        assert report['overlength_dropped_rows']==0 and report['actual_prefix_rows']==2400
        assert report['actual_prefix_task_counts']=={'mot':800,'rmot':800,'ovmot':800}
        state['completed'][name]={k:report[k] for k in ('max_actual_tokens','actual_prefix_rows',
            'actual_prefix_task_counts','actual_prefix_question_families','actual_prefix_tokens')}
    # Native plan_steps sorts the four micro-batches by length within each
    # step. Pair exact row-to-step membership; preserve that native ordering.
    signature=lambda r:sorted((p['step'],p['index'],p['id'],p['bucket'],p['family']) for p in r['prefix'])
    assert signature(reports['old'])==signature(reports['refreshed'])
    configs={name:json.loads(Path(path).read_text()) for name,path in study['configs'].items()}
    assert {k:v for k,v in configs['old'].items() if k not in ('train','name')}=={
        k:v for k,v in configs['refreshed'].items() if k not in ('train','name')}
    rows={name:[json.loads(line) for line in Path(config['train']).open()] for name,config in configs.items()}
    changed=0;media=set()
    for before,after in zip(rows['old'],rows['refreshed']):
        if before!=after:
            assert before['supervision']['tracking_task_bucket']=='mot'
            assert before['supervision']['source_row_id'].startswith('native_')
            assert before['task']==after['task'];changed+=1
        for row in (before,after):
            media.update(item['path'] for item in row.get('media',[]))
    assert changed==400 and len(rows['old'])==len(rows['refreshed'])==2400
    assert all((ROOT/path).is_file() for path in media)
    state.update(status='BOTH_HISTORY_REFRESH_OFFICIAL_CPU_PLANS_COMPLETE',
        paired_every_step_task_and_question_family_exact=True,
        native_intra_step_length_order_may_differ=True,
        unchanged_teacher_MOT_and_RMOT_OVMOT_payloads=True,replaced_native_history_rows=changed,
        all_referenced_media_exist=True,unique_media_paths=len(media),
        token_or_FLOP_matched=False,new_training_complete=False)
    save()


if __name__=='__main__':
    main()
