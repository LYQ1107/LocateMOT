"""Resume verified completed native tasks/caches through unchanged native scorer functions."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'external/GroundAnything'
sys.path.insert(0, str(SOURCE))


def main():
    import yaml
    from scripts import evaluate as native
    from models.dependency_contract import verify_dependency
    from eval_preflight import preflight
    parser=argparse.ArgumentParser()
    parser.add_argument('recipe');parser.add_argument('--run-id',required=True)
    parser.add_argument('--resume-result-root',type=Path,required=True)
    args=parser.parse_args()
    old=args.resume_result_root
    prior=json.loads((old/'run.json').read_text())
    config=native.resolve_suite(yaml.safe_load((SOURCE/args.recipe).read_text()),root=SOURCE,run_id=args.run_id)
    for key in set(prior['config'])-{'run_id'}:
        assert config[key]==prior['config'][key],f'Cannot reuse native responses across changed {key}'
    prepared=native.plan(config)
    assert not prepared['missing_inputs'] and prepared['evaluation_scope']=='full'
    env=native.evaluation_environment(prepared)
    os.environ.clear();os.environ.update(env)
    verify_dependency('lmms_eval',SOURCE)
    sys.path.insert(0,str(SOURCE/'vendor/lmms-eval'))
    from lmms_eval.models import get_model
    get_model('async_openai')
    service=preflight(config,prepared)
    output=SOURCE/prepared['output'];output.mkdir(parents=True,exist_ok=False)
    summary=json.loads((old/'summary.json').read_text())
    completed={record['task'] for record in summary['records']}
    missing=[task for task in config['tasks'] if task not in completed]
    assert set(missing)==set(summary['missing_tasks'])
    reused=[]
    for record in summary['records']:
        source=old/record['source'];data=json.loads(source.read_text())
        assert record['task'] in data['results']
        samples=data['n-samples'][record['task']]
        assert samples['effective']==samples['original'] and samples['effective']>0
        target=output/'resumed_completed'/source.name;target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source,target)
        reused.append({'task':record['task'],'source':str(source),'all_original_samples':samples})
    caches=[]
    old_cache=old/'cache'/prior['config']['run_id']
    new_cache=output/'cache'/args.run_id;new_cache.mkdir(parents=True)
    for source in old_cache.glob('*.db'):
        with sqlite3.connect('file:'+str(source)+'?mode=ro',uri=True) as before,sqlite3.connect(new_cache/source.name) as after:
            before.backup(after)
        caches.append(source.name)
    command=prepared['command'];command[command.index('--tasks')+1]=','.join(missing)
    resume={'completed_tasks_reused':reused,'unfinished_tasks':missing,'native_response_cache_backups':caches,
        'source':str(old),'native_source_parser_scorer_data_decode_profile_and_seed_unchanged':True,
        'exact_stochastic_request_counter_continuation_claim':False}
    (output/'run.json').write_text(json.dumps({'config':config,'plan':prepared,'service':service,'resume':resume},indent=2)+'\n')
    (output/'resume_receipt.json').write_text(json.dumps(resume,indent=2)+'\n')
    return native.execute_evaluation(prepared,output,env,root=SOURCE)


if __name__=='__main__':
    raise SystemExit(main())
