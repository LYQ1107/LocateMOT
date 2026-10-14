"""One blocking owner for unchanged native full dense/referring retention scoring."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from urllib.request import urlopen
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.groundjev.service_process_group import stop_owned_service
SOURCE = ROOT/'external/GroundAnything'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/groundjev/ground_retention/original_dense_ref9_v1')
    parser.add_argument('--run-id',default='groundjev_original_dense_ref9_20261014')
    parser.add_argument('--gpu',type=int,default=7)
    parser.add_argument('--port',type=int,default=8101)
    parser.add_argument('--model',type=Path,default=ROOT/'models/GroundAnything-DLM')
    parser.add_argument('--recipe',default='configs/eval/groundjev_original_dense_ref9.local.yaml')
    parser.add_argument('--max-queued-requests',type=int,default=32)
    parser.add_argument('--resume-result-root',type=Path)
    args=parser.parse_args();out = args.out
    out.mkdir(parents=True, exist_ok=False)
    recipe=json.loads((SOURCE/args.recipe).read_text())
    assert recipe['limit'] is None and len(recipe['tasks'])==9
    assert recipe['api_url']==f'http://127.0.0.1:{args.port}/v1'
    baseline=json.loads((ROOT/'configs/groundjev/ground_retention_dense_ref9_original.yaml').read_text())
    for key in set(baseline)-{'model_path','api_url','run_id','base_port'}:
        assert recipe[key]==baseline[key], f'Retention protocol differs at {key}'
    original=args.model.resolve()==(ROOT/'models/GroundAnything-DLM').resolve()
    state = {'status': 'STARTING_NATIVE_DENSE_REF9', 'physical_server_gpu': args.gpu,
        'local_evaluator_CPU_only': True, 'limit': None, 'declared_canonical_rows': 91124,
        'nine_original_dense_referring_tasks_not_all30': True, 'adaptation_or_retention_gain_claim': False,
        'checkpoint':str(args.model.resolve()),'original_model':original,'recipe':args.recipe,
        'native_parser_scorers_decode_profiles_and_all_task_settings_unchanged':True,
        'execution_rendezvous_base_port':recipe['base_port']}
    state['max_queued_requests'] = args.max_queued_requests
    state['resume_result_root'] = str(args.resume_result_root) if args.resume_result_root else None
    def save():
        (out/'owner_state.json').write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps(state), flush=True)
    save(); server = None
    try:
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', args.port))
        with (out/'server.log').open('w') as log:
            server = subprocess.Popen(['python3', 'tools/groundjev/run_with_resources.py', '--gpus', str(args.gpu), '--',
                str(SOURCE/'.venv-serve/bin/python'), 'tools/groundjev/serve_groundanything.py', '--port', str(args.port),
                '--model',str(args.model),
                '--max-queued-requests',str(args.max_queued_requests),
                '--mem-fraction-static', '.4', '--out', str(out/'service')], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state.update(status='WAITING_OWNED_ORIGINAL_GROUND_SERVICE', owned_server_process_group=server.pid); save()
            deadline = time.monotonic()+600
            while True:
                if server.poll() is not None:
                    raise RuntimeError('Owned original Ground service exited before readiness')
                try:
                    with urlopen(f'http://127.0.0.1:{args.port}/v1/models', timeout=5) as reply:
                        models = json.load(reply)
                    if any(m['id'] == 'groundinganything' for m in models['data']):
                        break
                except (OSError, ValueError):
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError('Owned original Ground readiness timeout')
                time.sleep(2)
            state['status'] = 'RUNNING_UNMODIFIED_FULL_DENSE_REF9_BASELINE' if original else 'RUNNING_UNMODIFIED_FULL_DENSE_REF9_ADAPTED'; save()
            env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = ''
            for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
                env.pop(key,None)
            with (out/'native_evaluator.log').open('w') as log:
                if args.resume_result_root:
                    command=[str(SOURCE/'.venv-eval/bin/python'),str(ROOT/'tools/groundjev/resume_ground_dense_ref9.py'),
                        args.recipe,'--run-id',args.run_id,'--resume-result-root',str(args.resume_result_root.resolve())]
                else:
                    command=[str(SOURCE/'.venv-eval/bin/python'),'scripts/evaluate.py',args.recipe,'--run-id',args.run_id]
                subprocess.run(command, cwd=SOURCE,
                    env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            result_root = SOURCE/'outputs/groundjev_retention'/args.run_id
            summary = json.loads((result_root/'summary.json').read_text())
            assert summary['status'] == 'completed' and not summary['missing_tasks'] and summary['evaluation_scope'] == 'full'
            state.update(status='UNMODIFIED_FULL_ORIGINAL_DENSE_REF9_BASELINE_COMPLETE' if original else 'UNMODIFIED_FULL_ADAPTED_DENSE_REF9_COMPLETE', summary=summary,
                         native_result_root=str(result_root)); save()
    except Exception as error:
        state.update(status='ORIGINAL_DENSE_REF9_BASELINE_FAILED' if original else 'ADAPTED_DENSE_REF9_FAILED', error=repr(error)); save()
        raise
    finally:
        if server is not None:
            stop_owned_service(server)
        state['owned_Ground_service_stopped'] = True
        if original: state['owned_original_Ground_service_stopped'] = True
        save()


if __name__ == '__main__':
    main()
