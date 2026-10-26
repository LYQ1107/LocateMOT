"""Complete-video J1 input controls on an already registered physical GPU.

The native G2 queue continues. This owner executes both controls sequentially,
waits for their real inference and official scoring, and preserves failures.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.groundjev.eval_no_mot20.common import load, save, job_manifest
from tools.groundjev.eval_no_mot20.dispatch_4gpu import gpu_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT/'configs/groundjev/eval_no_mot20_4gpu_v1.json')
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--birth-only', action='store_true', help='isolated TRAIN birth-question control against v4')
    args = parser.parse_args()
    config = load(args.config)
    worker = next(w for w in config['workers'] if w['gpu'] == args.gpu)
    run = Path(config['output_root'])
    audit = load(run/'model_audit.json')
    manifest = job_manifest(args.job)
    baseline_arm='j1_current_crops_memory_v4' if args.birth_only else 'j1_competition_xyxy_v2'
    baseline = load(args.job/baseline_arm/'frozen_receipt.json')
    simple = load(args.job/'simple/frozen_receipt.json')
    if baseline['input'] != simple['input']:
        raise ValueError('The complete-video baselines do not share native candidates')
    arms = [
        {'arm': 'j1_current_crops_v3', 'candidate_evidence': 'train_current_crops',
         'memory_transport': 'legacy'},
        {'arm': 'j1_current_crops_memory_v4', 'candidate_evidence': 'train_current_crops',
         'memory_transport': 'local_xyxy'},
    ]
    if args.birth_only:
        arms=[{'arm':'j1_train_birth_prompt_v5','candidate_evidence':'train_current_crops',
               'memory_transport':'local_xyxy','empty_history_question':'train_birth_v1'}]
    directory = run/'engineering_checks'/('j1_birth_question_contract' if args.birth_only else 'j1_input_contract')
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        registration = {
            'status': 'REGISTERED_BEFORE_CONTROL_INFERENCE', 'job': str(args.job),
            'dataset': manifest['dataset'], 'input': simple['input'], 'arms': arms,
            'baseline_arm':baseline_arm,
            'competition_geometry': 'xyxy', 'J1': audit['J1_merged'],
            'G2_step': config['g2_step'], 'G2_sha256': config['g2_sha256'],
            'source_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
            'gpu': args.gpu, 'gpu_uuid': worker['uuid'], 'physical_gpu_pool': [w['gpu'] for w in config['workers']],
            'new_training': False, 'GT_runtime_access': False, 'frozen_thresholds': True,
            'main_evaluation_arm_unchanged': True,
        }
        if not (directory/'registration.json').exists():
            save(directory/'registration.json', registration)
        else:
            registered = load(directory/'registration.json')
            if any(registered[key] != registration[key] for key in ('job','input','arms','J1','G2_sha256')):
                raise ValueError('Existing input-control registration differs')
        env = gpu_environment(args.gpu)
        results = []
        for arm in arms:
            raw = subprocess.check_output(['nvidia-smi','-i',str(args.gpu),
                '--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'], text=True).strip()
            uuid, free = [value.strip() for value in raw.split(',')]
            required_free=config['minimum_free_MiB']+12288 if args.birth_only else 12288
            if uuid != worker['uuid'] or int(free) < required_free:
                raise RuntimeError(f'Registered GPU has insufficient space for a J1 control: {raw}')
            command = [config['onejev_python'],str(ROOT/'tools/groundjev/eval_no_mot20/replay_j1.py'),
                       '--job',str(args.job),'--checkpoint',audit['J1_merged']['path'],
                       '--arm',arm['arm'],'--competition-geometry','xyxy',
                       '--candidate-evidence',arm['candidate_evidence'],
                       '--memory-transport',arm['memory_transport'],'--no-followups']
            command+=['--empty-history-question',arm.get('empty_history_question','identity')]
            started = time.time()
            with (directory/f"{arm['arm']}.log").open('a') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                save(directory/'owner.json', {'status':'RUNNING_REAL_FULL_VIDEO_INFERENCE',
                    'pid':child.pid,'arm':arm,'start':started,'gpu':args.gpu,
                    'free_MiB_at_start':int(free),'command':command})
                exit_code = child.wait()
            result = {**arm, 'exit_code':exit_code,'wall_seconds':time.time()-started}
            if exit_code == 0:
                frozen = load(args.job/arm['arm']/'frozen_receipt.json')
                if frozen['input'] != simple['input']:
                    raise ValueError('Input control changed the frozen full-video candidates')
                result['frozen_receipt'] = frozen
            else:
                result['status'] = 'FAILED_WITH_LOG_PRESERVED'
                result['log'] = str(directory/f"{arm['arm']}.log")
            results.append(result)
            save(directory/'progress.json', {'results':results,'full_video_frames':simple['input']['frames']})
            print(json.dumps({'event':'J1_INPUT_CONTROL_FINISHED',**arm,
                'exit_code':exit_code,'wall_seconds':result['wall_seconds'],
                'metrics':result.get('frozen_receipt',{}).get('metrics',{}).get('metrics')}), flush=True)
        receipt = {'status':'FULL_VIDEO_INPUT_CONTROLS_COMPLETE' if all(r['exit_code']==0 for r in results)
                   else 'INPUT_CONTROLS_HAVE_EXPLICIT_FAILURES', 'registration':registration,'results':results,
                   'same_checkpoint_and_candidates':True,'new_training':False}
        save(directory/'receipt.json', receipt)
        save(directory/'owner.json', {'status':receipt['status'],'results':results})


if __name__ == '__main__':
    main()
