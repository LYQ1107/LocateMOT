"""One blocking owner for the registered paired Full600 runs on GPUs4 and5."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.groundjev.service_process_group import members


def register(pid, expected_script):
    proc = Path('/proc')/str(pid)
    if not proc.exists():
        return None
    argv = (proc/'cmdline').read_bytes().decode().split('\0')
    assert (proc/'cwd').resolve() == ROOT and expected_script in argv
    return (pid, (proc/'stat').read_text().rsplit(')', 1)[1].split()[19])


def alive(identity):
    if identity is None:
        return False
    try:
        fields = (Path('/proc')/str(identity[0])/'stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and fields[19] == identity[1]
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--probe-dir', type=Path, default=ROOT/'outputs/groundjev/validation/onejev500_native_rmot_probe_v2_validated')
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    probe_dir = args.probe_dir
    probe = json.loads((probe_dir/'owner_state.json').read_text())
    probe_owner = register(probe['owner_pid'], 'tools/groundjev/wait_onejev500_semantic_probe.py')
    tao_dir = ROOT/'outputs/groundjev/multitask_runs/tao_base_novel63_full_v2_expression_none'
    tao_owner = None
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc/'cmdline').read_bytes().decode().split('\0')
            if len(argv) > 1 and argv[1] == 'tools/groundjev/run_tao_base_novel_full.py' and (
                    str(tao_dir.relative_to(ROOT)) in argv or str(tao_dir) in argv):
                tao_owner = register(int(proc.name), argv[1])
        except (FileNotFoundError, PermissionError):
            pass
    if tao_owner is None:
        assert json.loads((tao_dir/'owner_state.json').read_text()).get('owned_native_service_groups_stopped')
    preflight = json.loads((ROOT/'outputs/groundjev/training_data/onejev_unified_matched600_preflight_v2/owner_state.json').read_text())
    assert preflight['status'] == 'BOTH_UNIFIED600_OFFICIAL_CPU_PREFLIGHTS_COMPLETE_NOT_TRAINED'
    assert preflight['paired_per_step_semantic_rows_and_all_question_families_exact']
    initial = json.loads((ROOT/'outputs/groundjev/validation/onejev_fp32_stage500_v1/validation.json').read_text())
    assert initial['initialization']['initialization_max_difference'] == 0
    state = {'status': 'WAITING_VERIFIED_REAL_NATIVE_ONE500_SEMANTIC_PROBE', 'owner_pid': os.getpid(),
        'probe_owner': probe_owner, 'GPU5_TAO_resource_owner': tao_owner,
        'physical_gpus': {'teacher': 4, 'native': 5}, 'registered_additional_updates_each': 600,
        'sampler': 'paired-index-shuffle-v1', 'actual_semantic_rows_and_question_families_per_step_identical': True,
        'actual_prefix_rows_each': 2400, 'exact_old_optimizer_trajectory_resume_claim': False,
        'new_stage_optimizer_scheduler_RNG': 'explicit recipe reset from original500 FP32 masters',
        'completed': {}, 'new_training_started': False}
    def save():
        (args.out/'owner_state.json').write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps({k: v for k, v in state.items() if k not in ('completed', 'original500_matched_metrics')}), flush=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', WORLD_SIZE='1', RANK='0', LOCAL_RANK='0',
               PYTHONPATH=str(ROOT/'external/OneJev'))
    children, threads = {}, []
    def launch(variant):
        free = subprocess.check_output(['nvidia-smi', '-i', str(state['physical_gpus'][variant]),
            '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip()
        state[variant+'_GPU_free_MiB_at_launch'] = int(free)
        command = ['python3', 'tools/groundjev/run_with_resources.py', '--gpus', str(state['physical_gpus'][variant]), '--',
            str(ROOT/'.venv-onejev-py310/bin/python'), 'tools/groundjev/onejev_resumable_sft.py',
            '--config', f'configs/groundjev/onejev_unified600_{variant}_v2.json', '--paired-index-shuffle',
            '--initial-training-state', 'outputs/groundjev/training/onejev_own_history_v3_exact500_replay_stateful/latest_training_state.pt',
            '--expected-initial-step', '500']
        training = ROOT/f'outputs/groundjev/training/onejev_unified600_{variant}_v2'
        assert not training.exists(), 'preserve existing training state and resume its valid cursor separately'
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        children[variant] = child
        def forward():
            with (args.out/(variant+'_train.log')).open('w') as log:
                for line in child.stdout:
                    log.write(line); log.flush(); print('['+variant+'] '+line, end='', flush=True)
        thread = threading.Thread(target=forward); thread.start(); threads.append(thread)
        state['new_training_started'] = True; state[variant+'_child_pid'] = child.pid; save()
    save()
    try:
        while alive(probe_owner):
            time.sleep(20)
        probe = json.loads((probe_dir/'owner_state.json').read_text())
        assert probe['status'] == 'REAL_ONEJEV500_NATIVE_RMOT_Noul_PROBE_COMPLETE', probe['status']
        controls = json.loads((ROOT/'outputs/groundjev/training/onejev500_completion_controls_v1/owner_state.json').read_text())
        assert controls['status'] == 'ONEJEV500_MATCHED_OFFICIAL_CONTROLS_COMPLETE' and len(controls['completed']) == 16
        state['original500_matched_metrics'] = {k: v['official_metrics']['metrics'] for k, v in controls['completed'].items()}
        state['registered_rationale'] = 'MOT/RMOT pretrain versus500 results are mixed; controlled candidate-domain and explicit common semantic supervision, no promotion on short clips'
        state['status'] = 'TEACHER_GPU4_TRAINING_NATIVE_GPU5_WAITING_OWNED_TAO'; launch('teacher')
        while alive(tao_owner):
            time.sleep(20)
        tao = json.loads((tao_dir/'owner_state.json').read_text())
        assert tao.get('owned_native_service_groups_stopped')
        assert all(not members(group) for group in tao['owned_service_groups'])
        # TAO is an independent measurement: its failure is not a global
        # training NO_GO once its owned GPU5 resources have actually exited.
        state['GPU5_TAO_terminal_resource_handoff_status'] = tao['status']
        state['status'] = 'BOTH_PAIRED_UNIFIED600_FULL_MODELS_TRAINING'; launch('native')
        codes = {name: child.wait() for name, child in children.items()}
        for thread in threads:
            thread.join()
        state['return_codes'] = codes
        for variant, code in codes.items():
            training = ROOT/f'outputs/groundjev/training/onejev_unified600_{variant}_v2'
            if code == 0:
                logs = [json.loads(x) for x in (training/'log.jsonl').read_text().splitlines()]
                done = next(x for x in logs if x.get('event') == 'done')
                assert done['steps'] == 600 and done['examples'] == 2400
                master = json.loads((training/'latest_training_state.json').read_text())
                assert master['step'] == 600
                state['completed'][variant] = {'done': done, 'latest_full_state': master,
                    'native_reload_and_three_task_official_controls': 'pending, not claimed by training completion'}
        if any(codes.values()):
            raise RuntimeError('A registered600 arm failed; retain logs/checkpoints and resume its unfinished cursor')
        state['status'] = 'PAIRED_UNIFIED600_TRAINING_COMPLETE_NATIVE_OFFICIAL_CONTROLS_PENDING'; save()
    except Exception as error:
        state.update(status='PAIRED_UNIFIED600_OWNER_FAILED', error=repr(error)); save()
        # Keep any independently valid arm running to its registered endpoint.
        for child in children.values():
            child.wait()
        for thread in threads:
            thread.join()
        raise


if __name__ == '__main__':
    main()
