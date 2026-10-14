"""One blocking owner for both independent CPU-only official curriculum plans."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--version', type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    state = {'status': 'RUNNING_TWO_CPU_OFFICIAL_UNIFIED600_PREFLIGHTS', 'owner_pid': os.getpid(),
             'models_or_GPUs_loaded': False, 'training_started': False, 'completed': {}}
    def save():
        (args.out/'owner_state.json').write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps({k: v for k, v in state.items() if k != 'completed'}), flush=True)
    env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES='', PLAN_WORLD='1', WORLD_SIZE='1',
                                        PYTHONPATH=str(ROOT/'external/OneJev'))
    children, logs = {}, {}
    try:
        for variant in ('teacher', 'native'):
            logs[variant] = (args.out/(variant+'.log')).open('w')
            suffix = '' if args.version == 1 else f'_v{args.version}'
            command = [str(ROOT/'.venv-onejev-py310/bin/python'), 'tools/groundjev/onejev_plan_with_exposure.py',
                       '--config', f'configs/groundjev/onejev_unified600_{variant}_token_preflight{suffix}.json',
                       '--plan-only', '--receipt', str(args.out/(variant+'.json'))]
            if args.version == 2:
                command.append('--paired-index-shuffle')
            children[variant] = subprocess.Popen(command, cwd=ROOT, env=env, stdout=logs[variant], stderr=subprocess.STDOUT)
        save()
        codes = {name: proc.wait() for name, proc in children.items()}
        state['return_codes'] = codes
        for variant, code in codes.items():
            if code == 0:
                result = json.loads((args.out/(variant+'.json')).read_text())
                state['completed'][variant] = {k: v for k, v in result.items() if k not in ('prefix', 'config', 'actual_prefix_label_counts')}
        if any(codes.values()):
            raise RuntimeError('Official CPU preflight failed; preserve individual logs and valid completed arm')
        reports = {v: json.loads((args.out/(v+'.json')).read_text()) for v in children}
        common = {v: {r['id'] for r in report['prefix'] if r['bucket'] != 'mot'} for v, report in reports.items()}
        state['common_semantic_prefix_row_ids'] = {'teacher': len(common['teacher']), 'native': len(common['native']),
            'intersection': len(common['teacher'] & common['native']),
            'identical_actual_semantic_prefix': common['teacher'] == common['native']}
        if args.version == 2:
            a, b = reports['teacher']['prefix'], reports['native']['prefix']
            assert [(r['step'], r['index'], r['bucket'], r['family']) for r in sorted(a, key=lambda r: r['index'])] == [
                    (r['step'], r['index'], r['bucket'], r['family']) for r in sorted(b, key=lambda r: r['index'])]
            assert state['common_semantic_prefix_row_ids']['identical_actual_semantic_prefix']
            state['paired_per_step_semantic_rows_and_all_question_families_exact'] = True
        state['same_optimizer_update_budget_not_equal_token_FLOPs'] = True
        state['status'] = 'BOTH_UNIFIED600_OFFICIAL_CPU_PREFLIGHTS_COMPLETE_NOT_TRAINED'; save()
    except Exception as error:
        state.update(status='UNIFIED600_CPU_PREFLIGHT_FAILED', error=repr(error)); save()
        raise
    finally:
        for proc in children.values():
            if proc.poll() is None:
                proc.terminate(); proc.wait()
        for log in logs.values():
            log.close()


if __name__ == '__main__':
    main()
