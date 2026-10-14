"""One blocking owner reuses GPU4 after the existing matched500 controller exits."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--controller-pid', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    proc = Path('/proc')/str(args.controller_pid)
    start = None
    if proc.exists():
        assert (proc/'cwd').resolve() == ROOT
        assert 'tools/groundjev/run_onejev500_completion_controls.py' in (proc/'cmdline').read_bytes().decode()
        start = (proc/'stat').read_text().rsplit(')', 1)[1].split()[19]
    state = {'status': 'WAITING_ORIGINAL500_MATCHED_CONTROLLER', 'owner_pid': os.getpid(),
             'controller_pid': args.controller_pid, 'controller_start_ticks': start,
             'physical_GPU_after_controller_exit': 4, 'new_training_started': False,
             'probe_scope': 'first4 real registered RMOT frames, functional native Noul check only',
             'GT_used': False, 'benchmark_accuracy_or_E2E_claim': False}
    def save():
        (args.out/'owner_state.json').write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps(state), flush=True)
    def alive():
        try:
            fields = (proc/'stat').read_text().rsplit(')', 1)[1].split()
            return fields[0] != 'Z' and fields[19] == start
        except FileNotFoundError:
            return False
    save()
    try:
        while start is not None and alive():
            time.sleep(20)
        controller = json.loads((ROOT/'outputs/groundjev/training/onejev500_completion_controls_v1/owner_state.json').read_text())
        assert controller['status'] == 'ONEJEV500_MATCHED_OFFICIAL_CONTROLS_COMPLETE', controller['status']
        assert controller['actual_training_done']['steps'] == 500
        state['status'] = 'RUNNING_REAL_ONEJEV500_NATIVE_RMOT_Noul_PROBE'; save()
        source = ROOT/'outputs/groundjev/compact/joint1200_controls_v1/rmot_single8_before/frame_decisions.jsonl'
        output = args.out/'native_rmot4'
        command = ['python3', 'tools/groundjev/run_with_resources.py', '--gpus', '4', '--',
            str(ROOT/'.venv-onejev-py310/bin/python'), 'tools/groundjev/run_video.py', '--task', 'rmot',
            '--video', 'outputs/groundjev/multitask_pilot_data/rmot_single_empty.json', '--backend', 'full',
            '--checkpoint', 'outputs/groundjev/training/onejev_own_history_v3_exact500_replay_stateful/ckpt-step500',
            '--evidence-format', 'v3_crop_motion', '--adaptive-context-groups', '--native-rmot-membership',
            '--candidate-replay', str(source), '--limit', '4', '--out', str(output)]
        env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = ''; env['PYTHONPATH'] = str(ROOT/'external/OneJev')
        with (args.out/'native_probe.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        rows = [json.loads(x) for x in (output/'frame_decisions.jsonl').open()]
        expected = [json.loads(x) for x in source.open()][:4]
        assert len(rows) == 4 and [r['frame'] for r in rows] == [r['frame'] for r in expected]
        values = []
        for row, old in zip(rows, expected):
            assert row['candidates'] == old['candidates']
            meta = row['state']['decision_metadata']
            assert meta['GT_read'] is False and meta['query_membership_source'].startswith('native pretrained OneJev Noul')
            for event in row['state']['events']:
                if 'candidate' in event:
                    probability = event['decision']['query_membership_probability']
                    assert math.isfinite(probability) and 0 <= probability <= 1
                    values.append(probability)
        assert values
        state.update(status='REAL_ONEJEV500_NATIVE_RMOT_Noul_PROBE_COMPLETE', frames=4,
                     native_membership_values=values, all_original_candidates_retained=True,
                     new_RMOT_semantic_training_or_generalization_claim=False)
        save()
    except Exception as error:
        state.update(status='NATIVE_RMOT_PROBE_OWNER_FAILED', error=repr(error)); save()
        raise


if __name__ == '__main__':
    main()
