"""Hand off only the prepared paired-study CPU waiter before its first update."""
import argparse
import json
import os
from pathlib import Path
import select
import signal

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    old = ROOT/plan['original_v2_CPU_prepared_owner']
    state = json.loads((old/'owner_state.json').read_text())
    assert state['status'] == 'WAITING_ORIGINAL_NATIVE6000_OWNER_SUCCESSFUL_EXIT'
    assert state['actual_production_optimizer_steps'] == 0 and 'child_pid' not in state
    proof = json.loads((old/'prepared_and_CPU_validated.json').read_text())
    assert proof['new_optimizer_updates'] == 0
    assert json.loads((ROOT/'outputs/groundjev/validation/gpu7_temporal_window_v1/validation.json').read_text())['status'] == 'REAL_CPU_WRAPPER_TRAINING_WINDOW_AND_PREPARED_INPUT_REGRESSION_PASSED'
    for name in plan['training_names'].values():
        assert not (Path(plan['runs_dir'])/name).exists()
    assert not (ROOT/plan['owner_output']).exists()
    pid, tick = state['owner_pid'], state['owner_starttick']
    proc = Path('/proc')/str(pid)
    fields = (proc/'stat').read_text().rsplit(')',1)[1].split()
    assert fields[19] == tick and fields[0] != 'Z'
    assert (proc/'cwd').resolve() == ROOT
    command = (proc/'cmdline').read_bytes().decode().split('\0')
    assert 'tools/groundjev/run_paired_temporal_teacher.py' in command
    assert str(ROOT/'configs/groundjev/paired_temporal_teacher24000_v2_indexed.json') in command or 'configs/groundjev/paired_temporal_teacher24000_v2_indexed.json' in command
    assert not (proc/f'task/{pid}/children').read_text().strip()
    assert b'CUDA_VISIBLE_DEVICES=\x00' in (proc/'environ').read_bytes()
    receipt = ROOT/plan['CPU_wait_handoff_receipt']
    receipt.parent.mkdir(parents=True, exist_ok=False)
    intent = dict(status='VERIFIED_CPU_ONLY_WAIT_OWNER_HANDOFF_INTENT',old_owner_directory=str(old),
        old_identity=[pid,tick],old_state=state,replacement_plan=str(args.plan),
        target_has_no_children_and_empty_CUDA_visibility=True,old_training_updates=0,
        original_four_GPU_epoch_native6000_controls_and_other_jobs_targeted=False,
        all_completed_inputs_and_original_owner_state_preserved=True,full_goal_complete=False)
    (receipt.parent/'intent.json').write_text(json.dumps(intent,indent=2)+'\n')
    descriptor=os.pidfd_open(pid)
    try:
        assert (proc/'stat').read_text().rsplit(')',1)[1].split()[19] == tick
        signal.pidfd_send_signal(descriptor,signal.SIGTERM)
        waiter=select.poll();waiter.register(descriptor,select.POLLIN)
        assert waiter.poll(15000), 'CPU owner did not exit; do not launch duplicate'
    finally:
        os.close(descriptor)
    result=dict(intent,status='CPU_ONLY_WAIT_OWNER_EXIT_CONFIRMED',signal='SIGTERM',
        old_owner_exit_confirmed_by_pidfd=True,old_prepared_data_or_state_files_modified=False,
        old_wrapper_terminal_exit_code='Record original session60582 separately; pidfd proves target exited, not wrapper code.',
        replacement_started=False,new_GPU_or_optimizer_updates=0)
    receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],old_identity=[pid,tick],receipt=str(receipt))))


if __name__ == '__main__':
    main()
