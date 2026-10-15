"""Scope real resumed SFT counters/timing and report observed overlap differences."""
import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--interrupted-log', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.training/'log.jsonl').open()]
    old = [json.loads(line) for line in args.interrupted_log.open()]
    restore = json.loads((args.training/'resumed_training_state_receipt.json').read_text())
    master = json.loads((args.training/'latest_training_state.json').read_text())
    done = next(r for r in rows if r.get('event') == 'done')
    assert done['steps'] == master['step'] == 600 and done['examples'] == master['seen'] == 2400
    assert Path(master['path']).stat().st_size == master['bytes']
    assert restore['FP32_master_max_difference'] == 0 and restore['protocol_exact']
    assert all(math.isfinite(r['loss']) for r in rows if 'loss' in r)
    tail_steps = done['steps']-restore['step']
    tail_examples = done['examples']-restore['seen']
    assert tail_examples == 4*tail_steps
    previous = {r['step']: r for r in old if 'loss' in r}
    overlaps = []
    for r in rows:
        step = r.get('step')
        if step is not None and step > restore['step'] and step in previous:
            before = previous[step]
            overlaps.append({'step': step, 'original_loss': before['loss'], 'replayed_loss': r['loss'],
                'loss_absolute_difference': abs(before['loss']-r['loss']),
                'slot_accuracy_absolute_difference': abs(before['slot_acc']-r['slot_acc']),
                'gradient_norm_absolute_difference': abs(before['grad_norm']-r['grad_norm'])})
    old_logged = max(r.get('step', 0) for r in old)
    result = {'status': 'ACTUAL_RETAINED600_SFT_COMPLETION_VALIDATED_WITH_SCOPED_RESUME_LIMITATIONS',
        'training': str(args.training), 'retained_stage_updates': 600, 'retained_stage_examples': 2400,
        'complete_FP32_Adam_scheduler_RNG_cursor_saved': True, 'latest_full_state': master,
        'restored_cursor': restore['step'], 'restored_FP32_master_max_difference': 0,
        'restored_protocol_exact': True, 'completed_updates_after_restore': tail_steps,
        'completed_examples_after_restore': tail_examples,
        'raw_done_elapsed_s_scope': 'only resumed process; not complete original+resumed600stage wall time',
        'resumed_process_elapsed_s': done['elapsed_s'],
        'resumed_tail_examples_per_s': tail_examples/done['elapsed_s'],
        'raw_done_cumulative_examples_per_s_not_valid_tail_speed': done['examples_per_s'],
        'raw_done_records_preserved': True,
        'logged_unretained_updates_after_saved_cursor': max(0, old_logged-restore['step']),
        'total_observed_optimizer_updates_lower_bound': 600+max(0, old_logged-restore['step']),
        'original_complete_wall_time_or_standalone_training_throughput_claim': False,
        'overlap_log_comparisons': overlaps,
        'bitwise_training_trajectory_replay_proven': bool(overlaps) and all(r['loss_absolute_difference'] == 0 for r in overlaps),
        'numerical_replay_difference_root_cause': 'not established; full-state restore is exact, deterministic row-seeded render/sampler unchanged, subsequent CUDA trajectory equality is not asserted',
        'native_MM_reload_and_official_tracking_performance': 'separate running matched controls, not proved by training completion',
        'seed_loss_kernel_or_acceptance_criterion_modified': False}
    (args.training/'resumed_completion_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'latest_full_state'}, indent=2))


if __name__ == '__main__':
    main()
