"""Validate the completed matched Full study without launching another GPU job."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'external/OneJev')]


def main():
    import torch
    from safetensors import safe_open
    from tools.groundjev.onejev_resumable_sft import protocol

    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); args.out = args.out.resolve()
    torch.set_num_threads(4)
    owner_root = ROOT/'outputs/groundjev/training/full_history_refresh600_pair_v2'
    owner = json.loads((owner_root/'owner_state.json').read_text())
    assert owner['status'] == 'PAIRED_HISTORY_REFRESH600_TRAINING_AND_MM_RELOAD_COMPLETE'
    actual_rows, reports = {}, {}
    for arm in ('old', 'refreshed'):
        training = ROOT/f'outputs/groundjev/training/onejev_history_refresh600_{arm}_v1'
        settings = json.loads((training/'args.json').read_text())
        records = [json.loads(l) for l in (training/'log.jsonl').open()]
        done = next(r for r in records if r.get('event') == 'done')
        logged = [r for r in records if 'loss' in r]
        assert done['steps'] == 600 and done['examples'] == 2400
        assert logged[-1]['step'] == 600
        assert all(math.isfinite(r[k]) for r in logged for k in ('loss', 'grad_norm'))
        initial = json.loads((training/'initial_fp32_state_receipt.json').read_text())
        assert initial['source_step'] == 600 and initial['initialization_max_difference'] == 0
        assert initial['parameters'] == 752393024 and initial['FP32_parameter_entries'] == 320
        latest = json.loads((training/'latest_training_state.json').read_text())
        assert latest['step'] == 600 and latest['seen'] == 2400
        assert latest['bytes'] == (training/'latest_training_state.pt').stat().st_size
        mm = json.loads((training/'checkpoint600_validation.json').read_text())
        assert mm['status'] == 'REAL_ONEJEV600_NATIVE_MM_RELOAD_VALIDATED'
        assert mm['formal_training_steps'] == 600 and mm['repeated_reload_probability_max_difference'] == 0
        assert mm['native_fork_auto_check_unchanged'] and not mm['GT_read_by_inference']
        actual_rows[arm] = [json.loads(l) for l in Path(settings['train_file']).open()]
        assert len(actual_rows[arm]) == done['examples']
        exposures = Counter(r['supervision']['tracking_task_bucket'] for r in actual_rows[arm])
        assert exposures == {'mot': 800, 'rmot': 800, 'ovmot': 800}
        if arm == 'old':
            report = json.loads((training/'completion_validation.json').read_text())
            assert report['actual_CPU_mmap_state_read'] and report['all_FP32_parameters_and_Adam_moments_finite']
            assert report['all_Adam_steps'] == 600
        else:
            saved = torch.load(training/'latest_training_state.pt', map_location='cpu', weights_only=True, mmap=True)
            assert saved['format'] == 'groundjev-full-fp32-adam-rng-v1'
            assert saved['step'] == saved['next_step_in_epoch'] == 600 and saved['seen'] == 2400
            assert saved['protocol'] == protocol(SimpleNamespace(**settings))
            assert all(k in saved for k in ('optimizer', 'scheduler', 'torch_rng', 'cuda_rng', 'python_rng', 'numpy_rng'))
            assert saved['cuda_rng'] and saved['scheduler']['last_epoch'] == 600
            masters = saved['fp32_trainable_parameters']; opt = saved['optimizer']['state']
            assert len(masters) == len(opt) == 320
            assert sum(v.numel() for v in masters.values()) == 752393024
            with safe_open(training/'ckpt-step600/model.safetensors', framework='pt', device='cpu') as exported:
                for name, master in masters.items():
                    assert master.dtype == torch.float32 and torch.isfinite(master).all()
                    assert torch.equal(master.bfloat16(), exported.get_tensor(name))
            moment_counts = Counter()
            for item in opt.values():
                assert int(item['step']) == 600
                for key in ('exp_avg', 'exp_avg_sq'):
                    assert item[key].dtype == torch.float32 and torch.isfinite(item[key]).all()
                    moment_counts[key] += item[key].numel()
            assert dict(moment_counts) == {'exp_avg': 752393024, 'exp_avg_sq': 752393024}
            report = dict(status='REFRESHED_HISTORY600_COMPLETE_FULL_STATE_EXPORT_AND_REAL_MM_RELOAD_VALIDATED',
                arm=arm, training=str(training), source_config=settings['config'], stage_updates=600,
                stage_examples=2400, complete_uninterrupted_process_elapsed_s=done['elapsed_s'],
                logged_training_records=len(logged), all_logged_losses_and_grad_norms_finite=True,
                logged_final_step=logged[-1], initial_FP32_source_step=600, initial_FP32_max_difference=0,
                full_state_bytes=latest['bytes'], actual_CPU_mmap_state_read=True,
                FP32_parameter_entries=320, FP32_parameters=752393024,
                all_FP32_parameters_and_Adam_moments_finite=True, Adam_state_entries=320,
                all_Adam_steps=600, Adam_moment_elements_each=752393024,
                all320_BF16_inference_parameters_equal_saved_FP32_master_cast=True,
                optimizer_scheduler_all_RNG_and_cursor_present=True, saved_protocol_matches_actual_training_args=True,
                native_MM_weight_entries=mm['weight_entries'], real_MM_independent_reloads=2,
                repeated_reload_probability_max_difference=0., native_fork_auto_check_unchanged=True,
                original_training_data_rows=2400, actual_source_task_exposures=dict(exposures),
                question_family_counts=dict(Counter(r['task'] for r in actual_rows[arm])),
                exact_restart_trajectory_tested=False, tracking_metrics='pending registered four-scope controls',
                tracking_improvement_proven=False, Ground_One_end_to_end_joint_training=False,
                large_scale_unique_frame_training=False, full_goal_complete=False,
                new_GPU_job_or_feature_archive_created_by_validation=False)
            (training/'completion_validation.json').write_text(json.dumps(report, indent=2)+'\n')
        reports[arm] = report
    changed = Counter()
    for old, new in zip(actual_rows['old'], actual_rows['refreshed']):
        assert old['id'] == new['id'] and old['task'] == new['task']
        assert old['supervision']['tracking_task_bucket'] == new['supervision']['tracking_task_bucket']
        expected = old['supervision']['tracking_task_bucket'] == 'mot' and old['supervision']['source_row_id'].startswith('native_')
        assert (old != new) == expected
        changed['changed_native_history_rows' if expected else 'exactly_unchanged_rows'] += 1
    assert changed == {'changed_native_history_rows': 400, 'exactly_unchanged_rows': 2000}
    assert not torch.cuda.is_initialized()
    result = dict(status='BOTH_FULL600_HISTORY_ARMS_ACTUALLY_COMPLETE_SHARED_TASK_EXPOSURE_AND_STATES_VALIDATED',
        original_blocking_session=21963, actual_exit_code=0, arms=reports,
        rowwise_actual_task_ID_and_question_family_paired=True, changed=dict(changed),
        sampling_protocol='paired-index-shuffle-v1', single_epoch_all2400_rows_consumed_each=True,
        same_initial_real_FP32_masters=True, token_or_FLOP_matched=False,
        tracking_metrics_still_pending=True, checkpoint_promoted=False, full_goal_complete=False)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'arms'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
