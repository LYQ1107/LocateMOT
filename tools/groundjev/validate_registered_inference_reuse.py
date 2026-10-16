"""Check future binding selection and reuse an actual completed native run."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from tools.groundjev.registered_inference_reuse import REGISTRATION, prepare_command, materialize_completed_run
    from tools.groundjev.run_history_refresh_controls import validate_run
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); args.out = args.out.resolve(); args.out.mkdir(parents=True, exist_ok=False)
    registration = json.loads(REGISTRATION.read_text())
    bindings = []
    for binding in registration['bindings']:
        command = ['python', str(ROOT/'tools/groundjev/run_video.py')]
        for flag, value in binding['required_options'].items():
            command.extend([flag, str(ROOT/value) if flag in ('--video', '--checkpoint', '--candidate-replay') else value])
        command += binding['required_flags']+['--out', str(ROOT/binding['consumer'])]
        effective, receipt = prepare_command(command)
        assert receipt['binding'] == binding
        assert ('--category-replay' in effective) == (binding['mode'] == 'same_checkpoint_current_frame_categories')
        other = deepcopy(command); other[other.index('--out')+1] += '_unregistered'
        assert prepare_command(other) == (other, None)
        wrong_checkpoint = deepcopy(command)
        wrong_checkpoint[wrong_checkpoint.index('--checkpoint')+1] = str(ROOT/'models/OneJev-0.8B')
        try:
            prepare_command(wrong_checkpoint)
        except AssertionError:
            pass
        else:
            raise AssertionError('Different checkpoint must not reuse categories/baseline')
        bindings.append(dict(mode=binding['mode'], exact_consumer_only=True,
            other_checkpoint_rejected=True, effective_category_replay_flag='--category-replay' in effective))
    controls = json.loads((ROOT/'configs/groundjev/full_history_refresh600_controls_v1.json').read_text())
    job = controls['scopes']['mot_full1203']; source = ROOT/controls['output_root']/'old_mot_full1203'
    target = args.out/'completed_native_DT1203_reuse'
    manifest = json.loads((ROOT/job['manifest']).read_text()); checkpoint = ROOT/controls['checkpoints']['old']
    receipt = materialize_completed_run(source, target, manifest, checkpoint, ROOT/job['candidates'])
    checked = validate_run(target, job, checkpoint) if (target/'official_metrics.json').exists() else None
    # Reused inference deliberately has no scorer output; validate using an
    # independent copy of its already verified source scorer receipt.
    assert checked is None
    (target/'official_metrics.json').write_bytes((source/'official_metrics.json').read_bytes())
    checked = validate_run(target, job, checkpoint)
    assert (target/'result.json').stat().st_ino != (source/'result.json').stat().st_ino
    assert (target/'official_metrics.json').stat().st_ino != (source/'official_metrics.json').stat().st_ino
    assert all((target/name).stat().st_ino == (source/name).stat().st_ino
        for name in ('frame_decisions.jsonl', 'predictions.txt', 'tracks.mp4'))
    result = dict(status='REGISTERED_FUTURE_BINDINGS_AND_REAL_COMPLETED_NATIVE_RUN_REUSE_VALIDATED',
        binding_checks=bindings, completed_run=checked, reuse=receipt,
        large_artifacts_same_inodes_no_new_feature_archive=True,
        result_and_scorer_files_independent=True, copied_scorer_receipt_not_fresh_score=True,
        no_GPU_or_model_calls=True, current_training_or_evaluations_changed=False,
        additional_small_artifact_reserve_bytes=524288,
        combined_registered_feature_upper_bytes=26840369872,
        global_temporary_feature_budget_bytes=26843545600, full_goal_complete=False)
    (args.out/'validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], frames=checked['frames'], native_candidates=checked['native_candidates'])), flush=True)


if __name__ == '__main__':
    main()
