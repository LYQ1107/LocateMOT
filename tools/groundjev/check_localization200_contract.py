"""Compare real native localization rows/schedule to the completed tracking200 run."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from tools.groundjev.train_native_ground_tracking200 import training_schedule
    from tools.groundjev.native_ground_training_data import build_pairs
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    original = json.loads((ROOT/'configs/groundjev/native_ground_tracking200_gpu6_20261014.json').read_text())
    control = json.loads((ROOT/'configs/groundjev/native_ground_localization200_v1.json').read_text())
    before, a = build_pairs(original)
    after, b = build_pairs(control)
    assert before == after and a == b
    schedule = training_schedule(control, after)
    assert schedule == training_schedule(original, before)
    checkpoint = ROOT/'outputs/groundjev/ground_adaptation/native_tracking200_v7_boundary_trace/projector_delta.pt'
    actual = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert actual['steps'] == 200 and actual['protocol'] == original and actual['schedule'] == schedule
    for key in ('seed', 'steps', 'lr', 'weight_decay', 'trainable_policy', 'native_loss'):
        assert original[key] == control[key]
    assert control['auxiliary_losses']['teacher_GT_same_ID_cross_frame_contrastive'] == 0
    assert control['auxiliary_losses']['actual_native_proposal_same_ID_cross_frame_contrastive'] == 0
    assert original['auxiliary_losses']['teacher_GT_same_ID_cross_frame_contrastive'] == .1
    assert original['auxiliary_losses']['actual_native_proposal_same_ID_cross_frame_contrastive'] == .1
    report = {'status': 'REAL_LOCALIZATION200_ROWS_AND_ACTUAL_TRACKING200_SCHEDULE_MATCH_VALIDATED',
        'corpus': a, 'same_all112_actual_image_bytes_prompt_answers_boxes_native_alignment': True,
        'same_all200_scheduled_pair_indices_vs_actual_completed_checkpoint': True,
        'reference_checkpoint': str(checkpoint), 'same_seed_steps_optimizer_lr_and_initialization': True,
        'new_identity_loss_coefficients': [0., 0.], 'reference_identity_loss_coefficients': [.1, .1],
        'localization_training_started_or_accuracy_claim': False,
        'comparison_limit': 'Reference tracking200 had interrupted/resumed execution; same data/schedule does not imply bitwise CUDA trajectory or matched wall time.',
        'original_dense_ref_benchmark_GT_used_for_training': False, 'new_feature_cache_files': False}
    (args.out/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
