"""Training-fit membership and expression sensitivity, never validation tuning."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from locatemot.groundjev.compact import CompactCore
    torch.set_num_threads(4)
    base = ROOT / 'outputs/groundjev/compact'
    pack = torch.load(base / 'joint_pack_v3/examples.pt', weights_only=True)
    model = CompactCore(semantic=True)
    model.load_state_dict(torch.load(base / 'joint1200_v1/checkpoint.pt', weights_only=True)['state_dict'])
    model.eval()
    counts = {'positive': 0, 'negative': 0, 'correct': 0, 'total': 0}
    deltas = []
    rows = [e for e in pack['examples'] if e['task_index'] == 1]
    for i, e in enumerate(rows):
        e = dict(e, class_text=pack['class_text'])
        with torch.no_grad():
            p = model(e)['membership'].sigmoid()
        target = e['membership_target']
        counts['positive'] += int(target.sum())
        counts['negative'] += int((1-target).sum())
        counts['correct'] += int(((p >= .5) == target.bool()).sum())
        counts['total'] += len(target)
        others = [x for x in rows if not torch.equal(x['spec_text'], e['spec_text'])]
        if others:
            altered = dict(e, spec_text=others[i % len(others)]['spec_text'])
            with torch.no_grad():
                q = model(altered)['membership'].sigmoid()
            deltas.extend((p-q).abs().tolist())
    result = {'scope': 'training-fit teacher proposals/history only; not validation performance',
        'counts': counts, 'training_membership_accuracy': counts['correct']/counts['total'],
        'different_training_expression_mean_absolute_probability_change': sum(deltas)/len(deltas),
        'different_training_expression_max_absolute_probability_change': max(deltas),
        'swapped_expression_correctness_not_scored': True, 'test_predictions_or_threshold_modified': False}
    (base / 'joint1200_controls_v1/training_fit_diagnostic.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
