"""Actual intermediate125 weight changes and repeated native multimodal reload."""
import gc
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'external/OneJev')]


def main():
    import torch
    from PIL import Image
    from safetensors import safe_open
    from locatemot.groundjev.decision import OneJevDecision, SimpleDecision, overlap
    from locatemot.groundjev.tracker import OnlineTracker
    from tools.groundjev.run_video import restore_completed_prefix
    from qev.mm_engine import MMDecisionEngine
    training = ROOT/'outputs/groundjev/training/onejev_own_history_v3_exact500_replay_stateful'
    checkpoint = training/'ckpt-step125'
    initial = ROOT/'outputs/groundjev/training/onejev_teacher_check_100/final'
    probes = []
    with safe_open(initial/'model.safetensors', framework='pt') as before, safe_open(checkpoint/'model.safetensors', framework='pt') as after:
        assert set(before.keys()) == set(after.keys())
        entries = len(after.keys())
        names = [k for k in before.keys() if '.language_model.layers.' in k and k.endswith('mlp.gate_proj.weight')]
        for name in names[::max(1, len(names)//3)][:3]:
            a, b = before.get_tensor(name).bfloat16().float(), after.get_tensor(name).bfloat16().float()
            difference = (a-b).abs()
            probes.append({'name': name, 'changed_elements': int((difference>0).sum()), 'max_absolute_difference': float(difference.max())})
        assert all(p['changed_elements'] for p in probes)
    source = ROOT/'outputs/groundjev/runs/mot17_first_native_clip_repair2/frame_decisions.jsonl'
    with source.open() as stream:
        rows = [json.loads(next(stream)) for _ in range(2)]
    manifest = json.loads((ROOT/'outputs/groundjev/pilot_data/mot17_first4.json').read_text())
    tracker = OnlineTracker(SimpleDecision())
    specification = {'task': 'mot', 'vocabulary': ['person'], 'expression': None}
    restore_completed_prefix(tracker, rows[:1], list(zip(manifest['images'], manifest['frame_numbers'])), manifest['fps'], specification)
    with Image.open(rows[1]['image_path']) as raw:
        image = raw.convert('RGB')
    candidate = rows[1]['candidates'][:1]
    ranked = sorted(tracker.tracks.values(), key=lambda t: (-overlap(candidate[0]['box'], t.predicted_box(303/30)), t.track_id))[:3]
    builder = OneJevDecision.__new__(OneJevDecision); builder.evidence_format = 'v3_crop_motion'
    request, media = builder.build_request(image, candidate, ranked, 303/30, specification)
    request.questions = {k: v for k, v in request.questions.items()
        if k.startswith(('match_', 'update_', 'reliability_')) or k == f'life_{ranked[0].track_id}'}
    responses = []
    for _ in range(2):
        engine = MMDecisionEngine(str(checkpoint), device='cuda:0', dtype='bfloat16', cuda_graphs=False)
        response, _ = engine.decide(request, media=media)
        responses.append(response.model_dump()); del engine; gc.collect(); torch.cuda.empty_cache()
    maximum = 0.
    for key, answer in responses[0]['answers'].items():
        repeated = responses[1]['answers'][key]
        a = answer.get('probabilities') or {'noul': answer['noul']}
        b = repeated.get('probabilities') or {'noul': repeated['noul']}
        assert all(math.isfinite(x) for x in a.values())
        maximum = max(maximum, max(abs(value-b[label]) for label, value in a.items()))
    assert maximum <= 1e-6
    result = {'status': 'REAL_ONEJEV125_INTERMEDIATE_NATIVE_MM_RELOAD_VALIDATED', 'weight_entries': entries,
        'checkpoint': str(checkpoint), 'same_BF16_precision_weight_changes_vs_initial100': probes,
        'repeated_reload_probability_max_difference': maximum, 'reload_tolerance': 1e-6,
        'native_fork_auto_check_unchanged': True, 'GT_read_by_inference': False,
        'responses': responses, 'intermediate_only': True, 'formal500_complete': False,
        'tracking_accuracy_or_exact_FP32_training_resume_claim': False}
    (training/'checkpoint125_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'responses'}, indent=2))


if __name__ == '__main__':
    main()
