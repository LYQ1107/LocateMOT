"""Check real-pack evidence isolation and unchanged default model numerics."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from locatemot.groundjev.compact import CompactCore
    torch.set_num_threads(4)
    pack = torch.load(ROOT/'outputs/groundjev/compact/rmot_native72_v1/broad_native/examples.pt', weights_only=True)
    checkpoint = torch.load(ROOT/'outputs/groundjev/compact/task_subsets1200_cpu_v1/all_three_training/checkpoint.pt', weights_only=True)
    namespace = {}
    source = subprocess.check_output(['git', 'show', 'e9111b5:locatemot/groundjev/compact.py'], cwd=ROOT, text=True)
    exec(compile(source, 'previous_compact.py', 'exec'), namespace)
    old = namespace['CompactCore'](semantic=True).eval()
    old.load_state_dict(checkpoint['state_dict'])
    full = CompactCore(semantic=True).eval()
    full.load_state_dict(checkpoint['state_dict'])
    examples = [next(e for e in pack['examples'] if e['task_index'] == task and len(e['candidates']) > 1 and len(e['history']) > 1) for task in range(3)]
    rows = []
    for original in examples:
        e = dict(original, class_text=pack['class_text'])
        with torch.no_grad():
            before, after = old(e), full(e)
        assert all(torch.equal(before[k], after[k]) for k in before)
        visual = dict(e, candidates=e['candidates'].flip(0), history=e['history'].flip(0))
        alternate = next(x['spec_text'] for x in pack['examples'] if not torch.equal(x['spec_text'], e['spec_text']))
        language = dict(e, spec_text=alternate)
        for mode in ('geometry_motion', 'geometry_appearance'):
            model = CompactCore(semantic=True, input_evidence=mode).eval()
            model.load_state_dict(checkpoint['state_dict'])
            with torch.no_grad():
                reference, no_specification = model(e), model(language)
                altered_visual = model(visual)
            assert all(torch.equal(reference[k], no_specification[k]) for k in reference)
            if mode == 'geometry_motion':
                assert all(torch.equal(reference[k], altered_visual[k]) for k in reference)
                sum(v.sum() for v in model(e).values()).backward()
                assert all(p.grad is None for p in model.visual.parameters())
                assert all(p.grad is None for p in model.history_merge.parameters())
            else:
                assert not torch.equal(reference['match'], altered_visual['match'])
            rows.append({'task': original['task_index'], 'mode': mode, 'parameters': sum(p.numel() for p in model.parameters()),
                         'runtime_specification_blocked_exactly': True, 'appearance_blocked_exactly': mode == 'geometry_motion'})
    assert {r['parameters'] for r in rows} == {1146511}
    report = {'status': 'REAL_THREE_TASK_INPUT_EVIDENCE_ISOLATION_AND_DEFAULT_EXACT', 'results': rows,
              'default_all_outputs_equal_previous_committed_model': True,
              'class_vocabulary_output_head_text_retained_all_modes': True,
              'scope': 'Functional real-pack permutations only; no new video accuracy or speed claim.'}
    out = ROOT/'outputs/groundjev/validation/input_evidence_v1'
    out.mkdir(parents=True, exist_ok=True)
    (out/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(report['status'])


if __name__ == '__main__':
    main()
