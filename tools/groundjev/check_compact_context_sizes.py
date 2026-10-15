"""Real dense-TAO prefix: CLI default preservation and 64/128 option execution."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    plan = json.loads((ROOT/'configs/groundjev/compact_context_sizes_v1.json').read_text())
    job = plan['scopes']['tao_alias_full40']
    manifest = json.loads((ROOT/job['manifest']).read_text())
    manifest.update(images=manifest['images'][:2], frame_numbers=manifest['frame_numbers'][:2],
        image_ids=manifest['image_ids'][:2], scope='explicit_dense_context_functional_prefix_only')
    manifest_path = args.out/'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT/'external/OneJev'))
    python = str(ROOT/'.venv-onejev-py310/bin/python')
    reports, traces = {}, {}
    for name, size in [('default32', None), ('explicit32', 32), ('explicit64', 64), ('explicit128', 128)]:
        command = [python, 'tools/groundjev/run_video.py', '--task', 'ovmot', '--video', str(manifest_path),
            '--backend', 'compact', '--compact-device', 'cpu', '--checkpoint', str(ROOT/plan['checkpoint']),
            '--candidate-replay', str(ROOT/job['candidates']), '--omit-replay-feature-vectors',
            '--memory-checkpoint-format', 'binary', '--memory-export-interval', '0', '--out', str(args.out/name)]
        if size:
            command += ['--compact-shortlist', str(size)]
        with (args.out/(name+'.log')).open('w') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, check=True)
        reports[name] = json.loads((args.out/name/'result.json').read_text())
        traces[name] = [json.loads(line) for line in (args.out/name/'frame_decisions.jsonl').open()]
        expected = size or 32
        assert reports[name]['compact_shortlist_limit'] == expected and reports[name]['end_to_end_fps'] is None
        assert len(traces[name]) == 2
        first, second = traces[name]
        assert len(first['state']['persistent_ids']) > 128
        for event in second['state']['events']:
            if 'candidate' in event:
                assert len(event['decision']['shortlisted_ids']) == expected
        assert not reports[name]['gt_read_by_runtime'] and not reports[name]['future_frames_read_by_runtime']
    assert all(a['outputs'] == b['outputs'] and a['state'] == b['state']
        for a, b in zip(traces['default32'], traces['explicit32']))
    report = {'status': 'REAL_DENSE_TAO_CONTEXT32_64_128_FUNCTIONAL_VALIDATED',
        'actual_frames_each': 2, 'actual_live_IDs_before_second_frame': len(traces['default32'][0]['state']['persistent_ids']),
        'default32_vs_explicit32_all_predictions_and_state_exact': True,
        'every_second_frame_candidate_has_exact_requested32_64_128_options': True,
        'GT_or_future_read': False, 'official_accuracy_or_native_E2E_claim': False}
    (args.out/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
