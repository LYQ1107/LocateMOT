"""Actual dense native video checks for a separate fixed-size retrieval policy."""
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
    original = ROOT/'outputs/groundjev/validation/compact_context_sizes_v1'
    manifest = original/'manifest.json'
    source = ROOT/'outputs/groundjev/multitask_runs/tao_exact_geometry_alias40_v1/frame_candidates.jsonl'
    checkpoint = ROOT/'outputs/groundjev/compact/rmot_native72_stage_v1/broad_native_training/checkpoint.pt'
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT/'external/OneJev'),
        OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    traces = {}
    for policy in ('geometry', 'hybrid16_16'):
        command = [str(ROOT/'.venv-onejev-py310/bin/python'), 'tools/groundjev/run_video.py',
            '--task', 'ovmot', '--video', str(manifest), '--backend', 'compact', '--compact-device', 'cpu',
            '--checkpoint', str(checkpoint), '--candidate-replay', str(source), '--compact-shortlist', '32',
            '--compact-shortlist-policy', policy, '--omit-replay-feature-vectors',
            '--memory-checkpoint-format', 'binary', '--memory-export-interval', '0', '--out', str(args.out/policy)]
        with (args.out/(policy+'.log')).open('w') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads((args.out/policy/'result.json').read_text())
        assert report['frames'] == 2 and report['compact_shortlist_limit'] == 32
        assert report['compact_shortlist_policy'] == policy and report['end_to_end_fps'] is None
        assert not report['gt_read_by_runtime'] and not report['future_frames_read_by_runtime']
        traces[policy] = [json.loads(line) for line in (args.out/policy/'frame_decisions.jsonl').open()]
    previous = [json.loads(line) for line in (original/'explicit32/frame_decisions.jsonl').open()]
    assert all(a['outputs'] == b['outputs'] and a['state'] == b['state']
        for a, b in zip(traces['geometry'], previous))
    assert traces['geometry'][0]['outputs'] == traces['hybrid16_16'][0]['outputs']
    live = traces['geometry'][0]['state']['persistent_ids']
    assert len(live) == 251
    events = [{e['candidate']: e for e in traces[p][1]['state']['events'] if 'candidate' in e}
        for p in ('geometry', 'hybrid16_16')]
    assert len(events[0]) == len(events[1]) == 230
    changed = 0
    for candidate, event in events[0].items():
        old = event['decision']['shortlisted_ids']
        new = events[1][candidate]['decision']['shortlisted_ids']
        assert len(new) == len(set(new)) == 32 and set(new) <= set(live)
        assert new[:16] == old[:16]
        changed += set(new) != set(old)
    report = {'status': 'REAL_DENSE_TAO_HYBRID32_FUNCTIONAL_VALIDATED',
        'frames_each': 2, 'actual_live_IDs_before_second_frame': len(live),
        'actual_second_frame_candidates': 230,
        'default_geometry_all_predictions_and_states_exact_vs_prior_version': True,
        'every_hybrid_shortlist32_unique_live_IDs_and_first16_geometry_exact': True,
        'second_frame_candidates_with_changed_identity_set': changed,
        'GT_or_future_used': False, 'official_accuracy_or_E2E_claim': False}
    (args.out/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
