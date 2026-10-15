"""Separate proposal-policy control: exact-box aliases, unchanged Simple/Compact."""
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
    raw = ROOT/'outputs/groundjev/multitask_runs/tao_base_novel63_full_v2_expression_none/native_candidates/frame_candidates.jsonl'
    reference = ROOT/'outputs/groundjev/multitask_runs/tao_base_novel63_full_v3_empty_crop_cpu'
    assert json.loads((reference/'owner_state.json').read_text())['status'] == 'FULL40_BASE_NOVEL_NATIVE_SOURCE_CONTROLS_COMPLETE'
    manifest = ROOT/'outputs/groundjev/multitask_pilot_data/ovmot_tao_base_novel63_full.json'
    policy = {'group_key': 'exact unrounded native xyxy coordinate tuple',
        'representative': 'first original native candidate index; original box/label/score/feature retained',
        'all_alias_indices_labels_and_scores_preserved': True,
        'every_identical_box_ROI_feature_checked_exact': True,
        'IoU_threshold_NMS_topK_or_GT_selection': False,
        'different_proposal_protocol_from_raw_reference': True,
        'association_only_or_JEV_causal_gain_claim': False}
    state = {'status': 'BUILDING_SEPARATE_EXACT_GEOMETRY_ALIAS_SOURCE', 'owner_pid': os.getpid(),
        'device': 'cpu', 'additional_GPUs': 0, 'policy': policy, 'completed': {},
        'raw_native_source': str(raw), 'raw_reference': str(reference),
        'weights_thresholds_metrics_and_all40frames_unchanged': True,
        'GT_used_for_grouping': False, 'new_feature_budget_bytes': 2**30}
    def save():
        (args.out/'owner_state.json').write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps({'status': state['status'], 'completed': list(state['completed'])}), flush=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT/'external/OneJev'))
    def run(stage, command):
        state['status'] = stage
        save()
        with (args.out/(stage+'.log')).open('w') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    save()
    try:
        source = args.out/'frame_candidates.jsonl'
        counts = {'frames': 0, 'raw_native_candidates': 0, 'exact_geometry_groups': 0, 'alias_observations': 0}
        with source.open('w') as output, raw.open() as original:
            for line in original:
                row = json.loads(line)
                groups = {}
                for index, candidate in enumerate(row['candidates']):
                    groups.setdefault(tuple(candidate['box']), []).append((index, candidate))
                merged = []
                for aliases in groups.values():
                    representative = aliases[0][1]
                    for _, candidate in aliases:
                        assert candidate['native_region_feature'] == representative['native_region_feature']
                        assert candidate.get('temporal_region_feature') == representative.get('temporal_region_feature')
                    merged.append(dict(representative, native_exact_geometry_aliases=[
                        {'raw_candidate_index': index, 'label': candidate['label'],
                         'native_score': candidate['native_score'], 'score_kind': candidate['score_kind']}
                        for index, candidate in aliases]))
                perception = dict(row['perception'], candidate_grouping=policy,
                                  raw_native_candidates=len(row['candidates']), exact_geometry_groups=len(merged))
                output.write(json.dumps(dict(row, candidates=merged, perception=perception))+'\n')
                output.flush()
                assert output.tell() <= state['new_feature_budget_bytes']
                counts['frames'] += 1
                counts['raw_native_candidates'] += len(row['candidates'])
                counts['exact_geometry_groups'] += len(merged)
                counts['alias_observations'] += sum(len(g) for g in groups.values())
        assert counts == {'frames': 40, 'raw_native_candidates': 35228,
                          'exact_geometry_groups': 7343, 'alias_observations': 35228}
        state['source'] = dict(counts, file_bytes=source.stat().st_size)
        save()
        for name, backend in [('simple', 'simple'), ('shared_compact', 'compact')]:
            output = args.out/name
            command = [str(ROOT/'.venv-onejev-py310/bin/python'), 'tools/groundjev/run_video.py', '--task', 'ovmot',
                '--video', str(manifest), '--backend', backend, '--candidate-replay', str(source),
                '--omit-replay-feature-vectors', '--out', str(output)]
            if backend == 'compact':
                command += ['--compact-device', 'cpu', '--checkpoint',
                    str(ROOT/'outputs/groundjev/compact/rmot_native72_stage_v1/broad_native_training/checkpoint.pt')]
            run(name+'_INFERENCE', command)
            run(name+'_OFFICIAL_TETA', ['python3', 'tools/groundjev/evaluate_multitask.py', '--video', str(manifest), '--run', str(output)])
            run(name+'_POSTHOC', ['python3', 'tools/groundjev/analyze_tracking_events.py', '--video', str(manifest), '--run', str(output)])
            rows = [json.loads(line) for line in (output/'frame_decisions.jsonl').open()]
            m = json.loads(manifest.read_text())
            assert [r['frame'] for r in rows] == m['frame_numbers'] and len(rows) == 40
            with source.open() as stream:
                for actual, expected in zip(rows, map(json.loads, stream)):
                    geometry = [{k: v for k, v in c.items() if k not in ('native_region_feature', 'temporal_region_feature')}
                                for c in expected['candidates']]
                    assert actual['candidates'] == geometry and actual['image_path'] == expected['image_path']
            decoded = int(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
                '-show_entries', 'stream=nb_read_frames', '-of', 'default=noprint_wrappers=1:nokey=1', str(output/'tracks.mp4')], text=True))
            assert decoded == 40
            official = json.loads((output/'official_metrics.json').read_text())
            state['completed'][name] = {'run': str(output), 'decoded_frames': decoded,
                'metrics': official['metrics'], 'base_novel': official['base_novel'],
                'all7343group_inputs_and35228aliases_preserved': True,
                'original_reference_metrics': json.loads((reference/name/'official_metrics.json').read_text())['metrics']}
            save()
        state['status'] = 'FULL40_EXACT_GEOMETRY_ALIAS_PROPOSAL_POLICY_CONTROLS_COMPLETE'
        save()
    except Exception as error:
        state.update(status='TAO_EXACT_ALIAS_CONTROL_FAILED', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
