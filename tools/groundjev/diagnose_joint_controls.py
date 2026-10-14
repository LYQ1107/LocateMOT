"""Validate paired artifacts and diagnose RMOT output loss after inference."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from locatemot.groundjev.decision import overlap
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=ROOT/'outputs/groundjev/compact/joint1200_controls_v1')
    root = parser.parse_args().root
    owner = json.loads((root / 'owner_state.json').read_text())
    assert owner['status'] == 'JOINT_THREE_TASK_NATIVE_PROPOSAL_CONTROLS_COMPLETE'
    scopes = {}
    for name in ('mot4', 'rmot_single8', 'rmot_multi12', 'ovmot4'):
        before = Path(owner['completed'][name + '_before']['run'])
        if not before.is_absolute():
            before = ROOT / before
        after = Path(owner['completed'][name + '_after']['run'])
        rows = [[json.loads(x) for x in (p / 'frame_decisions.jsonl').open()] for p in (before, after)]
        assert [r['frame'] for r in rows[0]] == [r['frame'] for r in rows[1]]
        for a, b in zip(*rows):
            assert a['image_path'] == b['image_path'] and a['candidates'] == b['candidates']
            assert b['state']['decision_metadata']['GT_read'] is False
        videos = []
        for directory in (before, after):
            video = next(directory.glob('*.mp4'))
            info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
                '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames',
                '-of', 'json', str(video)], text=True))
            count = int(info['streams'][0]['nb_read_frames'])
            assert count == len(rows[0])
            videos.append(count)
        result = {'frames': len(rows[0]), 'identical_native_candidates_and_ROIs': True,
                  'decoded_video_frames': videos, 'runtime_GT_read': False}
        if name.startswith('rmot'):
            manifest_name = 'rmot_single_empty.json' if name == 'rmot_single8' else 'rmot_multi_empty.json'
            manifest = json.loads((ROOT / 'outputs/groundjev/multitask_pilot_data' / manifest_name).read_text())
            query = json.loads(Path(manifest['expression_file_for_evaluation_only']).read_text())
            width, height = manifest['image_size']
            covered, suppressed, emitted = 0, 0, 0
            probabilities = []
            for row in rows[1]:
                ids = set(query['label'].get(str(row['frame']), []))
                boxes = []
                if ids:
                    path = Path(manifest['label_dir_for_evaluation_only']) / f"{row['frame']:06d}.txt"
                    for line in path.read_text().splitlines():
                        f = line.split()
                        if int(f[1]) in ids:
                            x, y, w, h = map(float, f[2:6])
                            boxes.append([x*width, y*height, (x+w)*width, (y+h)*height])
                decisions = {e['candidate']: e['decision'] for e in row['state']['events'] if 'candidate' in e and 'decision' in e}
                for i, candidate in enumerate(row['candidates']):
                    if any(overlap(candidate['box'], box) >= .5 for box in boxes):
                        covered += 1
                        probability = decisions[i]['query_membership_probability']
                        probabilities.append(probability)
                        suppressed += probability < .5
                        emitted += any(p['box'] == candidate['box'] for p in row['outputs'])
            result['posthoc_query_diagnostic'] = {
                'GT_covered_candidate_observations': covered,
                'membership_below_registered_half_threshold': suppressed,
                'covered_candidate_boxes_emitted': emitted,
                'covered_membership_probabilities': probabilities,
                'evaluation_only_never_corrects_predictions': True,
                'no_threshold_tuning': True}
        scopes[name] = result
    receipt = {'status': 'PAIRED_ARTIFACTS_VALIDATED', 'scopes': scopes,
               'official_metrics': {k: v['metrics']['metrics'] for k, v in owner['completed'].items()},
               'all_scopes_are_registered_short_clips_not_full_datasets': True}
    (root / 'validation_and_diagnostic.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
