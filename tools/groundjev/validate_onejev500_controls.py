"""Completed formal500 short controls: actual encoded videos and metric provenance."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT/'outputs/groundjev/training/onejev500_completion_controls_v1')
    args = parser.parse_args()
    owner = json.loads((args.root/'owner_state.json').read_text())
    assert owner['status'] == 'ONEJEV500_MATCHED_OFFICIAL_CONTROLS_COMPLETE' and len(owner['completed']) == 16
    result = {'status': 'ALL16_ORIGINAL500_MATCHED_SHORT_CONTROLS_VALIDATED',
        'actual_training_steps': 500, 'actual_training_rows': 2000, 'decoded_video_frames': 0,
        'all_registered_frames_and_exact_frozen_native_candidates_retained': True,
        'GT_and_future_read_by_runtime': False, 'Novel_GT_in_this_TAO_clip': False,
        'scope': 'MOT4/RMOTsingle8/RMOTmulti12/TAOBase4, four models; not complete videos or full datasets',
        'inference_protocol': 'legacy v3_crop_motion/None native membership gate/native_label; new explicit semantic policy is a separate before-after comparison',
        'grounding_provenance': 'frozen source candidates and per-frame perception metadata; grounding CLI defaults are unused on replay',
        'end_to_end_or_standalone_speed_claim': False, 'runs': {}}
    for name, unit in owner['completed'].items():
        run = Path(unit['run'])
        rows = [json.loads(line) for line in (run/'frame_decisions.jsonl').open()]
        report = json.loads((run/'result.json').read_text())
        assert report['all_input_frames_processed']
        assert report['gt_read_by_runtime'] is False and report['future_frames_read_by_runtime'] is False
        decoded = int(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
            '-show_entries', 'stream=nb_read_frames', '-of', 'default=noprint_wrappers=1:nokey=1', str(run/'tracks.mp4')], text=True))
        assert decoded == len(rows) == report['frames']
        assert unit['all_exact_native_candidate_rows_preserved']
        result['decoded_video_frames'] += decoded
        result['runs'][name] = {'run': str(run), 'frames': decoded, 'metrics': unit['official_metrics']['metrics'],
            'metric_scale': unit['official_metrics'].get('metric_scale'),
            'association_replay_fps_not_E2E': report['association_replay_fps'],
            'peak_decision_gpu_memory_bytes': report.get('peak_decision_gpu_memory_bytes')}
    assert result['decoded_video_frames'] == 112
    (args.root/'validation_and_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'runs'}, indent=2))


if __name__ == '__main__':
    main()
