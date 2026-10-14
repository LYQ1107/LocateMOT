"""Finite owner for the preregistered joint-core full-video retention check."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/groundjev/compact/joint1200_dancetrack0004_v1')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'outputs/groundjev/compact/joint1200_v1/checkpoint.pt')
    parser.add_argument('--resume-from',type=Path)
    args=parser.parse_args();out = args.out
    out.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'outputs/groundjev/compact/dancetrack0004_native300_v2_empty_roi_fixed/compact_distilled'
    result = json.loads((source / 'result.json').read_text())
    assert result['frames'] == 1203
    state = {'status': 'JOINT_CORE_FULL_VIDEO_RETENTION_STARTED', 'physical_gpu': 7,
        'registered_frames': 1203, 'seed': 0, 'candidate_and_ROI_replay': str(source),
        'same_native_proposals_as_Compact300_and_matched_controls': True,
        'scope': 'one complete legal DanceTrack validation video, not full dataset',
        'GT_parser_metrics_unchanged': True, 'end_to_end_fps': None}
    if args.resume_from:state['resume_from']=str(args.resume_from)
    def save():
        (out / 'owner_state.json').write_text(json.dumps(state, indent=2) + '\n')
        print(json.dumps(state), flush=True)
    save()
    directory = out / 'joint1200'
    commands = [
        ['python3', 'tools/groundjev/run_with_resources.py', '--gpus', '7', '--',
         str(ROOT / '.venv-onejev-py310/bin/python'), 'tools/groundjev/run_video.py',
         '--task', 'mot', '--video', 'outputs/groundjev/pilot_data/dancetrack.json',
         '--backend', 'compact', '--checkpoint', str(args.checkpoint),
         '--candidate-replay', str(source / 'frame_decisions.jsonl'), '--out', str(directory)],
        ['python3', 'tools/groundjev/evaluate_video.py', '--video',
         'outputs/groundjev/pilot_data/dancetrack.json', '--run', str(directory)]]
    if args.resume_from:commands[0]+=['--resume-from',str(args.resume_from)]
    try:
        for stage, command in zip(('INFERENCE', 'OFFICIAL_EVALUATION'), commands):
            state['status'] = stage; save()
            with (out / (stage + '.log')).open('w') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        metrics = json.loads((directory / 'official_metrics.json').read_text())
        assert metrics['frames'] == 1203
        a, b = (source / 'frame_decisions.jsonl').open(), (directory / 'frame_decisions.jsonl').open()
        count = 0
        from itertools import zip_longest
        for left, right in zip_longest(a, b):
            assert left is not None and right is not None
            left, right = json.loads(left), json.loads(right)
            assert left['frame'] == right['frame'] and left['candidates'] == right['candidates']
            assert right['state']['decision_metadata']['GT_read'] is False
            count += 1
        assert count == 1203
        video = next(directory.glob('*.mp4'))
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
            '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames', '-of', 'json', str(video)], text=True))
        assert int(probe['streams'][0]['nb_read_frames']) == 1203
        state.update(status='JOINT_CORE_FULL_VIDEO_RETENTION_COMPLETE', metrics=metrics,
                     identical_candidate_ROI_rows=count, decoded_video_frames=1203)
        save()
    except Exception as error:
        state.update(status='JOINT_CORE_FULL_VIDEO_RETENTION_FAILED', error=str(error)); save()
        raise


if __name__ == '__main__':
    main()
