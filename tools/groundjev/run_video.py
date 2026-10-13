"""Stream real video frames through DLM + Simple/Full persistent tracking."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from PIL import Image, ImageDraw
    from locatemot.groundjev.perception import GroundAnythingAdapter
    from locatemot.groundjev.decision import OneJevDecision, SimpleDecision
    from locatemot.groundjev.tracker import OnlineTracker
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['mot', 'rmot', 'ovmot'], default='mot')
    parser.add_argument('--video', type=Path, required=True, help='legal JSON video scope manifest')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'models/OneJev-0.8B')
    parser.add_argument('--backend', choices=['simple', 'full'], default='full')
    parser.add_argument('--expression')
    parser.add_argument('--vocabulary', nargs='+')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, help='explicit short functional check; never a full-video metric')
    parser.add_argument('--grounding-url', default='http://127.0.0.1:8101/v1')
    parser.add_argument('--candidate-replay', type=Path, help='exact prior frame log for a fixed-candidate association control; no end-to-end FPS claim')
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('choose a fresh output directory to preserve existing experiment results')
    manifest = json.loads(args.video.read_text())
    vocabulary = args.vocabulary or manifest['vocabulary']
    if args.task == 'rmot' and not args.expression:
        parser.error('RMOT requires --expression')
    args.out.mkdir(parents=True, exist_ok=True)
    frontend = GroundAnythingAdapter(args.grounding_url)
    decision = SimpleDecision() if args.backend == 'simple' else OneJevDecision(args.checkpoint)
    tracker = OnlineTracker(decision)
    frames = list(zip(manifest['images'], manifest['frame_numbers']))
    if args.limit:
        frames = frames[:args.limit]
    replay = {}
    if args.candidate_replay:
        replay = {row['frame']: row for row in map(json.loads, args.candidate_replay.read_text().splitlines())}
        if any(number not in replay for _, number in frames):
            parser.error('replay lacks a requested frame')
    fps, latencies = manifest['fps'], []
    video_writer = None
    try:
        with (args.out / 'predictions.txt').open('w') as predictions, (args.out / 'frame_decisions.jsonl').open('w') as log:
            for path, frame_number in frames:
                start = time.perf_counter()
                with Image.open(path) as source:
                    image = source.convert('RGB')
                if replay:
                    original = replay[frame_number]
                    if original['image_path'] != path:
                        raise ValueError('Fixed-candidate replay uses a different physical image')
                    candidates, perception = original['candidates'], {'mode': 'fixed_candidate_replay',
                        'source': str(args.candidate_replay), 'original_perception': original['perception']}
                else:
                    candidates, perception = frontend(image, args.task, vocabulary, args.expression, image_path=path)
                outputs, state = tracker.step(image, candidates, frame_number/fps,
                    {'task': args.task, 'vocabulary': vocabulary, 'expression': args.expression})
                elapsed = time.perf_counter()-start
                latencies.append(elapsed)
                for row in outputs:
                    x1,y1,x2,y2 = row['box']
                    # MOTChallenge requires a confidence column; 1 here is an
                    # explicit unscored-output sentinel, not detector confidence.
                    predictions.write(f'{frame_number},{row["track_id"]},{x1:.3f},{y1:.3f},{x2-x1:.3f},{y2-y1:.3f},1,-1,-1,-1\n')
                log.write(json.dumps({'frame': frame_number, 'image_path': path, 'seconds': elapsed, 'candidates': candidates,
                                     'outputs': outputs, 'state': state, 'perception': perception})+'\n')
                log.flush()
                predictions.flush()
                canvas = image.copy()
                draw = ImageDraw.Draw(canvas)
                for row in outputs:
                    draw.rectangle(row['box'], outline='lime', width=2)
                    draw.text((row['box'][0], row['box'][1]), str(row['track_id']), fill='red')
                if video_writer is None:
                    video_writer = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                        '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{image.width}x{image.height}',
                        '-framerate', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                        str(args.out/'tracks.mp4')], stdin=subprocess.PIPE)
                video_writer.stdin.write(canvas.tobytes())
                print(json.dumps({'frame': frame_number, 'seconds': elapsed, 'tracks': len(outputs)}), flush=True)
        video_writer.stdin.close()
        if video_writer.wait() != 0:
            raise RuntimeError('Visualization encoder failed')
    finally:
        if video_writer is not None and video_writer.poll() is None:
            video_writer.terminate()
            video_writer.wait()
    result = {'status': 'VIDEO_INFERENCE_COMPLETE', 'task': args.task, 'backend': args.backend,
              'scope': 'short_functional_check' if args.limit else manifest['scope'],
              'video_manifest': str(args.video), 'frames': len(frames), 'latencies_seconds': latencies,
              'end_to_end_fps': None if replay else len(frames)/sum(latencies),
              'association_replay_fps': len(frames)/sum(latencies) if replay else None,
              'official_metrics': 'PENDING_OFFICIAL_EVALUATOR',
              'checkpoint': str(args.checkpoint) if args.backend == 'full' else None,
              'motchallenge_confidence_column': 'constant unscored-output sentinel; native detection score is absent',
              'gt_read_by_runtime': False, 'future_frames_read_by_runtime': False}
    if args.backend == 'full':
        import torch
        result['peak_decision_gpu_memory_bytes'] = torch.cuda.max_memory_allocated()
    (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    (args.out/'final_identity_memory.json').write_text(json.dumps(tracker.state_dict())+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
