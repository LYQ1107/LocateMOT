"""Stream real video frames through DLM + Simple/Full persistent tracking."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def restore_completed_prefix(tracker, rows, frames, fps, specification):
    """Rehydrate exact prior memory from saved native decisions, without GT."""
    from PIL import Image
    original = tracker.decision
    class RecordedDecision:
        def decide(self, image, candidates, tracks, timestamp, specification):
            decisions = [None]*len(candidates)
            lifecycle = {}
            for event in self.row['state']['events']:
                if 'candidate' in event:
                    decisions[event['candidate']] = event['decision']
                if event.get('lifecycle_decision') is not None:
                    lifecycle[event['track_id']] = event['lifecycle_decision']
            if any(d is None for d in decisions):
                raise ValueError('Completed frame has incomplete native decision evidence')
            return decisions, lifecycle, self.row['state']['decision_metadata']
    replay = RecordedDecision()
    tracker.decision = replay
    try:
        for row, (path, number) in zip(rows, frames):
            if row['frame'] != number or row['image_path'] != path:
                raise ValueError('Resume prefix does not match the physical video')
            replay.row = row
            with Image.open(path) as source:
                image = source.convert('RGB')
            outputs, state = tracker.step(image, row['candidates'], number/fps, specification)
            if outputs != row['outputs'] or state != row['state']:
                raise ValueError('Native decision replay did not restore identical causal outputs/state')
    finally:
        tracker.decision = original


def main():
    from PIL import Image, ImageDraw
    from locatemot.groundjev.perception import GroundAnythingAdapter, GroundingProtocolError
    from locatemot.groundjev.decision import OneJevDecision, SimpleDecision
    from locatemot.groundjev.tracker import OnlineTracker
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['mot', 'rmot', 'ovmot'], default='mot')
    parser.add_argument('--video', type=Path, required=True, help='legal JSON video scope manifest')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'models/OneJev-0.8B')
    parser.add_argument('--backend', choices=['simple', 'full'], default='full')
    parser.add_argument('--evidence-format',choices=['v1','v2_crop_motion','v3_crop_motion'],default='v1')
    parser.add_argument('--expression')
    parser.add_argument('--vocabulary', nargs='+')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, help='explicit short functional check; never a full-video metric')
    parser.add_argument('--grounding-url', default='http://127.0.0.1:8101/v1')
    parser.add_argument('--grounding-parser', choices=['strict_client', 'official_scoring'], default='strict_client',
                        help='explicit candidate protocol; official scorer preserves strict wire diagnostics')
    parser.add_argument('--grounding-decoder-policy', choices=['generic_greedy', 'tracking_dense'], default='generic_greedy')
    parser.add_argument('--perception-failure-policy', choices=['stop', 'empty_and_record'], default='stop',
                        help='official scorer rejection is an explicitly logged empty prediction, included in every metric denominator')
    parser.add_argument('--candidate-replay', type=Path, help='exact prior frame log for a fixed-candidate association control; no end-to-end FPS claim')
    parser.add_argument('--resume-from', type=Path, help='preserve source run and restore its completed causal prefix in this fresh output')
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('choose a fresh output directory to preserve existing experiment results')
    manifest = json.loads(args.video.read_text())
    vocabulary = args.vocabulary or manifest['vocabulary']
    if args.task == 'rmot' and not args.expression:
        parser.error('RMOT requires --expression')
    args.out.mkdir(parents=True, exist_ok=True)
    frontend = GroundAnythingAdapter(args.grounding_url, parser_policy=args.grounding_parser, decoder_policy=args.grounding_decoder_policy)
    decision = SimpleDecision() if args.backend == 'simple' else OneJevDecision(args.checkpoint,evidence_format=args.evidence_format)
    tracker = OnlineTracker(decision)
    frames = list(zip(manifest['images'], manifest['frame_numbers']))
    if args.limit:
        frames = frames[:args.limit]
    replay = {}
    if args.candidate_replay:
        replay = {row['frame']: row for row in map(json.loads, args.candidate_replay.read_text().splitlines())}
        if any(number not in replay for _, number in frames):
            parser.error('replay lacks a requested frame')
    fps, latencies, rejected_frames, strict_invalid_frames = manifest['fps'], [], [], []
    resumed = []
    specification = {'task': args.task, 'vocabulary': vocabulary, 'expression': args.expression}
    if args.resume_from:
        resumed = list(map(json.loads, (args.resume_from/'frame_decisions.jsonl').read_text().splitlines()))
        if len(resumed) > len(frames):
            raise ValueError('Resume prefix is longer than the requested video')
        restore_completed_prefix(tracker, resumed, frames, fps, specification)
        (args.out/'resume_memory.json').write_text(json.dumps(tracker.state_dict())+'\n')
        print(json.dumps({'restored_completed_frames': len(resumed), 'source': str(args.resume_from),
                          'identical_native_outputs_and_state': True, 'gt_used': False}), flush=True)
    video_writer = None
    try:
        with (args.out / 'predictions.txt').open('w') as predictions, (args.out / 'frame_decisions.jsonl').open('w') as log:
            for index, (path, frame_number) in enumerate(frames):
                start = time.perf_counter()
                with Image.open(path) as source:
                    image = source.convert('RGB')
                if index < len(resumed):
                    row = resumed[index]
                    candidates, perception, outputs, state = (row[k] for k in ('candidates','perception','outputs','state'))
                    elapsed = row['seconds']
                elif replay:
                    original = replay[frame_number]
                    if original['image_path'] != path:
                        raise ValueError('Fixed-candidate replay uses a different physical image')
                    candidates, perception = original['candidates'], {'mode': 'fixed_candidate_replay',
                        'source': str(args.candidate_replay), 'original_perception': original['perception'],
                        'model_response_rejected': original['perception'].get('model_response_rejected',False)}
                else:
                    print(json.dumps({'frame': frame_number, 'stage': 'native_perception'}), flush=True)
                    try:
                        candidates, perception = frontend(image, args.task, vocabulary, args.expression, image_path=path)
                    except GroundingProtocolError as error:
                        if args.perception_failure_policy == 'stop':
                            raise
                        candidates, perception = [], dict(error.metadata)
                        perception.update(model_response_rejected=True,
                            rejected_output_policy='empty official prediction set; frame retained for all metrics')
                        print(json.dumps({'frame': frame_number,'stage':'decoder_output_rejected',
                            'error':perception['result']['parse_error']}),flush=True)
                if perception.get('model_response_rejected'):
                    rejected_frames.append(frame_number)
                diagnostic = perception.get('original_perception',perception)
                if diagnostic.get('strict_wire_valid') is False:
                    strict_invalid_frames.append(frame_number)
                if index >= len(resumed):
                    print(json.dumps({'frame': frame_number, 'stage': 'association_and_memory', 'candidates': len(candidates)}), flush=True)
                    outputs, state = tracker.step(image, candidates, frame_number/fps, specification)
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
                if index >= len(resumed):
                    temporary = args.out/'latest_memory.tmp'
                    temporary.write_text(json.dumps(tracker.state_dict())+'\n')
                    temporary.replace(args.out/'latest_identity_memory.json')
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
    except Exception as error:
        failure = {'status': 'VIDEO_INFERENCE_FAILED', 'frame': frame_number,
                   'image_path': path, 'error_type': type(error).__name__, 'error': str(error),
                   'completed_frames': len(latencies), 'failed_frame_not_skipped': True}
        if isinstance(error, GroundingProtocolError):
            failure['perception'] = error.metadata
        (args.out/'failure.json').write_text(json.dumps(failure, indent=2)+'\n')
        raise
    finally:
        if video_writer is not None and video_writer.poll() is None:
            # On perception failure ffmpeg still waits for the next raw frame.
            # Close the pipe before waiting so cleanup cannot hide the exception.
            if video_writer.stdin is not None and not video_writer.stdin.closed:
                try:
                    video_writer.stdin.close()
                except BrokenPipeError:
                    pass
            video_writer.terminate()
            try:
                video_writer.wait(timeout=10)
            except subprocess.TimeoutExpired:
                video_writer.kill()
                video_writer.wait()
    result = {'status': 'VIDEO_INFERENCE_COMPLETE_WITH_REJECTED_PERCEPTION' if rejected_frames else 'VIDEO_INFERENCE_COMPLETE',
              'task': args.task, 'backend': args.backend,
              'evidence_format':args.evidence_format,
              'scope': 'short_functional_check' if args.limit else manifest['scope'],
              'video_manifest': str(args.video), 'frames': len(frames), 'latencies_seconds': latencies,
              'grounding_parser_policy': args.grounding_parser,
              'grounding_decoder_policy': args.grounding_decoder_policy,
              'perception_failure_policy': args.perception_failure_policy,
              'perception_rejected_frames': rejected_frames,
              'perception_rejection_rate': len(rejected_frames)/len(frames),
              'strict_wire_invalid_frames': strict_invalid_frames,
              'all_input_frames_processed': True,
              'end_to_end_fps': None if replay or resumed else len(frames)/sum(latencies),
              'successful_frame_pipeline_fps': len(frames)/sum(latencies),
              'association_replay_fps': len(frames)/sum(latencies) if replay else None,
              'resume_source': str(args.resume_from) if args.resume_from else None,
              'resumed_completed_frames': len(resumed),
              'timing_limitations': 'resumed legacy run lacks original failed-attempt/interruption timing; no complete end-to-end FPS claim' if resumed else None,
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
