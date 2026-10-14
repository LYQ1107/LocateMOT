"""Stream real video frames through DLM + Simple/Full persistent tracking."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class ReplayFrames:
    """Read only the next requested native frame, bounding replay memory."""
    def __init__(self, path):
        self.file = path.open()
    def frame(self, number):
        for line in self.file:
            row = json.loads(line)
            if row['frame'] < number: continue
            if row['frame'] != number: break
            return row
        raise ValueError(f'Native replay lacks requested frame {number}; frame cannot be skipped')
    def close(self): self.file.close()


def restore_completed_prefix(tracker, rows, frames, fps, specification, timestamps=None):
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
            timestamp = timestamps[number] if timestamps is not None else number/fps
            outputs, state = tracker.step(image, row['candidates'], timestamp, specification)
            if outputs != row['outputs'] or state != row['state']:
                raise ValueError('Native decision replay did not restore identical causal outputs/state')
    finally:
        tracker.decision = original


def main():
    invocation_start = time.perf_counter()
    from PIL import Image, ImageDraw
    from locatemot.groundjev.perception import GroundAnythingAdapter, GroundingProtocolError
    from locatemot.groundjev.decision import OneJevDecision, SimpleDecision, FeatureDecision
    from locatemot.groundjev.tracker import OnlineTracker
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['mot', 'rmot', 'ovmot'], default='mot')
    parser.add_argument('--video', type=Path, required=True, help='legal JSON video scope manifest')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'models/OneJev-0.8B')
    parser.add_argument('--backend', choices=['simple', 'simple_native', 'simple_temporal', 'full', 'compact'], default='full')
    parser.add_argument('--evidence-format',choices=['v1','v2_crop_motion','v3_crop_motion','v4_native_features','v5_native_features_csv','v6_native_compact_history'],default='v1')
    parser.add_argument('--native-feature-url',help='isolated official DLM visual encoder RPC; no substitute image features')
    parser.add_argument('--adaptive-context-groups',action='store_true',help='split native preflight-rejected groups, retaining all candidate rows and identity options')
    parser.add_argument('--category-policy', choices=['native_label', 'native_hierarchical'], default='native_label')
    parser.add_argument('--native-rmot-membership', action='store_true', help='Full: explicit native OneJev Noul expression decision, with fixed .5 output threshold')
    parser.add_argument('--expression')
    parser.add_argument('--vocabulary', nargs='+')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, help='explicit short functional check; never a full-video metric')
    parser.add_argument('--grounding-url', default='http://127.0.0.1:8101/v1')
    parser.add_argument('--grounding-parser', choices=['strict_client', 'official_scoring'], default='strict_client',
                        help='explicit candidate protocol; official scorer preserves strict wire diagnostics')
    parser.add_argument('--grounding-decoder-policy', choices=['generic_greedy', 'tracking_dense'], default='generic_greedy')
    parser.add_argument('--grounding-vocabulary-chunk-size', type=int,
                        help='OVMOT: exhaustively request this many vocabulary classes at a time, then associate the union once')
    parser.add_argument('--perception-failure-policy', choices=['stop', 'empty_and_record'], default='stop',
                        help='official scorer rejection is an explicitly logged empty prediction, included in every metric denominator')
    parser.add_argument('--candidate-replay', type=Path, help='exact prior frame log for a fixed-candidate association control; no end-to-end FPS claim')
    parser.add_argument('--resume-from', type=Path, help='preserve source run and restore its completed causal prefix in this fresh output')
    parser.add_argument('--memory-policy',choices=['gated','always','ema','frame_only'],default='gated')
    parser.add_argument('--memory-feature-view',choices=['all','recent','ema'],default='all')
    parser.add_argument('--lost-seconds',type=float,default=3.)
    parser.add_argument('--ema-alpha',type=float,default=.1)
    parser.add_argument('--omit-replay-feature-vectors',action='store_true',help='retain shared source reference and all boxes/decisions; avoid duplicating input feature arrays')
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('choose a fresh output directory to preserve existing experiment results')
    manifest = json.loads(args.video.read_text())
    if manifest['task'] != args.task:
        parser.error('requested task differs from the registered video scope')
    if args.native_rmot_membership and (args.backend != 'full' or args.task != 'rmot'):
        parser.error('native RMOT membership requires the Full OneJev backend and RMOT task')
    vocabulary = args.vocabulary or manifest['vocabulary']
    args.expression = args.expression or manifest.get('expression')
    if args.task == 'rmot' and not args.expression:
        parser.error('RMOT requires --expression')
    if args.backend=='compact' and not args.native_feature_url and not args.candidate_replay:
        parser.error('Compact requires native feature RPC or exact replay already containing native features')
    if args.backend=='compact' and args.category_policy!='native_label':
        parser.error('Compact category training is pending; the native category policy currently uses OneJev checkpoints')
    if args.memory_feature_view!='all' and args.backend!='compact':
        parser.error('feature-view ablation requires the native-feature Compact model')
    if args.memory_policy=='ema' and (args.backend!='compact' or args.memory_feature_view!='ema'):
        parser.error('EMA baseline uses causal native feature EMA with Compact; no synthetic image EMA')
    if args.omit_replay_feature_vectors and not args.candidate_replay:
        parser.error('feature omission requires the retained exact shared input source')
    args.out.mkdir(parents=True, exist_ok=True)
    frontend = GroundAnythingAdapter(args.grounding_url, parser_policy=args.grounding_parser,
        decoder_policy=args.grounding_decoder_policy, vocabulary_chunk_size=args.grounding_vocabulary_chunk_size)
    if args.backend=='simple': decision=SimpleDecision()
    elif args.backend in ('simple_native','simple_temporal'): decision=FeatureDecision(temporal=args.backend=='simple_temporal')
    elif args.backend=='compact':
        from locatemot.groundjev.compact import CompactDecision
        decision=CompactDecision(args.checkpoint,memory_feature_view=args.memory_feature_view)
    else: decision=OneJevDecision(args.checkpoint,evidence_format=args.evidence_format,adaptive_context=args.adaptive_context_groups,
                                rmot_membership=args.native_rmot_membership)
    if args.category_policy == 'native_hierarchical':
        from locatemot.groundjev.category import NativeCategoryHead, CategorizedDecision
        engine = decision.engine if args.backend == 'full' else OneJevDecision(args.checkpoint).engine
        decision = CategorizedDecision(decision, NativeCategoryHead(engine))
    tracker = OnlineTracker(decision,lost_seconds=args.lost_seconds,memory_policy=args.memory_policy,ema_alpha=args.ema_alpha)
    feature_client=None
    if args.native_feature_url:
        from locatemot.groundjev.feature_client import NativeFeatureClient
        feature_client=NativeFeatureClient(args.native_feature_url)
    frames = list(zip(manifest['images'], manifest['frame_numbers']))
    if args.limit:
        frames = frames[:args.limit]
    replay = {}
    if args.candidate_replay:
        replay = ReplayFrames(args.candidate_replay)
    fps, latencies, rejected_frames, strict_invalid_frames = manifest['fps'], [], [], []
    timestamps = dict(zip(manifest['frame_numbers'], manifest['timestamps'])) if 'timestamps' in manifest else None
    resumed = []
    specification = {'task': args.task, 'vocabulary': vocabulary, 'expression': args.expression}
    if args.resume_from:
        resumed = list(map(json.loads, (args.resume_from/'frame_decisions.jsonl').read_text().splitlines()))
        prior_result_path=args.resume_from/'result.json'
        feature_source_path=None
        if prior_result_path.exists():
            prior_result=json.loads(prior_result_path.read_text())
            if prior_result.get('replay_feature_vectors_omitted_from_log'):
                feature_source_path=Path(prior_result['shared_input_features_source'])
        elif args.omit_replay_feature_vectors:
            feature_source_path=args.candidate_replay
        if feature_source_path:
            feature_source=ReplayFrames(feature_source_path)
            for row in resumed:
                original_features=feature_source.frame(row['frame'])['candidates']
                geometry=[{k:v for k,v in c.items() if k not in ('native_region_feature','temporal_region_feature')} for c in original_features]
                assert row['candidates']==geometry
                row['candidates']=original_features
            feature_source.close()
        if len(resumed) > len(frames):
            raise ValueError('Resume prefix is longer than the requested video')
        restore_completed_prefix(tracker, resumed, frames, fps, specification, timestamps)
        (args.out/'resume_memory.json').write_text(json.dumps(tracker.state_dict())+'\n')
        print(json.dumps({'restored_completed_frames': len(resumed), 'source': str(args.resume_from),
                          'identical_native_outputs_and_state': True, 'gt_used': False}), flush=True)
    video_writer = None
    video_start = time.perf_counter()
    initialization_seconds = video_start-invocation_start
    try:
        with (args.out / 'predictions.txt').open('w') as predictions, (args.out / 'frame_decisions.jsonl').open('w') as log:
            for index, (path, frame_number) in enumerate(frames):
                start = time.perf_counter()
                with Image.open(path) as source:
                    image = source.convert('RGB')
                original = replay.frame(frame_number) if replay else None
                if original is not None and original['image_path'] != path:
                    raise ValueError('Fixed-candidate replay uses a different physical image')
                if index < len(resumed):
                    row = resumed[index]
                    candidates, perception, outputs, state = (row[k] for k in ('candidates','perception','outputs','state'))
                    elapsed = row['seconds']
                elif replay:
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
                diagnostic = perception
                while 'original_perception' in diagnostic:
                    diagnostic = diagnostic['original_perception']
                if diagnostic.get('strict_wire_valid') is False:
                    strict_invalid_frames.append(frame_number)
                if index >= len(resumed):
                    if feature_client:
                        candidates,feature_metadata=feature_client(image,candidates,image_path=path)
                        perception['native_feature_extraction']=feature_metadata
                    print(json.dumps({'frame': frame_number, 'stage': 'association_and_memory', 'candidates': len(candidates)}), flush=True)
                    timestamp = timestamps[frame_number] if timestamps is not None else frame_number/fps
                    outputs, state = tracker.step(image, candidates, timestamp, specification)
                    elapsed = time.perf_counter()-start
                latencies.append(elapsed)
                for row in outputs:
                    x1,y1,x2,y2 = row['box']
                    # MOTChallenge requires a confidence column; 1 here is an
                    # explicit unscored-output sentinel, not detector confidence.
                    predictions.write(f'{frame_number},{row["track_id"]},{x1:.3f},{y1:.3f},{x2-x1:.3f},{y2-y1:.3f},1,-1,-1,-1\n')
                recorded_candidates = candidates
                if args.omit_replay_feature_vectors:
                    recorded_candidates = [{k:v for k,v in c.items() if k not in ('native_region_feature','temporal_region_feature')} for c in candidates]
                log.write(json.dumps({'frame': frame_number, 'image_path': path, 'seconds': elapsed, 'candidates': recorded_candidates,
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
                        '-framerate', str(fps), '-i', '-', '-an', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
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
        if replay: replay.close()
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
    complete_video_wall_seconds = time.perf_counter()-video_start
    result = {'status': 'VIDEO_INFERENCE_COMPLETE_WITH_REJECTED_PERCEPTION' if rejected_frames else 'VIDEO_INFERENCE_COMPLETE',
              'task': args.task, 'backend': args.backend,
              'evidence_format':args.evidence_format,
              'native_feature_url':args.native_feature_url,
              'adaptive_context_groups':args.adaptive_context_groups,
              'category_policy':args.category_policy,
              'scope': 'short_functional_check' if args.limit else manifest['scope'],
              'video_manifest': str(args.video), 'frames': len(frames), 'latencies_seconds': latencies,
              'grounding_parser_policy': args.grounding_parser,
              'grounding_decoder_policy': args.grounding_decoder_policy,
              'grounding_vocabulary_chunk_size':args.grounding_vocabulary_chunk_size,
              'specification':specification,
              'timestamp_policy':manifest.get('timestamp_policy','physical frame number / registered video FPS'),
              'perception_failure_policy': args.perception_failure_policy,
              'perception_rejected_frames': rejected_frames,
              'perception_rejection_rate': len(rejected_frames)/len(frames),
              'strict_wire_invalid_frames': strict_invalid_frames,
              'all_input_frames_processed': True,
              'end_to_end_fps': None if replay or resumed else len(frames)/complete_video_wall_seconds,
              'complete_video_wall_seconds':complete_video_wall_seconds,
              'model_initialization_seconds':initialization_seconds,
              'timing_scope':'all image reads, native requests including service queueing, association, memory persistence, prediction/log writes, drawing, video encoding and encoder completion; preloaded remote DLM server startup excluded',
              'successful_frame_pipeline_fps': len(frames)/sum(latencies),
              'association_replay_fps': len(frames)/sum(latencies) if replay else None,
              'resume_source': str(args.resume_from) if args.resume_from else None,
              'resumed_completed_frames': len(resumed),
              'timing_limitations': 'resumed legacy run lacks original failed-attempt/interruption timing; no complete end-to-end FPS claim' if resumed else None,
              'official_metrics': 'PENDING_OFFICIAL_EVALUATOR',
              'checkpoint': str(args.checkpoint) if args.backend in ('full','compact') or args.category_policy == 'native_hierarchical' else None,
              'motchallenge_confidence_column': 'constant unscored-output sentinel; native detection score is absent',
              'gt_read_by_runtime': False, 'future_frames_read_by_runtime': False,
              'memory_policy':args.memory_policy,'memory_feature_view':args.memory_feature_view,
              'lost_seconds':args.lost_seconds,'ema_alpha':args.ema_alpha,
              'replay_feature_vectors_omitted_from_log':args.omit_replay_feature_vectors,
              'shared_input_features_source':str(args.candidate_replay) if args.omit_replay_feature_vectors else None}
    if args.backend in ('full','compact') or args.category_policy == 'native_hierarchical':
        import torch
        result['peak_decision_gpu_memory_bytes'] = torch.cuda.max_memory_allocated()
    (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    (args.out/'final_identity_memory.json').write_text(json.dumps(tracker.state_dict())+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
