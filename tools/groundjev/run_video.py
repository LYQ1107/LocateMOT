"""Stream real video frames through DLM + Simple/Full persistent tracking."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import resource
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class ReplayFrames:
    """Read only the next requested native frame, bounding replay memory."""
    def __init__(self, path):
        self.directory = path.parent
        self.file = path.open()
    def frame(self, number):
        for line in self.file:
            row = json.loads(line)
            if row['frame'] < number: continue
            if row['frame'] != number: break
            from locatemot.groundjev.candidate_archive import decode_candidate_row
            return decode_candidate_row(row, self.directory)
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
    invocation_utc = datetime.now(timezone.utc).isoformat()
    invocation_git_commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    from PIL import Image, ImageDraw
    from locatemot.groundjev.perception import GroundAnythingAdapter, GroundingProtocolError
    from locatemot.groundjev.decision import OneJevDecision, SimpleDecision, FeatureDecision
    from locatemot.groundjev.tracker import OnlineTracker
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['mot', 'rmot', 'ovmot'], default='mot')
    parser.add_argument('--video', type=Path, required=True, help='legal JSON video scope manifest')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'models/OneJev-0.8B')
    parser.add_argument('--trackjev-arm', choices=['B2','B3','B4','B6'], default='B4',
                        help='registered ablation arm for the TrackJEV backend')
    parser.add_argument('--trackjev-shortlist', type=int, default=31)
    parser.add_argument('--trackjev-duplicate-penalty', type=float, default=0.0)
    parser.add_argument('--trackjev-device', default='cuda:0')
    parser.add_argument('--backend', choices=['simple', 'simple_native', 'simple_temporal', 'full', 'compact', 'trackjev'], default='full')
    parser.add_argument('--compact-device',choices=['cpu','cuda:0'],default='cuda:0',help='explicit device for trained Compact replay; report actual device, no detector E2E claim')
    parser.add_argument('--global-identity',action='store_true',help='isolated Global Identity Compact checkpoint')
    parser.add_argument('--global-memory-policy',choices=['original','always','confidence','assigned'],default='original')
    parser.add_argument('--compact-shortlist',type=int,choices=[32,64,128],default=32,
                        help='registered context-size intervention; larger values exceed the original 32-ID training context')
    parser.add_argument('--compact-shortlist-policy',choices=['geometry','hybrid16_16'],default='geometry')
    parser.add_argument('--compact-candidate-grouping',choices=['contiguous','overlap_neighbors'],default='contiguous')
    parser.add_argument('--compact-association-policy',choices=['learned','fixed_iou','fixed_iou_cosine','rank_appearance'],default='learned')
    parser.add_argument('--compact-auxiliary-policy',choices=['learned','fixed'],default='learned')
    parser.add_argument('--evidence-format',choices=['v1','v2_crop_motion','v3_crop_motion','v4_native_features','v5_native_features_csv','v6_native_compact_history'],default='v1')
    parser.add_argument('--native-feature-url',help='isolated official DLM visual encoder RPC; no substitute image features')
    parser.add_argument('--adaptive-context-groups',action='store_true',help='split native preflight-rejected groups, retaining all candidate rows and identity options')
    parser.add_argument('--category-policy', choices=['native_label', 'native_hierarchical'], default='native_label')
    parser.add_argument('--category-replay', type=Path,
                        help='explicit same-checkpoint native categories from a complete run on identical frames/candidates')
    parser.add_argument('--native-rmot-membership', action='store_true', help='Full: explicit native OneJev Noul expression decision, with fixed .5 output threshold')
    parser.add_argument('--assigned-identity-updates', action='store_true',
                        help='experimental Full v3 second native Noul pass after global matching and before memory write; needs matching question training')
    parser.add_argument('--expression')
    parser.add_argument('--vocabulary', nargs='+')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, help='explicit short functional check; never a full-video metric')
    parser.add_argument('--grounding-url', default='http://127.0.0.1:8101/v1')
    parser.add_argument('--grounding-parser', choices=['strict_client', 'official_scoring', 'official_scoring_repaired'], default='strict_client',
                        help='explicit candidate protocol; repaired mode deduplicates exact duplicate wire tuples while retaining strict diagnostics')
    parser.add_argument('--grounding-decoder-policy', choices=['generic_greedy', 'tracking_dense'], default='generic_greedy')
    parser.add_argument('--grounding-vocabulary-chunk-size', type=int,
                        help='OVMOT: exhaustively request this many vocabulary classes at a time, then associate the union once')
    parser.add_argument('--perception-failure-policy', choices=['stop', 'empty_and_record'], default='stop',
                        help='official scorer rejection is an explicitly logged empty prediction, included in every metric denominator')
    parser.add_argument('--candidate-replay', type=Path, help='exact prior frame log for a fixed-candidate association control; no end-to-end FPS claim')
    parser.add_argument('--resume-from', type=Path, help='preserve source run and restore its completed causal prefix in this fresh output')
    parser.add_argument('--memory-policy',choices=['gated','always','ema','frame_only'],default='gated')
    parser.add_argument('--memory-feature-view',choices=['all','recent','ema'],default='all')
    parser.add_argument('--lifecycle-termination-policy',choices=['learned_or_timeout','timeout_only'],default='learned_or_timeout')
    parser.add_argument('--no-match-policy',choices=['learned','only_if_omitted'],default='learned')
    parser.add_argument('--appearance-projection-checkpoint',type=Path)
    parser.add_argument('--appearance-ranking-temperature',type=float,default=.07)
    parser.add_argument('--lost-seconds',type=float,default=3.)
    parser.add_argument('--ema-alpha',type=float,default=.1)
    parser.add_argument('--omit-replay-feature-vectors',action='store_true',help='retain shared source reference and all boxes/decisions; avoid duplicating input feature arrays')
    parser.add_argument('--record-native-candidates', action='store_true', help='archive actual current native candidates once for subsequent matched controls')
    parser.add_argument('--native-candidate-format',choices=['jsonl','packed_npz_f32'],default='jsonl',
                        help='packed stores exact native FP32 ROI vectors in per-frame compressed files; no quantization')
    parser.add_argument('--native-candidate-byte-budget', type=int, default=2*2**30)
    parser.add_argument('--memory-checkpoint-format',choices=['json','binary'],default='json')
    parser.add_argument('--memory-export-interval',type=int,default=1,
                        help='0 retains per-frame decisions for causal replay and writes only final full memory; default exports every frame')
    args = parser.parse_args()
    if args.appearance_projection_checkpoint and (args.backend!='compact' or args.compact_association_policy not in ('fixed_iou_cosine','rank_appearance')):
        parser.error('appearance projection requires an explicit Compact appearance control')
    if args.compact_association_policy=='rank_appearance' and args.compact_auxiliary_policy!='learned':
        parser.error('registered appearance ranks retain learned auxiliary heads')
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('choose a fresh output directory to preserve existing experiment results')
    manifest = json.loads(args.video.read_text())
    if manifest['task'] != args.task:
        parser.error('requested task differs from the registered video scope')
    if args.native_rmot_membership and (args.backend != 'full' or args.task != 'rmot'):
        parser.error('native RMOT membership requires the Full OneJev backend and RMOT task')
    if args.assigned_identity_updates and (args.backend != 'full' or args.evidence_format != 'v3_crop_motion' or args.memory_policy != 'gated'):
        parser.error('assigned-identity updates require Full v3_crop_motion and gated memory')
    if args.category_replay and (args.backend != 'full' or args.category_policy != 'native_hierarchical' or not args.candidate_replay):
        parser.error('category replay requires Full native_hierarchical and fixed candidate replay')
    if args.backend!='compact' and (args.compact_association_policy!='learned' or args.compact_auxiliary_policy!='learned' or args.compact_shortlist!=32 or args.compact_shortlist_policy!='geometry'):
        parser.error('Compact policy controls require the Compact backend')
    if args.compact_shortlist_policy=='hybrid16_16' and args.compact_shortlist!=32:
        parser.error('registered hybrid protocol has exactly32 options:16geometry then appearance fill')
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
    if args.omit_replay_feature_vectors and not (args.candidate_replay or args.record_native_candidates):
        parser.error('feature omission requires the retained exact shared input source or current native archive')
    args.out.mkdir(parents=True, exist_ok=True)
    frontend = GroundAnythingAdapter(args.grounding_url, parser_policy=args.grounding_parser,
        decoder_policy=args.grounding_decoder_policy, vocabulary_chunk_size=args.grounding_vocabulary_chunk_size)
    if args.backend=='simple': decision=SimpleDecision()
    elif args.backend in ('simple_native','simple_temporal'): decision=FeatureDecision(temporal=args.backend=='simple_temporal')
    elif args.backend=='compact':
        if args.compact_device=='cpu':
            import torch
            torch.set_num_threads(4)
        from locatemot.groundjev.compact import CompactDecision
        if args.global_identity:
            assert args.compact_device=='cpu' and args.compact_association_policy=='learned' and args.compact_auxiliary_policy=='learned'
            from locatemot.groundjev.global_identity_runtime import GlobalIdentityDecision
            decision=GlobalIdentityDecision(args.checkpoint,args.global_memory_policy)
        else:
            decision=CompactDecision(args.checkpoint,device=args.compact_device,shortlist=args.compact_shortlist,memory_feature_view=args.memory_feature_view,shortlist_policy=args.compact_shortlist_policy,candidate_grouping=args.compact_candidate_grouping)
        if args.compact_association_policy=='rank_appearance':
            from locatemot.groundjev.appearance_ranking import AppearanceRankDecision
            decision=AppearanceRankDecision(decision,args.appearance_projection_checkpoint,args.appearance_ranking_temperature)
        elif args.compact_association_policy!='learned' or args.compact_auxiliary_policy!='learned':
            from locatemot.groundjev.compact import CompactAffinityControl
            if not decision.model.semantic:
                parser.error('matched semantic affinity control requires the trained shared semantic checkpoint')
            decision=CompactAffinityControl(decision,args.compact_association_policy,args.compact_auxiliary_policy,
                appearance_projection_checkpoint=args.appearance_projection_checkpoint)
    elif args.backend=='trackjev':
        from locatemot.groundjev.trackjev.native_onejev_adapter import NativeOneJevAdapter
        from locatemot.groundjev.trackjev.tracker_decision import (TrackJEVTrackerDecision,
                                                                  TrackerDecisionConfig)
        arm = args.trackjev_arm
        adapter = NativeOneJevAdapter(args.checkpoint, device=args.trackjev_device,
                                      shortlist=args.trackjev_shortlist,
                                      evidence_format='trackjev_v1',
                                      ask_risk=(arm == 'B6'))
        config = TrackerDecisionConfig(shortlist=args.trackjev_shortlist,
                                       use_competition=arm in ('B3', 'B4', 'B6'),
                                       assignment_conditioned_memory=arm in ('B4', 'B6'),
                                       ask_risk=(arm == 'B6'),
                                       duplicate_penalty=args.trackjev_duplicate_penalty)
        decision = TrackJEVTrackerDecision(adapter, config)
    else:
        decision_type = OneJevDecision
        if args.assigned_identity_updates:
            from locatemot.groundjev.assigned_identity_decision import AssignedIdentityUpdateDecision
            decision_type = AssignedIdentityUpdateDecision
        decision=decision_type(args.checkpoint,evidence_format=args.evidence_format,adaptive_context=args.adaptive_context_groups,
                               rmot_membership=args.native_rmot_membership)
    category_replay = None
    if args.category_policy == 'native_hierarchical':
        from locatemot.groundjev.category import NativeCategoryHead, CategorizedDecision
        engine = decision.engine if args.backend == 'full' else OneJevDecision(args.checkpoint).engine
        head = NativeCategoryHead(engine)
        if args.category_replay:
            from locatemot.groundjev.category_replay import CausalNativeCategoryReplay
            head = category_replay = CausalNativeCategoryReplay(args.category_replay, args.checkpoint, manifest)
        decision = CategorizedDecision(decision, head)
    tracker_type=OnlineTracker
    if args.global_identity:
        from locatemot.groundjev.global_identity_runtime import GlobalIdentityTracker
        tracker_type=GlobalIdentityTracker
    tracker = tracker_type(decision,lost_seconds=args.lost_seconds,memory_policy=args.memory_policy,ema_alpha=args.ema_alpha,
                            lifecycle_termination_policy=args.lifecycle_termination_policy,no_match_policy=args.no_match_policy)
    def write_memory(stem):
        snapshot=tracker.state_dict()
        suffix='json' if args.memory_checkpoint_format=='json' else 'pt'
        target=args.out/(stem+'.'+suffix)
        temporary=target.with_suffix('.tmp')
        if args.memory_checkpoint_format=='json':
            temporary.write_text(json.dumps(snapshot)+'\n')
        else:
            import torch
            torch.save(snapshot,temporary)
        temporary.replace(target)
        return str(target)
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
            if bool(prior_result.get('assigned_identity_updates', False)) != args.assigned_identity_updates:
                raise ValueError('Resume source uses a different observation/final-ID memory update protocol')
            if prior_result.get('lifecycle_termination_policy','learned_or_timeout')!=args.lifecycle_termination_policy:
                raise ValueError('Resume source uses a different lifecycle termination policy')
            if prior_result.get('no_match_policy','learned')!=args.no_match_policy:
                raise ValueError('Resume source uses a different NO_MATCH execution contract')
            if prior_result.get('compact_candidate_grouping','contiguous')!=args.compact_candidate_grouping:
                raise ValueError('Resume source uses a different candidate grouping policy')
            if prior_result.get('replay_feature_vectors_omitted_from_log'):
                feature_source_path=Path(prior_result['shared_input_features_source'])
        elif args.omit_replay_feature_vectors:
            feature_source_path=(args.resume_from/'frame_candidates.jsonl') if args.record_native_candidates else args.candidate_replay
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
        write_memory('resume_memory')
        print(json.dumps({'restored_completed_frames': len(resumed), 'source': str(args.resume_from),
                          'identical_native_outputs_and_state': True, 'gt_used': False}), flush=True)
    video_writer = None
    native_archive_path = args.out/'frame_candidates.jsonl'
    native_archive = None
    video_start = time.perf_counter()
    initialization_seconds = video_start-invocation_start
    if args.record_native_candidates:
        from locatemot.groundjev.candidate_archive import CandidateArchiveWriter
        native_archive = CandidateArchiveWriter(native_archive_path, packed=args.native_candidate_format=='packed_npz_f32')
    try:
        with (args.out / 'predictions.txt').open('w') as predictions, (args.out / 'frame_decisions.jsonl').open('w') as log, \
                (native_archive if native_archive else nullcontext(None)) as native_log:
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
                if native_log is not None:
                    native_log.write({'frame': frame_number, 'image_path': path,
                        'candidates': candidates, 'perception': perception})
                    native_log.flush()
                    if native_log.tell() > args.native_candidate_byte_budget:
                        raise RuntimeError('Registered per-study native feature budget exceeded; preserve source and do not skip candidates')
                if index >= len(resumed):
                    print(json.dumps({'frame': frame_number, 'stage': 'association_and_memory', 'candidates': len(candidates)}), flush=True)
                    timestamp = timestamps[frame_number] if timestamps is not None else frame_number/fps
                    if category_replay is not None:
                        category_replay.begin_frame(path, frame_number)
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
                if index >= len(resumed) and args.memory_export_interval and (index+1)%args.memory_export_interval==0:
                    write_memory('latest_identity_memory')
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
        if category_replay is not None: category_replay.close()
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
    final_memory_path=write_memory('final_identity_memory')
    complete_video_wall_seconds = time.perf_counter()-video_start
    result = {'status': 'VIDEO_INFERENCE_COMPLETE_WITH_REJECTED_PERCEPTION' if rejected_frames else 'VIDEO_INFERENCE_COMPLETE',
              'invocation_utc':invocation_utc,'invocation_git_commit':invocation_git_commit,
              'task': args.task, 'backend': args.backend,
              'evidence_format':args.evidence_format,
              'native_feature_url':args.native_feature_url,
              'compact_association_policy':args.compact_association_policy,'compact_auxiliary_policy':args.compact_auxiliary_policy,
              'compact_shortlist_limit':args.compact_shortlist if args.backend=='compact' else None,
              'compact_shortlist_policy':args.compact_shortlist_policy if args.backend=='compact' else None,
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
              'association_replay_fps': len(frames)/sum(latencies) if replay and not resumed else None,
              'resume_source': str(args.resume_from) if args.resume_from else None,
              'resumed_completed_frames': len(resumed),
              'timing_limitations': 'resumed legacy run lacks original failed-attempt/interruption timing; no complete end-to-end FPS claim' if resumed else None,
              'official_metrics': 'PENDING_OFFICIAL_EVALUATOR',
              'checkpoint': str(args.checkpoint) if args.backend in ('full','compact') or args.category_policy == 'native_hierarchical' else None,
              'motchallenge_confidence_column': 'constant unscored-output sentinel; native detection score is absent',
              'gt_read_by_runtime': False, 'future_frames_read_by_runtime': False,
              'memory_policy':args.memory_policy,'memory_feature_view':args.memory_feature_view,
              'lost_seconds':args.lost_seconds,'ema_alpha':args.ema_alpha,
              'lifecycle_termination_policy':args.lifecycle_termination_policy,
              'no_match_policy':args.no_match_policy,
              'compact_candidate_grouping':args.compact_candidate_grouping,
              'replay_feature_vectors_omitted_from_log':args.omit_replay_feature_vectors,
              'shared_input_features_source':str(native_archive_path) if args.record_native_candidates else
                  str(args.candidate_replay) if args.omit_replay_feature_vectors else None,
              'native_candidate_archive':str(native_archive_path) if args.record_native_candidates else None,
              'native_candidate_archive_bytes':native_archive.tell() if native_archive else 0,
              'native_candidate_archive_format':args.native_candidate_format if native_archive else None,
              'native_candidate_archive_feature_files':native_archive.feature_files if native_archive else 0,
              'native_candidate_archive_budget_bytes':args.native_candidate_byte_budget if args.record_native_candidates else None}
    result.update(memory_checkpoint_format=args.memory_checkpoint_format,memory_export_interval=args.memory_export_interval,
                  causal_resume_source='complete frame_decisions plus retained exact native input source',
                  final_identity_memory_path=final_memory_path,final_full_memory_export_in_pipeline_timer=True)
    if args.assigned_identity_updates:
        from locatemot.groundjev.assigned_identity_update import PROTOCOL
        result.update(assigned_identity_updates=True, assigned_identity_update_protocol=PROTOCOL,
                      assigned_identity_update_training_and_accuracy='requires separately validated same-question SFT and causal evaluation')
    if category_replay is not None:
        result.update(category_replay=category_replay.summary(),
            timing_limitations='category probabilities reused from identical current images/candidates/checkpoint; replay timing is not comparable to fresh native classification')
    result['peak_decision_process_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    if args.global_identity:
        result.update(global_identity=True,global_memory_policy=args.global_memory_policy,
            lifecycle_termination_policy='timeout_only',same_physical_checkpoint_for_all_tasks=True,
            frozen_semantic_checkpoint=decision.semantic.checkpoint,
            frozen_semantic_parameters=sum(p.numel() for p in decision.semantic.model.parameters()),
            physical_cost_complexity='O(NM) association, O(NK) sparse competition; no NxN candidate attention bank')
    if args.backend=='compact':
        result['trainable_compact_core_parameters'] = sum(p.numel() for p in decision.model.parameters())
        if decision.model.semantic:
            table=decision.text_features.embedding
            result['frozen_CPU_text_table']={'shape':list(table.shape),'dtype':str(table.dtype),
                'parameters':table.numel(),'nominal_tensor_bytes':table.numel()*table.element_size(),
                'device':str(table.device),'scope':'frozen originalOneJev lookup table, not decoder inference; nominal bytes are not resident RSS'}
    if args.backend in ('full','compact') or args.category_policy == 'native_hierarchical':
        import torch
        cpu_compact=args.backend=='compact' and args.compact_device=='cpu'
        result['decision_device']=args.compact_device if args.backend=='compact' else 'cuda:0'
        result['peak_decision_gpu_memory_bytes'] = 0 if cpu_compact else torch.cuda.max_memory_allocated()
    (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
