"""B2 causal tracking with original U2 assignment/state, exact shared proposals."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from PIL import Image, ImageDraw
    from locatemot.groundjev.u2_bridge import GroundU2Bridge, candidate_tensors
    from locatemot.unified.runtime.u2_online_tracker import U2OnlineTracker
    from tools.groundjev.run_video import ReplayFrames
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--candidate-replay', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--resume-state', type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.video.read_text())
    assert manifest['task'] == 'mot'
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    assert checkpoint['format'] == 'groundjev-B2-original-U2-core-v1'
    model = GroundU2Bridge().eval()
    model.load_state_dict(checkpoint['state_dict'])
    tracker = U2OnlineTracker(model.core)
    source = ReplayFrames(args.candidate_replay)
    resumed = []
    if args.resume_from:
        assert args.resume_state is not None
        resumed = [json.loads(line) for line in (args.resume_from/'frame_decisions.jsonl').open()]
        assert [r['frame'] for r in resumed] == manifest['frame_numbers'][:len(resumed)]
        tracker.load_state(args.resume_state)
        assert tracker.state.frame_index == resumed[-1]['frame']
    writer = None
    started = time.perf_counter()
    rejected = []
    rows_done = 0
    resume_delta = None
    try:
        with (args.out/'predictions.txt').open('w') as predictions, (args.out/'frame_decisions.jsonl').open('w') as log:
            for path, frame in zip(manifest['images'], manifest['frame_numbers']):
                original = source.frame(frame)
                assert original['image_path'] == path
                candidates = original['candidates']
                if original['perception'].get('model_response_rejected'):
                    rejected.append(frame)
                features, boxes, scores = candidate_tensors(candidates)
                if rows_done < len(resumed):
                    prior = resumed[rows_done]
                    geometry = [{k: v for k, v in c.items() if k not in ('native_region_feature', 'temporal_region_feature')}
                                for c in candidates]
                    assert prior['candidates'] == geometry and prior['image_path'] == path
                    emitted = prior['outputs']
                    indices = []
                else:
                    with torch.no_grad():
                        inputs = model.candidates(features, boxes, scores)
                        output = tracker.step(inputs, task_type='mot', frame_index=frame)
                    emitted = []
                    indices = list(zip(output['candidate_indices'].tolist(), output['track_ids'].tolist()))
                if not resumed and rows_done == 1:
                    tracker.save_state(args.out/'functional_resume_state.pt')
                    restored = U2OnlineTracker(model.core)
                    restored.load_state(args.out/'functional_resume_state.pt')
                elif not resumed and rows_done == 2:
                    with torch.no_grad():
                        repeated = restored.step(inputs, task_type='mot', frame_index=frame)
                    resume_delta = max((float((output[k]-repeated[k]).abs().max())
                        for k in ('new_logits', 'pair_logits') if output[k].numel()), default=0.)
                    assert resume_delta == 0 and torch.equal(output['track_ids'], repeated['track_ids'])
                    for key, value in tracker.state_dict()['state'].items():
                        other = restored.state_dict()['state'][key]
                        if torch.is_tensor(value):
                            assert torch.equal(value, other)
                for index, track_id in indices:
                    box = candidates[index]['box']
                    emitted.append({'track_id': track_id, 'box': box, 'label': candidates[index]['label']})
                for item in emitted:
                    x, y, x2, y2 = item['box']
                    predictions.write(f'{frame},{item["track_id"]},{x:.3f},{y:.3f},{x2-x:.3f},{y2-y:.3f},1,-1,-1,-1\n')
                geometry = [{k: v for k, v in c.items() if k not in ('native_region_feature', 'temporal_region_feature')}
                            for c in candidates]
                record = prior if rows_done < len(resumed) else {'frame': frame, 'image_path': path, 'candidates': geometry,
                    'outputs': emitted, 'perception': original['perception'],
                    'state': {'persistent_ids': tracker.state.track_ids.tolist(),
                        'next_track_id': tracker.state.next_track_id, 'frame_index': tracker.state.frame_index,
                        'matched_pairs': output['matched_pairs'], 'birth_candidates': output['birth_candidates']},
                    'native_U2': {'pair_logits': output['pair_logits'].tolist(),
                        'new_logits': output['new_logits'].tolist(), 'no_match_logits': output['no_match_logits'].tolist(),
                        'alive_logits': output['alive_logits'].tolist(), 'GT_read': False}}
                log.write(json.dumps(record)+'\n')
                predictions.flush()
                log.flush()
                with Image.open(path) as raw:
                    canvas = raw.convert('RGB')
                draw = ImageDraw.Draw(canvas)
                for item in emitted:
                    draw.rectangle(item['box'], outline='lime', width=2)
                    draw.text(item['box'][:2], str(item['track_id']), fill='red')
                if writer is None:
                    width, height = canvas.size
                    writer = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
                        '-r', str(manifest['fps']), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast',
                        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p', str(args.out/'tracks.mp4')],
                        stdin=subprocess.PIPE)
                writer.stdin.write(canvas.tobytes())
                rows_done += 1
                if rows_done % 100 == 0 and rows_done >= len(resumed):
                    tracker.save_state(args.out/'latest_identity_state.pt')
                    print(json.dumps({'frames_completed': rows_done}), flush=True)
        writer.stdin.close()
        assert writer.wait() == 0
        assert rows_done == len(manifest['images'])
        tracker.save_state(args.out/'final_identity_state.pt')
        elapsed = time.perf_counter()-started
        result = {'status': 'VIDEO_INFERENCE_COMPLETE_WITH_REJECTED_PERCEPTION' if rejected else 'VIDEO_INFERENCE_COMPLETE',
            'backend': 'B2_original_U2_SharedTrackCore', 'checkpoint': str(args.checkpoint),
            'scope': manifest['scope'], 'task': 'mot', 'frames': rows_done, 'device': 'cpu',
            'all_input_frames_processed': True, 'GT_read': False, 'gt_read_by_runtime': False,
            'future_frames_read_by_runtime': False, 'perception_rejected_frames': rejected,
            'perception_rejection_rate': len(rejected)/rows_done,
            'association_replay_fps': None if resumed else rows_done/elapsed,
            'observed_resumed_scope_video_fps_not_standalone_speed': rows_done/elapsed if resumed else None,
            'end_to_end_fps': None, 'shared_input_features_source': str(args.candidate_replay),
            'all_native_candidates_and_geometry_preserved': True, 'native_feature_vectors_omitted_from_log': True,
            'original_U2_assignment_lifecycle_and_motion_unchanged': True, 'max_age_source_frames': 30,
            'GT_or_OneJev_decision_used_by_runtime': False, 'causal_memory_save_restore_next_frame_max_difference': resume_delta,
            'resumed_completed_frames': len(resumed), 'resumed_source': str(args.resume_from) if resumed else None,
            'geometry_numeric_repair': 'every defined FP32 center distance retained; only undefined inf/inf ratios recomputed in FP64; no box clipping or track removal',
            'timing_scope': 'CPU frame/video writes for every frame; restored prefix bypasses association; not uninterrupted or detector-inclusive speed'}
        (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
    except Exception as error:
        (args.out/'failure.json').write_text(json.dumps({'error': repr(error), 'frames_completed': rows_done})+'\n')
        raise
    finally:
        source.close()
        if writer is not None and writer.poll() is None:
            if not writer.stdin.closed:
                writer.stdin.close()
            writer.wait()


if __name__ == '__main__':
    main()
