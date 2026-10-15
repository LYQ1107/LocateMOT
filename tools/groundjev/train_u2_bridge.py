"""Train B2 on the existing48 legal native MOT frames; no detector/cache restart."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def prepare_frames():
    import torch
    from locatemot.unified.data.u2_mot_clips import OriginalMOTSupervision
    native = ROOT/'outputs/groundjev/compact/native_pack_v1'
    receipt = json.loads((native/'owner_state.json').read_text())
    assert receipt['status'] == 'REAL_NATIVE_COMPACT_TRAIN_PACK_COMPLETE'
    examples = torch.load(native/'examples.pt', map_location='cpu', weights_only=True)
    groups = defaultdict(list)
    for row in examples:
        if not row['derived_withholding']:
            groups[(row['source_sequence'], row['frame'])].append(row)
    protocol_root = Path('/data2/user/LocateMOT/outputs/unified/u2/protocol')
    sequences = {}
    for name in ('mot17_half', 'mot20_half', 'dancetrack'):
        protocol = json.loads((protocol_root/(name+'.json')).read_text())
        for seq in protocol['sequences']:
            sequences[seq['sequence']] = (protocol['dataset'], seq)
    supervision, by_sequence = {}, defaultdict(list)
    source = ROOT/'outputs/groundjev/training_data/native_ground_own_history_v1/native_rollout.jsonl'
    for line in source.open():
        row = json.loads(line)
        name, frame = row['sequence'], row['frame']
        dataset, seq = sequences[name]
        allowed = {int(Path(p).stem) for p in seq['train_frames']}
        assert frame in allowed
        assert Path(row['image_path']) == Path(seq['image_dir'])/next(
            p for p in seq['train_frames'] if int(Path(p).stem) == frame)
        if name not in supervision:
            group = {'dataset': dataset, 'video': name, 'key': dataset+'/train/'+name,
                     'frames': seq['train_frames']}
            gt = Path(seq['gt_path'])
            supervision[name] = OriginalMOTSupervision(group, {'source_path': str(gt),
                'sha256': hashlib.sha256(gt.read_bytes()).hexdigest()})
        parts = sorted(groups.get((name, frame), []), key=lambda e: int(e['id'].rsplit('batch', 1)[1]))
        features = torch.cat([e['candidates'] for e in parts]) if parts else torch.empty(0, 2560)
        boxes = [b for e in parts for b in e['candidate_boxes']]
        assert boxes == [c['box'] for c in row['candidates']]
        assert len(features) == len(row['candidates'])
        scores = torch.tensor([c['native_score'] if c.get('native_score') is not None else 1.
                               for c in row['candidates']], dtype=torch.float32)
        by_sequence[name].append({'frame_index': frame, 'features': features,
            'boxes': torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4), 'scores': scores,
            'supervision': supervision[name].frames[frame]})
    assert sum(map(len, by_sequence.values())) == 48
    clips = []
    for rows in by_sequence.values():
        rows.sort(key=lambda r: r['frame_index'])
        clips.extend([a, b] for a, b in zip(rows, rows[1:]))
    return clips, {'frames': 48, 'sequences': {k: len(v) for k, v in by_sequence.items()},
                   'two_frame_clips': len(clips), 'new_feature_cache_bytes': 0,
                   'source': str(native), 'validation_or_Novel_GT_used': False}


def main():
    import torch
    from locatemot.groundjev.u2_bridge import GroundU2Bridge
    from locatemot.unified.runtime.u2_temporal_training import U2MOTTemporalTraining
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume-state', type=Path)
    parser.add_argument('--steps', type=int, default=300)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    rng = random.Random(0)
    clips, scope = prepare_frames()
    model = GroundU2Bridge().train()
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    temporal = U2MOTTemporalTraining(model.core).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    plan = []
    while len(plan) < args.steps:
        order = list(range(len(clips)))
        rng.shuffle(order)
        plan.extend(order)
    plan = plan[:args.steps]
    start_step = 0
    if args.resume_state:
        saved = torch.load(args.resume_state, map_location='cpu', weights_only=False)
        assert saved['plan'] == plan and saved['scope'] == scope
        model.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'])
        start_step = saved['step']
    state = {'status': 'TRAINING_GROUND_U2_CORE_B2', 'owner_pid': __import__('os').getpid(),
        'device': 'cpu', 'steps_registered': args.steps, 'seed': 0, 'scope': scope,
        'core_architecture_and_native_U2_losses_unchanged': True, 'hidden_dim': 256, 'max_age': 30,
        'adapter': 'trainable LayerNorm2560 and Linear2560to256 on original native ROI',
        'original_U2_tracking_checkpoint_available': False, 'initialization': 'new seed0 initialization',
        'training': 'original U2 MOT loss/scheduled forcing/query augmentation, FP32 CPU, 2frame clips',
        'Compact_budget_comparability': '300optimizer steps only; different loss, parameters, optimizer recipe and frame exposure, not FLOP/parameter matched',
        'completed_steps': start_step, 'loss_log': []}
    def save():
        (args.out/'owner_state.json').write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps({'status': state['status'], 'steps': state['completed_steps']}), flush=True)
    save()
    started = time.perf_counter()
    try:
        for step in range(start_step+1, args.steps+1):
            raw = clips[plan[step-1]]
            clip = [{'frame_index': r['frame_index'], 'supervision': r['supervision'],
                     'candidates': model.candidates(r['features'], r['boxes'], r['scores'])} for r in raw]
            output = temporal([clip], step, augment=True)
            loss = output['losses']['total']
            assert math.isfinite(float(loss.detach()))
            optimizer.zero_grad()
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
            assert math.isfinite(norm) and norm > 0
            optimizer.step()
            state['completed_steps'] = step
            if step == 1 or step % 50 == 0:
                state['loss_log'].append({'step': step, 'losses': {k: float(v.detach()) for k, v in output['losses'].items()},
                    'gradient_norm': norm, 'elapsed_seconds': time.perf_counter()-started})
                temporary = args.out/'training_state.tmp'
                torch.save({'step': step, 'plan': plan, 'scope': scope, 'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'torch_rng': torch.get_rng_state()}, temporary)
                temporary.replace(args.out/'latest_training_state.pt')
                save()
        changed = sum(int((initial[k] != v).sum()) for k, v in model.state_dict().items())
        assert changed > 0
        model.eval()
        probe = clips[0][0]
        with torch.no_grad():
            before = model.core(model.candidates(probe['features'], probe['boxes'], probe['scores']),
                                frame_index=probe['frame_index'])
        checkpoint = {'format': 'groundjev-B2-original-U2-core-v1', 'state_dict': model.state_dict(),
                      'training_steps': args.steps, 'seed': 0, 'scope': scope}
        torch.save(checkpoint, args.out/'checkpoint.pt')
        fresh = GroundU2Bridge().eval()
        fresh.load_state_dict(torch.load(args.out/'checkpoint.pt', weights_only=True)['state_dict'])
        with torch.no_grad():
            after = fresh.core(fresh.candidates(probe['features'], probe['boxes'], probe['scores']),
                               frame_index=probe['frame_index'])
        delta = max(float((before[k]-after[k]).abs().max()) for k in ('new_logits', 'candidate_id_tokens'))
        assert delta == 0 and torch.equal(before['track_ids'], after['track_ids'])
        state.update(status='GROUND_U2_CORE300_TRAINING_COMPLETE_OFFICIAL_METRICS_PENDING',
            changed_elements=changed, parameters=sum(p.numel() for p in model.parameters()),
            reload_max_difference=delta, elapsed_seconds=time.perf_counter()-started,
            checkpoint=str(args.out/'checkpoint.pt'))
        save()
    except Exception as error:
        state.update(status='GROUND_U2_BRIDGE_TRAINING_FAILED', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
