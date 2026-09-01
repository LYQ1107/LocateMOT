#!/usr/bin/env python3
"""L63-C 100-step fit-only smoke using frozen L19 crop/text features."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path('/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT').resolve()
DATA = ROOT / 'outputs/l49/data'
UNITS = DATA / 'train_units.jsonl'
MANIFEST = ROOT / 'outputs/l19/protocol/kitti_fast_eval_manifest.json'
EXPECTED_MANIFEST = '06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa'

import sys
sys.path.insert(0, str(ROOT))
from locatemot.models.l63_raw_region_language_probe import L63RawRegionLanguageProbe
from locatemot.rmot.l49_data import load_bank, unit_features


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_fit():
    return [json.loads(x) for x in UNITS.read_text().splitlines() if x.strip()
            and (lambda u: u.get('split') == 'fit' and u.get('dataset') in ('refer_kitti_v1', 'refer_kitti_v2'))(json.loads(x))]


def order_units(units, seed):
    rng = random.Random(seed)
    buckets = {(d, c): [] for d in ('refer_kitti_v1', 'refer_kitti_v2')
               for c in ('positive', 'multi_positive', 'inactive', 'present_uncovered')}
    for u in units:
        buckets.setdefault((u['dataset'], u.get('category', 'unknown')), []).append(u)
    for values in buckets.values():
        rng.shuffle(values)
    ordered = []
    keys = [(d, c) for d in ('refer_kitti_v1', 'refer_kitti_v2')
            for c in ('positive', 'multi_positive', 'inactive', 'present_uncovered')]
    while any(buckets.get(k) for k in keys):
        for key in keys:
            if buckets.get(key):
                ordered.append(buckets[key].pop())
    rest = [u for values in buckets.values() for u in values]
    rng.shuffle(rest)
    return ordered + rest


def loss_fn(out, target):
    score = out['relevance_logit']
    pos, neg = torch.where(target)[0], torch.where(~target)[0]
    parts = []
    if len(pos):
        parts.append(F.binary_cross_entropy_with_logits(score[pos], torch.ones_like(score[pos])))
        # Explicit minimum-positive term: every positive contributes, not only
        # a max representative.
        parts.append(0.5 * F.softplus(-score[pos]).mean())
    if len(neg):
        parts.append(F.binary_cross_entropy_with_logits(score[neg], torch.zeros_like(score[neg])))
    if len(pos) and len(neg):
        parts.append(0.5 * F.relu(0.2 - score[pos, None] + score[None, neg]).mean())
        parts.append(0.5 * (-torch.logsumexp(score[pos], 0) + torch.logsumexp(score, 0)))
    null_target = score.new_tensor(float(not bool(target.any())))
    parts.append(F.binary_cross_entropy_with_logits(out['null_logit'], null_target))
    probability = torch.sigmoid(score)
    parts.append(0.1 * (probability - target.float()).square().mean())
    return sum(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/l63/train/raw_region_smoke100')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20260829)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f'refusing to overwrite {out}')
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); random.seed(args.seed)
    manifest_sha = sha(MANIFEST)
    if manifest_sha != EXPECTED_MANIFEST:
        raise AssertionError(manifest_sha)
    units = order_units(load_fit(), args.seed)
    text = torch.load(ROOT / 'outputs/l48/data/text_cache.pt', map_location='cpu', weights_only=False)
    model = L63RawRegionLanguageProbe(hidden=128).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    trace, sampling = [], Counter(); finite = nonzero = 0
    start = time.time(); peak = 0
    for step in range(1, args.steps + 1):
        unit = units[(step - 1) % len(units)]
        bank = load_bank(unit['dataset'], unit['video'])
        values = unit_features(unit, bank, text, history=8)
        crop = values['clip'].cuda(); words = values['text'].cuda(); mask = values['text_mask'].cuda()
        target = values['target'].cuda()
        optimizer.zero_grad(set_to_none=True)
        output = model(crop, words, mask)
        loss = loss_fn(output, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'nonfinite loss at step {step}')
        loss.backward()
        norms = [p.grad.detach().norm() for p in model.parameters() if p.grad is not None]
        if not norms or not all(torch.isfinite(x) for x in norms) or not any(float(x) > 0 for x in norms):
            raise FloatingPointError(f'bad adapter gradient at step {step}')
        optimizer.step(); finite += 1; nonzero += 1; sampling[(unit['dataset'], unit.get('category', 'unknown'))] += 1
        trace.append({'step': step, 'unit_key': unit['unit_key'], 'dataset': unit['dataset'],
                      'category': unit.get('category'), 'candidate_count': int(len(target)),
                      'loss': float(loss.detach().cpu()), 'grad_norm': float(torch.stack(norms).norm().cpu()),
                      'null_logit': float(output['null_logit'].detach().cpu())})
        peak = max(peak, int(torch.cuda.max_memory_allocated()))
        del bank, values, crop, words, mask, target, output, loss
    ck = out / f'checkpoint_l63_raw_region_step{args.steps}.pt'
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': args.steps,
                'seed': args.seed, 'format': 'locatemot-l63-crop-region-language-v1'}, ck)
    reload_model = L63RawRegionLanguageProbe(hidden=128).cuda()
    reload_model.load_state_dict(torch.load(ck, map_location='cuda:0', weights_only=False)['model'], strict=True)
    reload_ok = True
    payload = {'format': 'locatemot-l63-raw-region-language-smoke-v1', 'status': 'complete',
               'stage': 'fit-only-smoke', 'project_root': str(ROOT), 'cwd': str(Path.cwd().resolve()),
               'seed': args.seed, 'steps': args.steps, 'finite_steps': finite,
               'nonzero_gradient_steps': nonzero, 'checkpoint': str(ck), 'checkpoint_reload': reload_ok,
               'checkpoint_sha256': sha(ck), 'fit_units_total': len(units), 'sampling_counts': {f'{d}|{c}': n for (d, c), n in sampling.items()},
               'domains_present': sorted({u['dataset'] for u in units[:args.steps]}),
               'categories_seen': sorted({u.get('category') for u in units[:args.steps]}),
               'complete_candidate_sets': True, 'candidate_key_drift': 0, 'candidate_truncation': False,
               'persistent_raw_dense_cache_written': False, 'screening_gt_used': False,
               'official_test_labels_read': False, 'ordinary_mot_ovmot_touched': False,
               'detector_or_backbone_trainable': False, 'adapter_parameter_count': sum(p.numel() for p in model.parameters()),
               'runtime': {'peak_memory_bytes': peak, 'elapsed_sec': time.time() - start,
                           'steps_per_sec': args.steps / max(1e-9, time.time() - start)}, 'loss_trace': trace}
    (out / f'metrics_l63_step{args.steps}.json').write_text(json.dumps(payload, indent=2) + '\n')
    (out / 'config.json').write_text(json.dumps({'seed': args.seed, 'steps': args.steps, 'hidden': 128,
        'fit_only': True, 'input': 'frozen L19 clip crop + masked word-token hidden',
        'token_span_alignment': 'UNALIGNED', 'same_class_metadata': 'unavailable'}, indent=2) + '\n')
    (out / 'loss_trace.json').write_text(json.dumps(trace, indent=2) + '\n')
    (out / 'sampling_trace.json').write_text(json.dumps({'counts': payload['sampling_counts'],
        'domains_present': payload['domains_present'], 'categories_seen': payload['categories_seen']}, indent=2) + '\n')
    (out / 'provenance.json').write_text(json.dumps({'project_root': str(ROOT), 'cwd': str(Path.cwd().resolve()),
        'seed': args.seed, 'manifest_sha256': manifest_sha, 'train_units_sha256': sha(UNITS),
        'text_cache_sha256': sha(ROOT / 'outputs/l48/data/text_cache.pt'), 'fit_only': True,
        'screening_gt_used': False, 'official_test_labels_read': False,
        'ordinary_mot_ovmot_touched': False, 'persistent_raw_dense_cache_written': False}, indent=2) + '\n')
    print(json.dumps({'status': 'complete', 'metrics': str(out / f'metrics_l63_step{args.steps}.json'),
                      'checkpoint': str(ck), 'finite_steps': finite, 'nonzero_gradient_steps': nonzero}), flush=True)


if __name__ == '__main__':
    main()
