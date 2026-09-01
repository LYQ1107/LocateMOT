#!/usr/bin/env python3
"""L62-A: independent spatial/fused ROI contract audit.

This reads no labels until all detector feature construction for the fixed
representative units has completed.  It writes compact statistics only.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np

ROOT = Path('/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT').resolve()
OUT = ROOT / 'outputs/l62/audit/spatial_roi_contract'
UNIT_FILES = [ROOT / 'outputs/l49/data/calibration_units.jsonl', ROOT / 'outputs/l49/data/validation_units.jsonl']
MANIFEST = ROOT / 'outputs/l19/protocol/kitti_fast_eval_manifest.json'
MANIFEST_SHA = '06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa'
MMDET = Path('/data1/LWR/LLM/mmdetection-3.3.0')
PYTHON = Path('/home/lwr/anaconda3/envs/masaenv_debug/bin/python')
CONTROL_SENTENCE = 'an unrelated object in a remote scene'
FIXED_KEYS = [
    'refer_kitti_v1|0016|574|207',  # inactive
    'refer_kitti_v1|0016|569|75',   # multi-positive
    'refer_kitti_v1|0016|575|178',  # positive
    'refer_kitti_v2|0015|5965|81',  # present-uncovered
    'refer_kitti_v1|0018|593|42',   # inactive, separate video
]

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def jsonable(x):
    if isinstance(x, (np.floating, np.integer)): return x.item()
    if isinstance(x, np.ndarray): return x.tolist()
    return x

def cpu_lattice_test():
    import torch
    import torch.nn.functional as F
    H, W, h, w = 100., 140., 5, 7
    fmap = torch.arange(h * w, dtype=torch.float32).reshape(1, 1, h, w)
    # Seven point-box candidates exercise N != grid_size and both boundaries.
    xs = torch.tensor([0., 20., 40., 60., 80., 100., 120.])
    ys = torch.tensor([0., 20., 40., 60., 80., 0., 80.])
    gx = 2 * ((xs / W) * w + .5) / w - 1
    gy = 2 * ((ys / H) * h + .5) / h - 1
    got = F.grid_sample(fmap.expand(7, -1, -1, -1), torch.stack([gx, gy], -1).reshape(7, 1, 1, 2), align_corners=False).reshape(-1)
    # Interior points lie exactly at feature-cell centers under this contract.
    expected_idx = torch.tensor([0, 1 * 7 + 1, 2 * 7 + 2, 3 * 7 + 3, 4 * 7 + 4, 5, 4 * 7 + 6], dtype=torch.float32)
    interior_ok = bool(torch.allclose(got[1:-1], expected_idx[1:-1], atol=1e-5))
    boundary_finite = bool(torch.isfinite(got[[0, -1]]).all())
    edge = F.grid_sample(fmap, torch.tensor([[[[1.0, 1.0]]]]), align_corners=False).item()
    return {'passed': interior_ok and boundary_finite and math.isfinite(edge), 'candidate_count': 7, 'grid_size': 1,
            'align_corners': False, 'expected': expected_idx.tolist(), 'sampled': got.tolist(),
            'formula': 'g=2*((pixel/image_size)*feature_size+0.5)/feature_size-1',
            'boundary_finite': boundary_finite, 'out_of_bounds_corner_sample': edge,
            'out_of_bounds_corner_behavior': 'align_corners_false_zero_padding_with_edge_interpolation; not asserted as a feature-cell identity'}

def summarize(values):
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    return {'count': int(a.size), 'mean': float(a.mean()) if a.size else None,
            'std': float(a.std()) if a.size else None, 'min': float(a.min()) if a.size else None,
            'max': float(a.max()) if a.size else None}

def separation(scores, positives):
    s = np.asarray(scores, dtype=np.float64); pos = np.asarray(positives, dtype=np.int64); pos = pos[(pos >= 0) & (pos < len(s))]
    neg = np.asarray([i for i in range(len(s)) if i not in set(pos.tolist())], dtype=np.int64)
    if not len(pos) or not len(neg): return {'positive_count': int(len(pos)), 'negative_count': int(len(neg)), 'pairs': 0, 'hard_violation': None, 'mean_positive_minus_max_negative': None}
    hard = s[neg].max(); d = s[pos] - hard
    return {'positive_count': int(len(pos)), 'negative_count': int(len(neg)), 'pairs': int(len(pos) * len(neg)), 'hard_violation': float(np.mean(d < 0)), 'mean_positive_minus_max_negative': float(d.mean()), 'positive_score': summarize(s[pos]), 'negative_score': summarize(s[neg])}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-root', required=True)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT: raise RuntimeError('wrong cwd')
    if sha256(MANIFEST) != MANIFEST_SHA: raise RuntimeError('manifest SHA mismatch')
    out = Path(args.out_root)
    out = out if out.is_absolute() else ROOT / out
    out = out.resolve()
    if out.exists(): raise FileExistsError(out)
    import torch
    sys.path.insert(0, str(ROOT))
    from tools.l59_fused_common import build_detector, detector_provenance, stream_fused_roi
    units = [json.loads(x) for path in UNIT_FILES for x in path.read_text().splitlines() if x.strip()]
    by_key = {x['unit_key']: x for x in units}
    fixed = [by_key[k] for k in FIXED_KEYS]
    # Feature construction is deliberately completed before reading any labels.
    detector, _, load = build_detector(); detector.eval()
    constructed = []; started = time.time()
    for unit in fixed:
        bank = torch.load(unit['bank_path'], map_location='cpu', weights_only=False)
        roi, text, text_valid, numeric, meta = stream_fused_roi(detector, unit, bank)
        roi_ctrl, text_ctrl, text_valid_ctrl, _, meta_ctrl = stream_fused_roi(detector, unit, bank, sentence=CONTROL_SENTENCE)
        tensors = bank['tensors']; begin, end = int(unit['begin']), int(unit['end']); n = end - begin
        keys = [(str(unit['video']), int(unit['frame_id']), str(Path(unit['bank_path']).resolve()), begin + i) for i in range(n)]
        candidate_index = tensors.get('candidate_index', torch.full((end,), -1))[begin:end].detach().cpu().numpy().tolist()
        per_level = roi.reshape(n, 4, 16, -1)
        text_base = text[0] if text.dim() == 3 else text
        valid_text = text_base[text_valid[0] if text_valid.dim() == 2 else text_valid]
        if not len(valid_text): valid_text = text_base
        query = valid_text.mean(0)
        scores = torch.nn.functional.cosine_similarity(per_level.mean(2), query.reshape(1, 1, -1), dim=-1)
        constructed.append({'unit': unit, 'meta': meta, 'meta_ctrl': meta_ctrl, 'roi': roi.cpu(), 'roi_ctrl': roi_ctrl.cpu(), 'text': text.cpu(), 'text_ctrl': text_ctrl.cpu(), 'text_valid': text_valid.cpu(), 'text_valid_ctrl': text_valid_ctrl.cpu(), 'numeric': numeric.cpu(), 'keys': keys, 'candidate_index': candidate_index, 'scores': scores.cpu().numpy(), 'n': n})
        del bank, roi, roi_ctrl, text, text_ctrl, numeric
    # Labels and frozen control fields are read only now, after construction.
    rows = []; label_audit = []; level_seps = [[] for _ in range(4)]
    for item in constructed:
        unit = item['unit']; labels = np.zeros(item['n'], dtype=bool); inds = np.asarray(unit['positive_indices'], dtype=np.int64)
        if len(inds) and (inds.min() < 0 or inds.max() >= item['n']): raise AssertionError('positive index range')
        labels[inds] = True; roi = item['roi'].numpy(); text = item['text'].numpy(); tv = item['text_valid'].numpy().reshape(-1).astype(bool)
        q = text[0, tv].mean(0) if text.ndim == 3 and tv.any() else text.reshape(-1, text.shape[-1]).mean(0)
        roi_levels = roi.reshape(item['n'], 4, 16, -1)
        level_scores = np.stack([np.dot(roi_levels[:, l].mean(1), q) / np.maximum(1e-9, np.linalg.norm(roi_levels[:, l].mean(1), axis=1) * np.linalg.norm(q)) for l in range(4)], 1)
        ctrl_fields = {}
        for name in ('objectness', 'clip', 'history', 'appearance', 'visual', 'identity'):
            if name in unit.get('_unused', {}): ctrl_fields[name] = True
        bank = torch.load(unit['bank_path'], map_location='cpu', weights_only=False); t = bank['tensors']
        available = [k for k in t.keys() if any(w in k.lower() for w in ('clip', 'history', 'appear', 'visual', 'identity', 'objectness'))]
        for k in available[:8]: ctrl_fields[k] = separation(np.asarray(t[k][int(unit['begin']):int(unit['end'])]).reshape(item['n'], -1).mean(1), inds)
        for l in range(4): level_seps[l].append(separation(level_scores[:, l], inds))
        fused_sep = separation(item['scores'].mean(1), inds)
        deltas = item['roi'].numpy() - item['roi_ctrl'].numpy(); rel = float(np.linalg.norm(deltas) / max(1e-9, np.linalg.norm(item['roi'].numpy())))
        per_level_delta = [float(np.linalg.norm(deltas[:, l*16:(l+1)*16]) / max(1e-9, np.linalg.norm(item['roi'].numpy()[:, l*16:(l+1)*16]))) for l in range(4)]
        text_arr = item['text'].numpy(); text_ctrl_arr = item['text_ctrl'].numpy()
        text_valid_arr = item['text_valid'].numpy().reshape(-1).astype(bool)
        text_ctrl_valid_arr = item['text_valid_ctrl'].numpy().reshape(-1).astype(bool)
        text_vec = text_arr[0, text_valid_arr].mean(0) if text_arr.ndim == 3 and text_valid_arr.any() else text_arr.reshape(-1, text_arr.shape[-1]).mean(0)
        text_ctrl_vec = text_ctrl_arr[0, text_ctrl_valid_arr].mean(0) if text_ctrl_arr.ndim == 3 and text_ctrl_valid_arr.any() else text_ctrl_arr.reshape(-1, text_ctrl_arr.shape[-1]).mean(0)
        text_delta = float(np.linalg.norm(text_vec - text_ctrl_vec) / max(1e-9, np.linalg.norm(text_vec)))
        duplicate_count = int(len(item['candidate_index']) - len(set(item['candidate_index']))) if item['candidate_index'] and item['candidate_index'][0] != -1 else None
        entry = {'unit_key': unit['unit_key'], 'dataset': unit['dataset'], 'video': unit['video'], 'frame_id': unit['frame_id'], 'category': unit.get('category'), 'candidate_count': item['n'], 'immutable_key_count': len(set(item['keys'])), 'candidate_index_duplicate_count': duplicate_count, 'row_order_exact': item['keys'] == sorted(item['keys'], key=lambda x: x[3]), 'roi_shape': list(item['roi'].shape), 'per_level_token_mean': [summarize(item['roi'].numpy()[:, l*16:(l+1)*16].mean((0, 1))) for l in range(4)], 'within_box_token_variance': [float(item['roi'].numpy()[:, l*16:(l+1)*16].var()) for l in range(4)], 'between_candidate_cosine': summarize([np.dot(item['roi'].numpy()[i].mean(0), item['roi'].numpy()[j].mean(0)) / max(1e-9, np.linalg.norm(item['roi'].numpy()[i].mean(0))*np.linalg.norm(item['roi'].numpy()[j].mean(0))) for i in range(item['n']) for j in range(i)]), 'expression_conditioning': {'relative_roi_l2': rel, 'relative_text_l2': text_delta, 'per_level_roi_relative_l2': per_level_delta, 'changed_levels_gt_1e-4': int(sum(x > 1e-4 for x in per_level_delta)), 'control_sentence': CONTROL_SENTENCE}, 'encoder_meta': {'spatial_shapes': item['meta']['spatial_shapes'], 'starts': item['meta']['level_start_index'], 'roi_valid_fraction_per_level': item['meta'].get('roi_valid_fraction_per_level'), 'roi_valid_samples': item['meta'].get('roi_valid_samples'), 'roi_total_samples': item['meta'].get('roi_total_samples'), 'key_padding_mask_was_none': item['meta'].get('zero_padding_mask'), 'text_attention_mask_missing': item['meta'].get('text_attention_mask_missing'), 'scale_factor': item['meta']['scale_factor'], 'img_shape': item['meta']['img_shape'], 'ori_shape': item['meta']['ori_shape']}, 'oracle_label_audit': {'fused_roi_query_cosine': fused_sep, 'per_level': [separation(level_scores[:, l], inds) for l in range(4)], 'frozen_l19_fields_available': ctrl_fields}}
        rows.append(entry); label_audit.append(entry)
        del bank
    contract_ok = True
    for item in constructed:
        m = item['meta']; shapes = np.asarray(m['spatial_shapes']); starts = np.asarray(m['level_start_index']); expected = np.cumsum(np.r_[0, shapes[:-1, 0] * shapes[:-1, 1]]).astype(int)
        contract_ok &= bool(np.array_equal(starts, expected) and m['candidate_count'] == item['n'] and len(item['keys']) == item['n'] and len(set(item['keys'])) == item['n'] and np.isfinite(item['roi'].numpy()).all())
    expression_values = [r['expression_conditioning']['relative_roi_l2'] for r in rows]
    changed_levels = sum(r['expression_conditioning']['changed_levels_gt_1e-4'] >= 2 for r in rows)
    separator = any(x.get('pairs', 0) and x.get('hard_violation', 1.0) < 1.0 for r in rows for x in r['oracle_label_audit']['per_level'])
    cpu = cpu_lattice_test()
    if not contract_ok or not cpu['passed']:
        conclusion = 'alignment_invalid'
    elif float(np.mean(expression_values)) > 1e-4 and changed_levels >= 2 and separator:
        conclusion = 'alignment_valid_signal_present'
    else: conclusion = 'alignment_valid_signal_weak'
    result = {'format': 'locatemot-l62-spatial-roi-contract-audit-v1', 'status': 'complete', 'conclusion': conclusion, 'conclusion_rule': 'invalid if metadata/impulse/key/finite checks fail; present if mean expression ROI relative L2 >1e-4, at least two levels change in at least one fixed unit, and one level has finite positive-vs-hard separation; otherwise weak', 'project_root': str(ROOT), 'cwd': str(Path.cwd().resolve()), 'fixed_units': FIXED_KEYS, 'labels_read_after_feature_construction': True, 'cpu_impulse_test': cpu, 'contract_ok': contract_ok, 'expression_change_summary': summarize(expression_values), 'units_with_two_changed_levels': changed_levels, 'fused_level_separator_present': separator, 'units': rows, 'detector_provenance': detector_provenance(load), 'elapsed_sec': time.time() - started, 'screening_gt_used': False, 'official_test_labels_read': False, 'ordinary_mot_ovmot_touched': False, 'persistent_dense_cache_written': False, 'token_span_alignment': 'UNALIGNED', 'static_motion_decomposition': 'UNALIGNED', 'unit_sources': [str(x) for x in UNIT_FILES]}
    out.mkdir(parents=True); (out / 'spatial_roi_audit.json').write_text(json.dumps(result, indent=2, default=jsonable) + '\n'); (out / 'provenance.json').write_text(json.dumps({'source_units': [str(x) for x in UNIT_FILES], 'manifest_sha256': MANIFEST_SHA, 'labels_used_only_after_construction': True, 'no_persistent_feature_cache': True}, indent=2) + '\n')
    with (out / 'representative_stats.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['unit_key', 'category', 'candidate_count', 'relative_roi_l2', 'relative_text_l2', 'changed_levels', 'fused_hard_violation', 'fused_mean_margin']); w.writeheader()
        for r in rows: w.writerow({'unit_key': r['unit_key'], 'category': r['category'], 'candidate_count': r['candidate_count'], 'relative_roi_l2': r['expression_conditioning']['relative_roi_l2'], 'relative_text_l2': r['expression_conditioning']['relative_text_l2'], 'changed_levels': r['expression_conditioning']['changed_levels_gt_1e-4'], 'fused_hard_violation': r['oracle_label_audit']['fused_roi_query_cosine']['hard_violation'], 'fused_mean_margin': r['oracle_label_audit']['fused_roi_query_cosine']['mean_positive_minus_max_negative']})
    print(json.dumps({'status': 'complete', 'conclusion': conclusion, 'out': str(out), 'expression_change': result['expression_change_summary'], 'separator_present': separator}), flush=True)

if __name__ == '__main__': main()
