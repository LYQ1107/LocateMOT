#!/usr/bin/env python3
"""Audit L59 candidate/NULL output contract without detector inference."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np

ROOT = Path('/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT').resolve()
SRC = ROOT / 'outputs/l59/eval/semantic_16cal_24val/score_records.jsonl'
OUT = ROOT / 'outputs/l62/audit/l59_null_contract'
MANIFEST = ROOT / 'outputs/l19/protocol/kitti_fast_eval_manifest.json'
MANIFEST_SHA = '06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa'

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()

def metric(rows, threshold, null_threshold=None):
    tp = fp = fn = top1 = top5 = empty = null_false = 0
    strict, best, average, violations, multi = [], [], [], [], []
    for row in rows:
        s = np.asarray(row['l59'], dtype=np.float64); y = np.asarray(row['label'], dtype=bool)
        if not np.isfinite(s).all() or not np.isfinite(float(row['null_logit'])): raise AssertionError('nonfinite score')
        suppress = null_threshold is not None and float(row['null_logit']) >= float(null_threshold)
        chosen = (s >= float(threshold)) & (not suppress)
        tp += int((chosen & y).sum()); fp += int((chosen & ~y).sum()); fn += int((~chosen & y).sum())
        empty += int(not chosen.any()); null_false += int(not y.any() and chosen.any())
        pos = np.flatnonzero(y); neg = np.flatnonzero(~y)
        if len(pos):
            order = np.argsort(-s, kind='stable'); top1 += int(y[order[:1]].any()); top5 += int(y[order[:5]].any())
            if len(pos) > 1: multi.append(float((chosen & y).sum() / len(pos)))
            if len(neg):
                d = float(s[pos].min() - s[neg].max()); strict.append(d)
                best.append(float(s[pos].max() - s[neg].max())); average.append(float(s[pos].mean() - s[neg].max())); violations.append(d < 0)
    units = len(rows); positives = sum(int(np.asarray(r['label'], dtype=bool).sum()) for r in rows); selected = tp + fp
    def dist(x): return {'count': len(x), 'mean': float(np.mean(x)) if x else None}
    return {'units': units, 'candidate_rows': sum(len(r['label']) for r in rows), 'positive_rows': positives,
            'top1': top1 / max(1, sum(bool(np.asarray(r['label']).any()) for r in rows)),
            'top5': top5 / max(1, sum(bool(np.asarray(r['label']).any()) for r in rows)),
            'candidate_precision': tp / max(1, selected), 'candidate_recall': tp / max(1, tp + fn),
            'fp_per_frame': fp / max(1, units), 'predictions_per_positive': selected / max(1, positives),
            'empty_rate': empty / max(1, units), 'null_false_acceptance': null_false / max(1, units),
            'hard_violation': float(np.mean(violations)) if violations else None, 'strict_margin': dist(strict),
            'best_margin': dist(best), 'average_margin': dist(average), 'multi_positive_recall': float(np.mean(multi)) if multi else None,
            'threshold': float(threshold), 'null_threshold': None if null_threshold is None else float(null_threshold)}

def fit_candidate(rows):
    vals = np.unique(np.concatenate([np.asarray(r['l59'], dtype=np.float64) for r in rows])); best = None
    for t in vals.tolist() + [float(vals.min()) - 1e-6, float(vals.max()) + 1e-6]:
        m = metric(rows, t); tp = fp = fn = 0
        for r in rows:
            s = np.asarray(r['l59']); y = np.asarray(r['label'], dtype=bool); z = s >= t
            tp += int((z & y).sum()); fp += int((z & ~y).sum()); fn += int((~z & y).sum())
        f1 = 2 * tp / max(1, 2 * tp + fp + fn); key = (f1, -m['fp_per_frame'], -float(t))
        if best is None or key > best[0]: best = (key, float(t), tp, fp, fn)
    return {'threshold': best[1], 'objective': 'candidate-level calibration F1; tie lower FP/frame; tie lower threshold',
            'tp': best[2], 'fp': best[3], 'fn': best[4], 'labels_source': '16 calibration rows only'}

def fit_null(rows, candidate_threshold):
    vals = np.unique(np.r_[[float(r['null_logit']) for r in rows]]); best = None
    for t in vals.tolist() + [float(vals.min()) - 1e-6, float(vals.max()) + 1e-6]:
        pred = []; truth = []
        for r in rows:
            y = np.asarray(r['label'], dtype=bool); pred.append(bool((np.asarray(r['l59']) >= candidate_threshold).any()) and float(r['null_logit']) < t); truth.append(bool(y.any()))
        tp = sum(a and b for a, b in zip(pred, truth)); fp = sum(a and not b for a, b in zip(pred, truth)); fn = sum((not a) and b for a, b in zip(pred, truth))
        f1 = 2 * tp / max(1, 2 * tp + fp + fn); inactive = sum(not b and a for a, b in zip(pred, truth)) / max(1, sum(not b for b in truth)); key = (f1, -inactive, -float(t))
        if best is None or key > best[0]: best = (key, float(t), tp, fp, fn, inactive)
    return {'null_threshold': best[1], 'rule': 'suppress all candidates iff null_logit >= threshold', 'objective': 'frame-presence F1; tie lower inactive false acceptance; tie lower threshold', 'tp': best[2], 'fp': best[3], 'fn': best[4], 'calibration_inactive_false_acceptance': best[5], 'labels_source': '16 calibration rows only'}

def main():
    if Path.cwd().resolve() != ROOT: raise RuntimeError('wrong cwd')
    if sha256(MANIFEST) != MANIFEST_SHA: raise RuntimeError('manifest SHA mismatch')
    if OUT.exists(): raise FileExistsError(OUT)
    rows = [json.loads(x) for x in SRC.read_text().splitlines() if x.strip()]
    if len(rows) != 40 or len({r['unit_key'] for r in rows}) != 40: raise AssertionError('expected 40 unique units')
    cal, val = rows[:16], rows[16:]; candidate = fit_candidate(cal); null = fit_null(cal, candidate['threshold'])
    result = {'format': 'locatemot-l62-l59-null-contract-audit-v1', 'status': 'complete', 'project_root': str(ROOT), 'cwd': str(Path.cwd().resolve()), 'source': str(SRC), 'source_sha256': sha256(SRC), 'manifest_sha256': MANIFEST_SHA, 'calibration_units': 16, 'validation_units': 24, 'candidate_rows_retained': True, 'calibration_selection': {'candidate': candidate, 'null': null}, 'calibration_candidate_only': metric(cal, candidate['threshold']), 'calibration_final_candidate_plus_null': metric(cal, candidate['threshold'], null['null_threshold']), 'validation_candidate_only': metric(val, candidate['threshold']), 'validation_final_candidate_plus_null': metric(val, candidate['threshold'], null['null_threshold']), 'screening_gt_used': False, 'official_test_labels_read': False, 'ordinary_mot_ovmot_touched': False, 'no_detector_rerun': True}
    OUT.mkdir(parents=True); (OUT / 'null_contract.json').write_text(json.dumps(result, indent=2) + '\n'); (OUT / 'provenance.json').write_text(json.dumps({'source_sha256': result['source_sha256'], 'calibration_only_selection': True, 'no_persistent_cache_written': True}, indent=2) + '\n'); print(json.dumps({'status': 'complete', 'out': str(OUT), 'candidate_threshold': candidate['threshold'], 'null_threshold': null['null_threshold'], 'validation_candidate_only': result['validation_candidate_only'], 'validation_final': result['validation_final_candidate_plus_null']}), flush=True)

if __name__ == '__main__': main()
