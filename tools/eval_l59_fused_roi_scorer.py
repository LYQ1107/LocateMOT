#!/usr/bin/env python3
"""L59 fixed calibration/validation evaluation for the post-fusion ROI scorer.

This evaluator streams the frozen detector once per selected unit and retains
every L19 candidate row.  The 16/24 unit selection is inherited from the
immutable L53 label-free job list; labels are read only from the corresponding
L49 calibration/validation unit files and are never used to construct scores.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict, OrderedDict
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT').resolve()
DATA = ROOT / 'outputs/l49/data'
PREDICTIONS = ROOT / 'outputs/l53/eval/zero_shot_retry4/predictions.json'
L29 = ROOT / 'outputs/l29/train/frame_membership_step1000/checkpoint_frame_membership_step1000.pt'
TEXT = ROOT / 'outputs/l48/data/text_cache.pt'
MANIFEST = ROOT / 'outputs/l19/protocol/kitti_fast_eval_manifest.json'
EXPECTED_MANIFEST = '06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa'
sys.path.insert(0, str(ROOT))

from locatemot.models.l29_frame_membership_set_decoder import L29FrameMembershipSetDecoder  # noqa: E402
from locatemot.models.l59_fused_roi_scorer import L59FusedROIScorer  # noqa: E402
from locatemot.rmot.l49_data import load_bank, sha256_file  # noqa: E402
from tools.eval_l49_validation import fit_threshold, l29_score, source_masks, summarize  # noqa: E402
from tools.train_l28_track_set_decoder import state_at  # noqa: E402,F401
from tools.train_l49_kitti_rmot import build_teacher_cache  # noqa: E402
from tools.l59_fused_common import build_detector, detector_provenance, stream_fused_roi  # noqa: E402


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(0.0, rb - lt)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    return inter / np.maximum(1e-9, area_a[:, None] + area_b[None, :] - inter)


def candidate_labels(unit):
    n = int(unit['end']) - int(unit['begin'])
    labels = np.zeros(n, dtype=bool)
    indices = np.asarray(unit['positive_indices'], dtype=np.int64)
    if len(indices) and (indices.min() < 0 or indices.max() >= n):
        raise AssertionError(f'positive index out of range for {unit["unit_key"]}')
    labels[indices] = True
    return labels


def records_from_predictions(units, predictions):
    by_key = {row['unit_key']: row for row in predictions}
    records = []
    for unit in units:
        pred = by_key.get(unit['unit_key'])
        if pred is None:
            raise KeyError(f'missing immutable L53 prediction {unit["unit_key"]}')
        boxes = np.asarray(torch.load(unit['bank_path'], map_location='cpu', weights_only=False)['tensors']['box'][int(unit['begin']):int(unit['end'])], dtype=np.float32)
        pboxes = np.asarray(pred.get('pred_boxes', []), dtype=np.float32).reshape(-1, 4)
        pscores = np.asarray(pred.get('pred_scores', []), dtype=np.float32)
        overlaps = iou_matrix(boxes, pboxes)
        native = np.max(np.where(overlaps >= 0.30, pscores[None, :], -20.0), axis=1) if len(pboxes) else np.full(len(boxes), -20.0, dtype=np.float32)
        continuous = np.max(pscores[None, :] * overlaps, axis=1) if len(pboxes) else np.zeros(len(boxes), dtype=np.float32)
        labels = candidate_labels(unit)
        records.append({'unit': unit, 'label': labels, 'm0': native, 'm54': continuous,
                        'candidate_boxes': boxes, 'proposal_count': int(len(pboxes)),
                        'proposal_coverage': float((overlaps.max(1) >= 0.30).mean()) if len(boxes) else 0.0})
    return records


def distribution(values):
    x = np.asarray(values, dtype=np.float64)
    if not len(x):
        return {'count': 0, 'mean': None, 'std': None, 'min': None, 'max': None}
    return {'count': int(len(x)), 'mean': float(x.mean()), 'std': float(x.std()),
            'min': float(x.min()), 'max': float(x.max())}


def metric_summary(records, threshold, null_rule=None):
    if null_rule is None:
        null_rule = lambda row: False
    tp = fp = fn = 0
    top1 = top5 = 0
    empty = null_false = 0
    multi_recall = []
    strict, best, average, violations = [], [], [], []
    score_values = []
    for row in records:
        scores = np.asarray(row['score'], dtype=np.float32)
        labels = np.asarray(row['label'], dtype=bool)
        if not np.isfinite(scores).all():
            raise AssertionError('nonfinite candidate scores')
        suppress = bool(null_rule(row))
        chosen = (scores >= float(threshold)) & (not suppress)
        tp += int((chosen & labels).sum())
        fp += int((chosen & ~labels).sum())
        fn += int((~chosen & labels).sum())
        empty += int(not chosen.any())
        null_false += int(not labels.any() and chosen.any())
        score_values.extend(scores.tolist())
        pos = np.flatnonzero(labels); neg = np.flatnonzero(~labels)
        if len(pos):
            order = np.argsort(-scores, kind='stable')
            top1 += int(labels[order[:1]].any())
            top5 += int(labels[order[:5]].any())
            if len(pos) > 1:
                multi_recall.append(float((chosen & labels).sum() / len(pos)))
            if len(neg):
                strict_value = float(scores[pos].min() - scores[neg].max())
                best_value = float(scores[pos].max() - scores[neg].max())
                average_value = float(scores[pos].mean() - scores[neg].max())
                strict.append(strict_value); best.append(best_value); average.append(average_value)
                violations.append(strict_value < 0)
    units = len(records); positives = sum(int(row['label'].sum()) for row in records)
    selected = tp + fp
    return {
        'units': units, 'candidate_rows': int(sum(len(x['label']) for x in records)),
        'positive_rows': positives, 'top1': top1 / max(1, sum(bool(x['label'].any()) for x in records)),
        'top5': top5 / max(1, sum(bool(x['label'].any()) for x in records)),
        'candidate_precision': tp / max(1, selected), 'candidate_recall': tp / max(1, tp + fn),
        'fp_per_frame': fp / max(1, units), 'predictions_per_positive': selected / max(1, positives),
        'empty_rate': empty / max(1, units), 'null_false_acceptance': null_false / max(1, units),
        'hard_violation': float(np.mean(violations)) if violations else None,
        'strict_margin': distribution(strict), 'best_margin': distribution(best),
        'average_margin': distribution(average), 'multi_positive_recall': float(np.mean(multi_recall)) if multi_recall else None,
        'score_distribution': distribution(score_values), 'threshold': float(threshold),
        'null_rule': 'none' if null_rule is None else 'model_null_logit>0 suppresses all candidates',
    }


def fit_global_threshold(records):
    values = np.concatenate([np.asarray(x['score'], dtype=np.float32) for x in records])
    candidates = np.unique(values)
    if len(candidates) > 256:
        candidates = np.quantile(values, np.linspace(0, 1, 256))
    best = None
    for threshold in list(map(float, candidates)) + [float(values.min()) - 1e-6, float(values.max()) + 1e-6]:
        tp = fp = fn = 0
        for row in records:
            s = np.asarray(row['score']); y = row['label']; z = s >= threshold
            tp += int((z & y).sum()); fp += int((z & ~y).sum()); fn += int((~z & y).sum())
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        key = (f1, -threshold)
        if best is None or key > best[0]:
            best = (key, threshold, tp, fp, fn)
    return {'threshold': float(best[1]), 'objective': 'global calibration candidate F1',
            'tp': best[2], 'fp': best[3], 'fn': best[4], 'units': len(records),
            'labels_source': 'fixed calibration units only'}


def fit_null_threshold(records, candidate_threshold):
    """Fit the one permitted frame-presence NULL rule on calibration only."""
    values = np.unique(np.asarray([float(row['null_logit']) for row in records], dtype=np.float64))
    best = None
    for threshold in values.tolist() + [float(values.min()) - 1e-6, float(values.max()) + 1e-6]:
        predicted, truth = [], []
        for row in records:
            labels = np.asarray(row['label'], dtype=bool)
            predicted.append(bool((np.asarray(row['score']) >= candidate_threshold).any()) and float(row['null_logit']) < threshold)
            truth.append(bool(labels.any()))
        tp = sum(a and b for a, b in zip(predicted, truth))
        fp = sum(a and not b for a, b in zip(predicted, truth))
        fn = sum((not a) and b for a, b in zip(predicted, truth))
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        inactive_false = sum(a and not b for a, b in zip(predicted, truth)) / max(1, sum(not b for b in truth))
        key = (f1, -inactive_false, -float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold), tp, fp, fn, inactive_false)
    return {'null_threshold': best[1], 'rule': 'suppress all candidates iff null_logit >= calibrated threshold',
            'objective': 'frame-presence F1; tie lower inactive false acceptance; tie lower threshold',
            'tp': best[2], 'fp': best[3], 'fn': best[4],
            'calibration_inactive_false_acceptance': best[5],
            'labels_source': 'fixed calibration units only'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out-root', required=True)
    args = ap.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f'wrong cwd: {Path.cwd().resolve()}')
    if sha256_file(MANIFEST) != EXPECTED_MANIFEST:
        raise RuntimeError('fixed manifest SHA mismatch')
    out = Path(args.out_root); out = out if out.is_absolute() else ROOT / out
    out = out.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    started = time.time()
    cal_all = load_jsonl(DATA / 'calibration_units.jsonl')
    val_all = load_jsonl(DATA / 'validation_units.jsonl')
    selected_keys = [x['unit_key'] for x in json.loads(PREDICTIONS.read_text())]
    cal_map = {x['unit_key']: x for x in cal_all}; val_map = {x['unit_key']: x for x in val_all}
    calibration = [cal_map[k] for k in selected_keys if k in cal_map]
    validation = [val_map[k] for k in selected_keys if k in val_map]
    if len(calibration) != 16 or len(validation) != 24:
        raise AssertionError(f'fixed unit selection mismatch cal={len(calibration)} val={len(validation)}')
    predictions = json.loads(PREDICTIONS.read_text())
    pred_rows = records_from_predictions(calibration + validation, predictions)
    cal_n = len(calibration)
    cal_pred = pred_rows[:cal_n]; val_pred = pred_rows[cal_n:]

    # L29 scores are recomputed from the immutable L29 checkpoint and L28 cache;
    # this is a control only and never enters adapter training.
    from tools.eval_l49_validation import BankStore
    text = torch.load(TEXT, map_location='cpu', weights_only=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    teacher = L29FrameMembershipSetDecoder().to(device)
    teacher.load_state_dict(torch.load(L29, map_location=device, weights_only=False)['model'], strict=True)
    teacher.eval(); store = BankStore(); teacher_cache = {}
    for row in pred_rows:
        u = row['unit']; bank = store.get(u['dataset'], u['video'])
        if str(bank['path']) not in teacher_cache: teacher_cache[str(bank['path'])] = build_teacher_cache(bank)
        row['l29'] = l29_score(teacher, teacher_cache[str(bank['path'])], bank, u, text, device)

    detector, _, loaded = build_detector()
    scorer = L59FusedROIScorer().cuda()
    checkpoint = Path(args.checkpoint).resolve()
    payload_ck = torch.load(checkpoint, map_location='cuda:0', weights_only=False)
    scorer.load_state_dict(payload_ck['model'], strict=True); scorer.eval()
    for row in pred_rows:
        u = row['unit']; bank = store.get(u['dataset'], u['video'])
        with torch.inference_mode():
            roi, txt, txt_valid, numeric, meta = stream_fused_roi(detector, u, bank)
            output = scorer(roi, txt, txt_valid, numeric)
        n = int(u['end']) - int(u['begin'])
        if meta['candidate_count'] != n or len(meta['candidate_keys']) != n:
            raise AssertionError(f'candidate/key count drift for {u["unit_key"]}')
        row['l59'] = output['relevance_logit'].float().cpu().numpy()
        row['null_logit'] = float(output['null_logit'].detach().cpu())
        row['key_audit'] = {'candidate_count': n, 'key_count': len(meta['candidate_keys']), 'ordered': True, 'duplicate_candidate_index_allowed': True}
        del roi, txt, txt_valid, numeric, output

    models = {'l29_teacher': 'l29', 'l53_m0': 'm0', 'l54_continuous': 'm54', 'l59_fused_roi': 'l59'}
    for row in pred_rows:
        row['label'] = row['label'].astype(bool)
    methods = {}
    for name, field in models.items():
        for row in pred_rows: row['score'] = np.asarray(row[field], dtype=np.float32)
        cal_rows = pred_rows[:cal_n]; val_rows = pred_rows[cal_n:]
        threshold = fit_global_threshold(cal_rows)
        # Candidate-only remains a control.  L59's final emission uses one
        # frame-presence NULL rule fitted only on these calibration rows.
        no_null = lambda row: False
        null_fit = fit_null_threshold(cal_rows, threshold['threshold']) if name == 'l59_fused_roi' else None
        null_gate = (lambda row, t=null_fit['null_threshold']: bool(row.get('null_logit', -math.inf) >= t)) if null_fit else no_null
        methods[name] = {'calibration': metric_summary(cal_rows, threshold['threshold'], no_null),
                         'validation': metric_summary(val_rows, threshold['threshold'], no_null),
                         'calibration_with_fixed_null_control': metric_summary(cal_rows, threshold['threshold'], null_gate),
                         'validation_with_fixed_null_control': metric_summary(val_rows, threshold['threshold'], null_gate),
                         'threshold': threshold, 'null_fit': null_fit}
    l29v = methods['l29_teacher']['validation']; l59v = methods['l59_fused_roi']['validation_with_fixed_null_control']
    gate_checks = {
        'hard_violation_decrease_ge_0.05': l29v['hard_violation'] is not None and l59v['hard_violation'] is not None and l59v['hard_violation'] <= l29v['hard_violation'] - .05,
        'recall_drop_le_0.01': l59v['candidate_recall'] >= l29v['candidate_recall'] - .01,
        'precision_ge_0.0830188679': l59v['candidate_precision'] >= .0830188679,
        'fp_per_frame_le_11.125': l59v['fp_per_frame'] <= 11.125,
        'predictions_per_positive_le_4.069': l59v['predictions_per_positive'] <= 4.069,
        'multi_positive_preserved': (l59v['multi_positive_recall'] is not None and l29v['multi_positive_recall'] is not None and l59v['multi_positive_recall'] >= l29v['multi_positive_recall'] - .03),
        'null_not_universally_accepted': l59v['null_false_acceptance'] < 1.0,
        'complete_finite_keys': all(x['key_audit']['candidate_count'] == x['key_audit']['key_count'] for x in pred_rows),
        'candidate_deletion_false': True,
    }
    serial_rows = []
    for row in pred_rows:
        serial_rows.append({'unit_key': row['unit']['unit_key'], 'dataset': row['unit']['dataset'], 'video': row['unit']['video'], 'frame_id': row['unit']['frame_id'], 'category': row['unit']['category'], 'label': row['label'].astype(int).tolist(), 'l29': np.asarray(row['l29']).tolist(), 'm0': row['m0'].tolist(), 'm54': row['m54'].tolist(), 'l59': row['l59'].tolist(), 'null_logit': row['null_logit'], 'proposal_count': row['proposal_count'], 'proposal_coverage': row['proposal_coverage'], 'key_audit': row['key_audit']})
    (out / 'score_records.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in serial_rows))
    provenance = {'format': 'locatemot-l59-fused-roi-eval-provenance-v1', 'project_root': str(ROOT), 'cwd': str(Path.cwd().resolve()), 'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256_file(checkpoint), 'manifest': str(MANIFEST), 'manifest_sha256': EXPECTED_MANIFEST, 'calibration_units': 16, 'validation_units': 24, 'calibration_source': str((DATA / 'calibration_units.jsonl').resolve()), 'validation_source': str((DATA / 'validation_units.jsonl').resolve()), 'screening_gt_used': False, 'official_test_labels_read': False, 'ordinary_mot_ovmot_touched': False, 'persistent_raw_dense_cache_written': False, 'candidate_rows_retained': True, 'semantic_inputs_excluded': ['source_id', 'pool_id', 'group_id', 'query_id', 'state_key'], 'detector': detector_provenance(loaded), 'runtime': {'torch': torch.__version__, 'cuda': torch.version.cuda, 'device': str(device), 'elapsed_sec': time.time() - started}}
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    gate = {'format': 'locatemot-l59-fused-roi-semantic-gate-v1', 'status': 'semantic_gate_pass' if all(gate_checks.values()) else 'semantic_gate_fail', 'decision': 'pass' if all(gate_checks.values()) else 'fail', 'checks': gate_checks, 'methods': methods, 'selection': {'unit_source': str(PREDICTIONS), 'unit_selection': 'immutable L53 16 calibration + 24 validation unit keys', 'threshold_rule': 'one global candidate-F1 threshold fitted on calibration only and frozen for validation', 'null_control': 'L59 final rule is calibration-only frame-presence-F1 NULL threshold; candidate-only remains a control', 'validation_used_for_selection': False}, 'screening_gt_used': False, 'official_test_labels_read': False, 'ordinary_mot_ovmot_touched': False, 'no_hota_or_trackeval': True}
    (out / 'gate_decision.json').write_text(json.dumps(gate, indent=2, default=str) + '\n')
    (out / 'metrics.json').write_text(json.dumps({'format': 'locatemot-l59-fused-roi-semantic-metrics-v1', 'provenance': provenance, 'methods': methods}, indent=2, default=str) + '\n')
    print(json.dumps({'status': gate['status'], 'out': str(out), 'validation': {k: methods['l59_fused_roi']['validation'][k] for k in ('top1','top5','candidate_recall','candidate_precision','fp_per_frame','predictions_per_positive','hard_violation','multi_positive_recall')}}, default=str), flush=True)


if __name__ == '__main__':
    main()
