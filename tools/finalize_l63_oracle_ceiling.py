#!/usr/bin/env python3
"""Finalize compact, derived L63 ceiling statistics without detector access."""
from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT').resolve()
SRC = ROOT / 'outputs/l63/audit/oracle_ceiling_retry4'
OUT = ROOT / 'outputs/l63/audit/oracle_ceiling_final'
BANK_ROOT = ROOT / 'outputs/l19/dual_banks_features/kitti'
sys_path = str(ROOT)


def cosine(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.dot(a, b) / max(1e-8, np.linalg.norm(a) * np.linalg.norm(b)))


def load_sidecar(path, count):
    p = path.with_suffix('.labels.json')
    labels = json.loads(p.read_text()).get('candidate_gt', [])
    if len(labels) != count:
        raise AssertionError(f'label length mismatch {p}')
    return [None if x is None else str(x) for x in labels]


def pair_auc(pos, neg):
    if not pos or not neg:
        return None
    values = sorted([(float(x), 1) for x in pos] + [(float(x), 0) for x in neg])
    seen = wins = 0.0
    for _, label in values:
        if label:
            wins += seen
        else:
            seen += 1
    return float(wins / max(1, len(pos) * len(neg)))


def pair_group_summary(same, different):
    if not same or not different:
        return {'same_count': len(same), 'different_count': len(different),
                'roc_auc': None, 'pair_order_violation': None,
                'finite_sample_note': 'one or both pair classes unavailable'}
    ordered = np.sort(np.asarray(different, np.float64))
    violations = sum(len(ordered) - np.searchsorted(ordered, x, side='left') for x in same)
    return {'same_count': len(same), 'different_count': len(different),
            'roc_auc': pair_auc(same, different),
            'pair_order_violation': float(violations / max(1, len(same) * len(different))),
            'same_similarity': {'mean': float(np.mean(same)), 'std': float(np.std(same))},
            'different_similarity': {'mean': float(np.mean(different)), 'std': float(np.std(different))},
            'finite_sample_note': 'descriptive fixed-unit pair audit; no population CI'}


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f'refusing non-empty output {OUT}')
    OUT.mkdir(parents=True)
    source = json.loads((SRC / 'oracle_ceiling.json').read_text())
    records = [json.loads(x) for x in (SRC / 'unit_records.jsonl').read_text().splitlines() if x.strip()]
    if len(records) != 40:
        raise AssertionError(len(records))
    # Load frozen row features and labels only for derived identity/sequence
    # statistics.  The detector ROI features were already discarded by the
    # streaming audit and are never reconstructed here.
    row_info = {}
    bank_cache = {}
    for r in records:
        for key in r['row_keys']:
            key = tuple(key)
            if key in row_info:
                continue
            bank_path, row = Path(key[2]), int(key[3])
            if str(bank_path) not in bank_cache:
                bank = torch.load(bank_path, map_location='cpu', weights_only=False)
                tensors = bank['tensors']
                labels = load_sidecar(bank_path, int(tensors['track_id'].numel()))
                bank_cache[str(bank_path)] = (tensors, labels)
            tensors, labels = bank_cache[str(bank_path)]
            row_info[key] = {'gt': labels[row], 'pool': int(tensors['pool_id'][row]),
                             'features': {name: tensors[name][row].float().numpy()
                                          for name in ('clip', 'history_clip', 'uidm_h', 'pbd',
                                                       'uidm_ref_pbd', 'uidm_anchor_pbd')
                                          if name in tensors}}
    by_domain = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in records:
        keys = [tuple(k) for k in r['row_keys']]
        for left_i in range(len(keys)):
            for right_i in range(left_i + 1, len(keys)):
                left, right = row_info[keys[left_i]], row_info[keys[right_i]]
                if left['gt'] is None or right['gt'] is None:
                    continue
                if left['gt'] == right['gt']:
                    bucket = 'same'
                else:
                    bucket = 'different'
                for name in left['features']:
                    if name in right['features']:
                        by_domain[r['dataset']][name][bucket].append(cosine(left['features'][name], right['features'][name]))
        # Cross-frame same-GT pairs are formed later from separate units.
    # Rebuild same-GT cross-frame and cross-fragment pairs by selected rows.
    by_video = defaultdict(list)
    for r in records:
        by_video[(r['dataset'], r['video'])].append(r)
    sequence = defaultdict(lambda: {'all': [0, 0], 'cross_fragment': [0, 0], 'inactive_false_continuity': [0, 0]})
    for (dataset, video), group in by_video.items():
        group = sorted(group, key=lambda r: (int(r['frame_id']), int(r['fixed_order']) if 'fixed_order' in r else r['unit_key']))
        for index, current in enumerate(group):
            prior = None
            for candidate in reversed(group[:index]):
                if int(candidate['frame_id']) < int(current['frame_id']):
                    prior = candidate
                    break
            if prior is None:
                continue
            cur_keys = [tuple(k) for k in current['row_keys']]
            prev_keys = [tuple(k) for k in prior['row_keys']]
            cur_targets = {row_info[k]['gt'] for k, positive in zip(cur_keys, current['label']) if positive and row_info[k]['gt'] is not None}
            prev_targets = {row_info[k]['gt'] for k, positive in zip(prev_keys, prior['label']) if positive and row_info[k]['gt'] is not None}
            for gt in sorted(cur_targets & prev_targets):
                prev_rows = [row_info[k] for k, positive in zip(prev_keys, prior['label']) if positive and row_info[k]['gt'] == gt]
                cur_rows = [row_info[k] for k in cur_keys]
                for name in prev_rows[0]['features']:
                    if name not in cur_rows[0]['features']:
                        continue
                    prototype = np.mean([x['features'][name] for x in prev_rows], axis=0)
                    scores = [cosine(prototype, x['features'][name]) for x in cur_rows]
                    best = int(np.argmax(scores))
                    bucket = sequence[(dataset, video, name)]
                    bucket['all'][1] += 1
                    bucket['all'][0] += int(cur_rows[best]['gt'] == gt)
                    if any(x['pool'] != y['pool'] for x in prev_rows for y in cur_rows if y['gt'] == gt):
                        bucket['cross_fragment'][1] += 1
                        bucket['cross_fragment'][0] += int(cur_rows[best]['gt'] == gt)
            for gt in sorted(prev_targets - cur_targets):
                prev_rows = [row_info[k] for k, positive in zip(prev_keys, prior['label']) if positive and row_info[k]['gt'] == gt]
                cur_rows = [row_info[k] for k in cur_keys]
                if not prev_rows or not cur_rows:
                    continue
                for name in prev_rows[0]['features']:
                    if name not in cur_rows[0]['features']:
                        continue
                    prototype = np.mean([x['features'][name] for x in prev_rows], axis=0)
                    best = int(np.argmax([cosine(prototype, x['features'][name]) for x in cur_rows]))
                    bucket = sequence[(dataset, video, name)]
                    bucket['inactive_false_continuity'][1] += 1
                    bucket['inactive_false_continuity'][0] += int(cur_rows[best]['gt'] is not None)
    identity_by_domain = {}
    for dataset, fields in by_domain.items():
        identity_by_domain[dataset] = {}
        for name, groups in fields.items():
            # Add cross-frame pairs from all distinct selected units in this domain.
            same = list(groups['same'])
            different = list(groups['different'])
            domain_rows = [r for r in records if r['dataset'] == dataset]
            for left_r in domain_rows:
                for right_r in domain_rows:
                    if (left_r['video'], left_r['unit_key']) >= (right_r['video'], right_r['unit_key']) or left_r['video'] != right_r['video']:
                        continue
                    for lk, lpos in zip(left_r['row_keys'], left_r['label']):
                        for rk, rpos in zip(right_r['row_keys'], right_r['label']):
                            left, right = row_info[tuple(lk)], row_info[tuple(rk)]
                            if not (lpos and rpos and left['gt'] is not None and left['gt'] == right['gt']):
                                continue
                            if name in right['features']:
                                same.append(cosine(left['features'][name], right['features'][name]))
                                if left['pool'] != right['pool']:
                                    groups['cross_fragment'].append(cosine(left['features'][name], right['features'][name]))
            identity_by_domain[dataset][name] = pair_group_summary(same, different)
            identity_by_domain[dataset][name]['cross_fragment_same_gt'] = {
                'count': len(groups['cross_fragment']),
                'mean': float(np.mean(groups['cross_fragment'])) if groups['cross_fragment'] else None,
                'finite_sample_note': 'fixed selected units only'}
    # Domain/category score summaries use only already persisted compact probe
    # scores; no labels enter feature construction or any selection.
    domain_category = {}
    from tools.audit_l63_oracle_ceiling import score_summary
    for dataset in ('refer_kitti_v1', 'refer_kitti_v2'):
        domain_category[dataset] = {}
        for category in ('positive', 'multi_positive', 'inactive', 'present_uncovered'):
            subset = [r for r in records if r['dataset'] == dataset and r['category'] == category]
            domain_category[dataset][category] = {name: score_summary(subset, name) for name in
                ('l29', 'roi_mean', 'roi_point_max', 'roi_level_mean_0', 'roi_level_mean_1', 'roi_level_mean_2', 'roi_level_mean_3') if subset}
    coverage = source['coverage_ceiling']
    floor = 0.7333333333333333 - 0.01
    domain_coverage = {}
    for dataset, cats in coverage['by_dataset_category'].items():
        target = sum(v['units'] for k, v in cats.items() if k != 'inactive')
        covered = sum(v['covered'] for k, v in cats.items() if k != 'inactive')
        domain_coverage[dataset] = {'target_frame_units': target, 'covered_target_units': covered,
                                    'coverage': covered / max(1, target),
                                    'necessary_floor_from_L29_validation': floor,
                                    'below_floor': covered / max(1, target) < floor,
                                    'present_uncovered_units': cats['present_uncovered']['units']}
    # The automatic branch is intentionally conservative: report domain
    # coverage risk separately, then require the fixed ROI probe to show
    # stable hard/multi-positive separation before calling the decoder missing.
    region = source['expression_region_ceiling']['methods']['roi_point_max']
    region_weak = (region['hard_violation'] is None or region['hard_violation'] > .80) and \
                  (region['top1'] < .50) and \
                  (region['multi_positive_set_recall_at_positive_count']['mean'] < .50)
    identity_strength = max(source['identity_ceiling'][x]['all_same_gt_vs_same_frame_different_gt']['roc_auc'] or 0
                            for x in ('clip', 'history_clip'))
    aggregate_decision = 'representation_semantic_ceiling_insufficient' if region_weak and identity_strength >= .70 else \
                         'identity_oracle_insufficient' if identity_strength < .65 else \
                         'representation_has_ceiling_but_decoder_missing'
    domain_decisions = {}
    for dataset in ('refer_kitti_v1', 'refer_kitti_v2'):
        domain_decisions[dataset] = {
            'candidate_coverage': 'candidate_coverage_blocked' if domain_coverage[dataset]['below_floor'] else 'not_primary',
            'semantic': aggregate_decision,
            'note': 'domain result retained; aggregate cannot hide V2 coverage risk'}
    source['identity_ceiling_by_dataset'] = identity_by_domain
    source['sequence_identity_ceiling'] = {
        'protocol': 'frozen feature prototype retrieval between nearest selected frames; diagnostic only, no deployed DP/Viterbi',
        'by_dataset_video_feature': {f'{d}|{v}|{name}': {
            'continuation_recall': vals['all'][0] / max(1, vals['all'][1]),
            'continuation_correct': vals['all'][0], 'continuation_attempts': vals['all'][1],
            'cross_fragment_recall': vals['cross_fragment'][0] / max(1, vals['cross_fragment'][1]),
            'cross_fragment_correct': vals['cross_fragment'][0], 'cross_fragment_attempts': vals['cross_fragment'][1],
            'inactive_false_continuity': vals['inactive_false_continuity'][0] / max(1, vals['inactive_false_continuity'][1]),
            'inactive_checks': vals['inactive_false_continuity'][1],
            'finite_sample_note': 'selected 40 units, not a full sequence benchmark'}
            for (d, v, name), vals in sequence.items()},
        'gt_conditioned_candidate_ceiling': coverage['visible_target_coverage'],
        'multiple_positive_policy': 'all positive rows retained; no singletonization',
    }
    source['domain_category_expression_metrics'] = domain_category
    source['automatic_decision'] = {'aggregate': aggregate_decision,
                                    'coverage_floor': floor,
                                    'by_domain': domain_decisions,
                                    'identity_strength_reference_auc': identity_strength,
                                    'roi_probe_weak': region_weak,
                                    'rule_precedence': 'coverage risk is retained per domain; aggregate semantic decision requires frozen identity and ROI evidence'}
    source['status'] = 'complete'
    source['authoritative_output'] = str(OUT / 'oracle_ceiling.json')
    source['derived_from'] = str(SRC / 'oracle_ceiling.json')
    (OUT / 'oracle_ceiling.json').write_text(json.dumps(source, indent=2, default=str) + '\n')
    shutil.copy2(SRC / 'provenance.json', OUT / 'provenance.json')
    shutil.copy2(SRC / 'representative_hard_negatives.csv', OUT / 'representative_hard_negatives.csv')
    for p in (SRC / 'representative_hard_negatives').glob('*.svg'):
        (OUT / 'representative_hard_negatives').mkdir(exist_ok=True)
        shutil.copy2(p, OUT / 'representative_hard_negatives' / p.name)
    print(json.dumps({'status': 'complete', 'output': str(OUT / 'oracle_ceiling.json'),
                      'automatic_decision': aggregate_decision, 'domain_coverage': domain_coverage}, indent=2))


if __name__ == '__main__':
    main()
