"""Offline train/screening feature-distribution audit for L24 F7."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT')
sys.path.insert(0, str(ROOT))
from tools.train_l23_dense_correspondence import fixed_refs  # noqa: E402
from tools.train_rmot_candidate_scorer import load_bank, load_metadata, make_refs  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='outputs/l19/protocol/kitti_fast_eval_manifest.json')
    ap.add_argument('--v3-root', default='outputs/l23/candidate_bank_v3')
    ap.add_argument('--out-root', default='outputs/l24/fallback/F7_domain_shift')
    ap.add_argument('--frames-per-split', type=int, default=6000)
    ap.add_argument('--seed', type=int, default=17)
    args = ap.parse_args()

    def p(x):
        x = Path(x)
        return x if x.is_absolute() else ROOT / x

    manifest, root, out = map(p, (args.manifest, args.v3_root, args.out_root))
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    queries = sorted(json.loads(manifest.read_text())['queries'], key=lambda x: int(x['query_index']))
    metadata = load_metadata()
    vids = sorted({str(x['video']) for x in queries})
    banks = {v: load_bank(root / 'kitti' / f'{v}.pt') for v in vids}
    refs = make_refs(queries, metadata, banks)
    groups = {
        'calibration': fixed_refs([r for r in refs if r['split'] == 'calibration'], args.frames_per_split, args.seed),
        'screening': fixed_refs([r for r in refs if r['split'] == 'screening'], args.frames_per_split, args.seed + 1),
    }
    fields = {
        'dense_roi': ('dense_roi',),
        'dense_points': ('dense_points',),
        'context': ('dense_context_1p5', 'dense_context_3'),
        'geometry': ('geometry_v2',),
        'motion': ('motion_v2',),
        'objectness': ('objectness',),
    }
    report = {
        'format': 'locatemot-l24-domain-shift-v1',
        'manifest': str(manifest), 'manifest_sha256': sha(manifest),
        'v3_root': str(root), 'screening_gt_used_for_selection': False,
        'frame_units': {k: len(v) for k, v in groups.items()}, 'features': {},
    }
    for name, keys in fields.items():
        report['features'][name] = {}
        for split, split_refs in groups.items():
            vals, labels = [], []
            for ref in split_refs:
                b = banks[ref['video']]['tensors']
                sl = slice(ref['begin'], ref['end'])
                pieces = [b[k][sl].float().numpy().reshape(b[k][sl].shape[0], -1) for k in keys]
                vals.append(np.concatenate(pieces, axis=1))
                labels.append(ref['positive'].astype(bool))
            x = np.concatenate(vals)
            y = np.concatenate(labels)
            report['features'][name][split] = {
                'count': int(len(x)), 'mean': float(x.mean()), 'std': float(x.std()),
                'quantiles': np.quantile(x, [.01, .25, .5, .75, .99]).tolist(),
                'positive_rate': float(y.mean()),
            }
        cal = report['features'][name]['calibration']
        scr = report['features'][name]['screening']
        report['features'][name]['mean_shift'] = scr['mean'] - cal['mean']
        report['features'][name]['std_ratio'] = scr['std'] / max(1e-8, cal['std'])
    shifts = [abs(v['mean_shift']) / (abs(v['calibration']['mean']) + 1e-6) for v in report['features'].values()]
    report['conclusion'] = {
        'large_relative_mean_shift': bool(max(shifts) > .25),
        'interpretation': 'descriptive audit only; no model or threshold was selected from screening labels',
    }
    (out / 'domain_shift.json').write_text(json.dumps(report, indent=2) + '\n')
    (out / 'domain_shift.md').write_text('# L24 F7 domain-shift audit\n\n' + json.dumps(report['conclusion'], indent=2) + '\n')
    print(json.dumps({'output': str(out / 'domain_shift.json'), 'conclusion': report['conclusion']}, indent=2))


if __name__ == '__main__':
    main()
