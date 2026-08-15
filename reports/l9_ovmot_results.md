# Stage L9 — TAO OVMOT results (official TETA)

Status: baselines below; full-PBD rows pending the TAO val PBD cache
(`outputs/l9/cache/tao_val_pbd`, in progress).

## Protocol

- Dataset: TAO val, official TETA evaluator (Base/Novel/All), Detic public
  dets (teta_50_internms), spec "all objects", one shared checkpoint per
  row.
- Observation:
  - L7/L8 rows: PBD-zero (CLIP + spec only; UIDM trained with
    PBD-dropout 0.15).
  - L9 rows: full observation — cached `pbd_box_end_last` per candidate
    + CLIP + spec.
- Tracker: OnlineTracker (UIDM variant), score threshold 0.05,
  L1D weights (0.4, 0.2, 0.4), threshold 0.25.

## Rows

| Method | Observation | Split | TETA | LocA | AssocA | ClsA |
|---|---|---|---|---|---|---|
| L7 CLIP probe (ref) | CLIP-only | All | 33.94 | — | 29.51 | 7.51 |
| L8-B2 (identity-pure) | PBD-zero | Base | 34.33 | 65.14 | 30.45 | 7.40 |
| L8-B2 (identity-pure) | PBD-zero | Novel | 34.36 | 64.41 | 30.40 | 8.27 |
| L8-B2 (identity-pure) | PBD-zero | All | 34.33 | 65.05 | 30.44 | 7.51 |
| L8-B1 (sem-in-core) | PBD-zero | All | 34.07 | 65.06 | 29.64 | 7.52 |
| L8-B2 | full PBD | Base / Novel / All | TBD | TBD | TBD | TBD |
| L8-B1 | full PBD | Base / Novel / All | TBD | TBD | TBD | TBD |
| L9 main (10k, MOT+RMOT) | full PBD | Base / Novel / All | TBD | TBD | TBD | TBD |
| L9 main (+OVMOT, planned) | full PBD | Base / Novel / All | TBD | TBD | TBD | TBD |

## Interpretation notes

- Full-PBD rows will answer whether adding per-candidate identity tokens
  improves association (AssocA) over the PBD-zero regime while keeping
  Base ≈ Novel (open-vocabulary generalization).
- External OVMOT comparisons (OVTR, OVTrack, TRACT) use different
  detectors/training; only same-protocol rows are directly comparable.

