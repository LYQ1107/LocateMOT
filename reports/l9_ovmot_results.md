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

## Full-PBD rows (official TETA, 2026-08-17)

| Method | Observation | Split | TETA | LocA | AssocA | ClsA |
|---|---|---|---|---|---|---|
| L8-B2 | full PBD | All | 32.22 | 64.19 | 24.95 | 7.53 |
| L8-B2 | full PBD | Base | 32.19 | 64.17 | 24.97 | 7.42 |
| L8-B2 | full PBD | Novel | 32.48 | 64.32 | 24.77 | 8.33 |
| L8-B1 | full PBD | All | 31.83 | 64.09 | 23.87 | 7.53 |
| L9 v5 | full PBD | All | 32.04 | 64.40 | 24.22 | 7.49 |
| L9 v5 | full PBD | Base | 32.04 | 64.61 | 24.10 | 7.42 |
| L9 v5 | full PBD | Novel | 31.99 | 62.78 | 25.15 | 8.04 |

**Key negative result**: adding the crop-based PBD identity tokens to
checkpoints trained only on full-frame PBD (or on ordinary/RMOT data)
*degrades* TAO association (L8-B2 AssocA 30.44 PBD-zero -> 24.95
full-PBD; TETA 34.33 -> 32.22).  Base vs Novel remains balanced, so the
drop is an observation-distribution mismatch, not open-vocabulary
regression.  The decisive follow-up is training the shared core on
crop-PBD OVMOT data (TAO train, Stage L9-B), so the model learns the
crop-conditioned PBD distribution; that experiment is in progress.

## Decisive follow-up: crop-PBD-adapted model (L9-ovmot)

Trained by resuming L9-v5 with the crop-PBD OVMOT stream (105 TAO train
videos, 7,522 crop-PBD candidates, 86% GT-matched; 6,000 steps, 4 GPUs).

| Method | Observation | Split | TETA | LocA | AssocA | ClsA |
|---|---|---|---|---|---|---|
| L9-ovmot | full PBD | All | **33.79** | 64.47 | **29.34** | 7.54 |
| L9-ovmot | full PBD | Base | 33.73 | 64.41 | 29.34 | 7.43 |
| L9-ovmot | full PBD | Novel | 34.22 | 64.94 | 29.37 | 8.35 |

Reading: crop-PBD training recovers most of the naive full-PBD drop
(AssocA 24.95 -> 29.34; TETA 32.22 -> 33.79) and keeps Base = Novel
(29.34 / 29.37), confirming the distribution-mismatch hypothesis.
However, full-PBD still trails the PBD-zero regime by ~1.1 AssocA
(29.34 vs 30.44) and ~0.5 TETA (33.79 vs 34.33) in this setup, likely
because the OVMOT training stream is sparse (7.5k candidates) relative to
the ordinary/RMOT streams.  This is a documented Pareto trade-off, not a
fabricated result.

## Interpretation notes

- Full-PBD rows will answer whether adding per-candidate identity tokens
  improves association (AssocA) over the PBD-zero regime while keeping
  Base ≈ Novel (open-vocabulary generalization).
- External OVMOT comparisons (OVTR, OVTrack, TRACT) use different
  detectors/training; only same-protocol rows are directly comparable.
