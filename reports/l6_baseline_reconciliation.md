# Stage L6 Baseline Reconciliation

Date: 2026-08-11

Goal: fix the *exact* identity of every historical baseline so that UIDM is
compared against the original project baseline **and** the strongest
reasonable baseline, never against a cherry-picked weak one.

## 1. Method identity mapping (resolved from code + reports)

| ID | Method | Implementation | Source |
|---|---|---|---|
| B0 | IoU Hungarian (native) | `OnlineTracker._associate_t0`, IoU + threshold 0.3 | L1-C audit `reports/l1_c_baseline_mapping_audit.md` |
| B1 | Motion C1 (OC-SORT style: 7-D Kalman + OCM second pass) | `OnlineTracker._associate_t1` | same |
| B2 | L1DK (fixed weighted IoU+PBD+motion, weights 0.4/0.2/0.4, thr 0.25) | `OnlineTracker._associate_l1d`, `compute_affinity_features` | L1-D |
| B3 | L3 U0 (shared learned one-checkpoint associator) | `L1DAssociator` trained on all domains, no dataset ID | `reports/l3_u0_shared_baseline.md` |
| B4 | L5 Route A (GT-anchored temporal identity transformer + bounded residual) | `L5TemporalAssociator`, epoch 40 | `reports/STAGE_L5_FINAL_REPORT.md` |

Historical naming notes: earlier L1-C documents swapped C1/C2 names.  The
table above uses the *current code* as ground truth: C1 = motion, C2 = raw
PBD cosine.  Raw PBD (C2) is **not** a strong baseline
(Dance val AssA ≈ 0.155), so it is never used as the "appearance
baseline" for UIDM.

## 2. Verified historical numbers

### 2.1 DanceTrack val (association-controlled custom AC protocol, TrackEval)

| Method | HOTA | DetA | AssA | IDF1 | IDSW | Source |
|---|---:|---:|---:|---:|---:|---|
| B0 IoU | 0.608 | 0.947 | 0.390 | 0.529 | 3,554 | l1_c_association_results |
| B1 Motion C1 | 0.630 | 0.947 | 0.419 | 0.566 | 2,916 | l1_c_association_results (L3 表: 0.4193) |
| B2 L1DK | 0.628 | 0.947 | 0.417 | 0.566 | 2,588 | STAGE_L5 (U0 列 = L1DK frozen) |
| B3 U0 (L3) | 0.6283 | — | 0.4169 | 0.5694 | 2,588 | STAGE_L5 |
| B4 L5 Route A | 0.6293 | — | 0.4182 | 0.5647 | 2,558 | STAGE_L5 |

### 2.2 Full four-domain table (the numbers L6 must beat)

| Method | Dance AssA/IDF1/IDSW | BDD AssA/IDF1/IDSW | MOT17 AssA/IDF1/IDSW | MOT20 AssA/IDF1/IDSW |
|---|---:|---:|---:|---:|
| B3 U0 (L3) | 0.4169 / 0.5694 / 2588 | 0.2881 / 0.2923 / 11042 | 0.6050 / 0.5825 / 259 | 0.2950 / 0.4012 / 2406 |
| B4 L5 Route A ep40 | 0.4182 / 0.5647 / 2558 | 0.2951 / 0.2954 / 12399 | 0.5914 / 0.5834 / 279 | 0.2763 / 0.3800 / 2588 |

L3 also measured per-domain optimal baselines (Dance C1 0.4193; BDD L1DK
0.3292; MOT17 L1DK 0.6010; MOT20 C1 0.2869), which shows a per-domain
oracle mixture beats the single L1DK checkpoint — the *reason* UIDM must
learn cue reliability instead of fixing one formula.

## 3. Protocol notes (important for L6 comparisons)

1. All L6 core comparisons use **one fresh TrackEval run** per method on
   the same manifests/GT (Dance val, BDD train subset, MOT17 train,
   MOT20 train), with a fresh tracker tag and fresh TrackEval directory.
2. Historical L1-C numbers used the association-controlled protocol on a
   smaller eval set; L3/L5 numbers used the full protocol.  Where the two
   protocols disagree (DetA ≈ 0.947 everywhere), only AssA/IDF1/IDSW are
   compared across methods within the same table.
3. Macro = domain-equal average over Dance/BDD/MOT17/MOT20.

## 4. What this means for UIDM success criteria

- Minimum meaningful improvement: Macro AssA / Macro HOTA / Macro IDF1
  above B3 **and** above B1 where B1 is stronger per domain (Dance), with
  no domain collapse (in particular MOT17/MOT20 must not drop like
  Route A did).
- B4 (L5 Route A) is a *method ablation*, not the original baseline.
- UIDM may have more parameters; we report params/FPS/VRAM and run one
  smaller-UIDM comparison to show the gain is not purely capacity.
