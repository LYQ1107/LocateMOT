# Stage L11 — OVMOT Repair Results

Date: 2026-08-19 (live; updated as checkpoints complete)

## Baseline (L9/L10)

| model | TAO full-PBD TETA | AssocA | note |
| --- | ---: | ---: | ---: |
| L8-B2 PBD-zero | 34.33 | 30.44 | best known |
| L9-ovmot adapted full-PBD | 33.79 | 29.34 | main shared ckpt |
| L10 v1 expanded | 26.39 | 7.26 | unmatched -> NEW collapse |
| L10 v2 target fix | 26.24 | 7.86 | still collapsed |

L9-ovmot over-birth reference (10 TAO val videos, quick eval):
18,260 prediction rows, 3,854 unique IDs, unique/rows = 0.211,
467 reused IDs, mean track length 4.74.

## L11 repair training (in progress)

Run: resume L9-ovmot, 4 GPUs (2,7,8,9), batch 4/GPU, 20k-step budget,
streams MOT + OVMOT(GT@0.3 base_and_novel + pseudo-track) + RMOT
(Refer-Dance).  Checkpoints: `outputs/l11/checkpoints/uidm_l11_main/`.

### Step 7000 intermediate (1000 steps)

10-video quick eval (same subset as baseline):

| quantity | L9-ovmot | L11 step7000 |
| --- | ---: | ---: |
| prediction rows | 18,260 | 18,260 |
| unique IDs | 3,854 | 5,444 |
| unique/rows | 0.211 | 0.298 |
| reused IDs | 467 | 672 |
| single-observation IDs | 2,616 (67.9%) | 2,954 (54.3%) |
| mean track length | 4.74 | 3.35 |

Reading: at 1000 steps the model has NOT yet reduced over-birth; NEW
rate is temporarily higher.  Evidence-driven correction prepared:
explicit NOT-NEW supervision (weight 0.3) for unmatched high-score
candidates without identity evidence (NEW tightening), plus continued
training to 8k-10k before judging.

### Step 8000 intermediate (2000 steps)

10-video quick eval (same subset):

| quantity | L9-ovmot | L11 s7000 | L11 s8000 |
| --- | ---: | ---: | ---: |
| prediction rows | 18,260 | 18,260 | 18,260 |
| unique IDs | 3,854 | 5,444 | 3,423 |
| unique/rows | 0.211 | 0.298 | 0.187 |
| single-observation IDs | 67.9% | 54.3% | 37.7% |
| mean track length | 4.74 | 3.35 | 5.33 |

Reading: over-birth is now BELOW the L9 baseline; identity reuse is
improving with more training.  Continue to 9k-10k, then run the full
TAO-val TETA.

## Key mechanisms (as implemented)

1. Class A: C-TAO base_and_novel GT at IoU >= 0.30 (~28% coverage).
2. Class B: 17,779 pseudo tracklets / 102,093 pseudo candidates
   (99.26% same-ID precision on the GT-covered subset).
3. Class C/D: unmatched candidates without pseudo id -> identity IGNORE
   and relevance 0; never NEW by default.
4. NEW tightening (step-8000 restart if needed): unmatched high-score
   candidates get explicit NOT-NEW with weight 0.3.
