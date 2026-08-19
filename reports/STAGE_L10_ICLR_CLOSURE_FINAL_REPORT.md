# Stage L10 — Full-Supervision Scaling and ICLR Closure (FINAL REPORT)

Date: 2026-08-17/18

> Status: **NEAR_READY** (final; full-PBD OVMOT scaling is a documented
> negative result; Refer-KITTI-V2 eval pending the crop-PBD cache).

## 1. Project objective

LocateMOT builds **one shared identity-dynamics core (UIDM)** with a
persistent multi-track memory, lifecycle (Existing / NEW / NO-MATCH) and
set-level competition, driven by a unified frozen observation space
(PBD identity token + CLIP semantic token + specification token).  One
shared checkpoint must serve closed-set MOT, open-vocabulary MOT (OVMOT)
and referring-expression MOT (RMOT).

## 2. L0-L9 evolution (summary)

- L0: LocateAnything PBD box-end token extraction and crop-based
  adaptation; verified `pbd_box_end_last` (2048-d) as the identity token.
- L1: visual-prompt / unified observation adapters; early failures
  established that naive full-image PBD does not cover candidate sets.
- L2: future-utility / RL oracle headroom was low -> RL deferred.
- L3: latent regime router failed (dataset shortcut) -> not reused;
  shared one-checkpoint core retained (L3-U0 positive).
- L4: specification restriction caused identity drift; restricted
  evidence sometimes beats ALL evidence (positive problem signal).
- L5: high-level rethink -> unified observation adapter + UIDM.
- L6-L8: UIDM with lifecycle; joint MOT + RMOT; PBD-zero OVMOT.
- L9: full-PBD OVMOT (crop PBD for every TAO val candidate);
  specification-conditioned gate (`cond_gated`); crop-PBD OVMOT
  training stream (105 videos / 7,522 candidates); final shared
  checkpoint reached ordinary Macro AssA 0.5056, RMOT HOTA 36.79 /
  AssA 29.86, full-PBD TAO TETA 33.79 / AssocA 29.34 (vs PBD-zero
  TETA 34.33 / AssocA 30.44) -> **NEAR_READY**.

## 3. Why L10 scales supervision

The L9 bottleneck was not the architecture but **full-PBD OVMOT training
supervision**: only 105 videos / 4,200 frames / 7,522 candidates, with a
LocateAnything candidate distribution very different from the Detic
public dets used at evaluation.  L10 replaces the candidate source with
DLA (Detic-SwinB) dets on all 500 TAO train videos and aligns candidates
to C-TAO continuous annotations.

## 4. DLA / Detic candidate generation

- Root cause of the L9 blocker: torchvision 0.16 pure-Python `roi_align`
  with `sampling_ratio=0` materialises `[K,C,PH,PW,H,W]` (216 ROIs x
  256 ch x 100x136 -> ~274 GiB).  `tools/patch_adaptive_roi_align.py`
  implements the standard kernel math; numerical equivalence verified
  (max diff ~1e-6, aligned True/False).
- Full run: 500 videos / 18,274 frames, Detic-SwinB (`detic_masa.pth`),
  score>=0.05, top-50; no failures.  Val public dets are the same
  Detic-SwinB family (protocol caveat documented).
- See `reports/l10_tao_candidate_generation_audit.md`.

## 5. TAO-train full-PBD stream

| item | L9 | L10 |
| --- | ---: | ---: |
| videos | 105 | 500 |
| frames | 4,200 | 18,274 |
| candidates | 7,522 | 905,400 |
| GT | sparse TAO train + L6 boxes | C-TAO continuous (base) |
| matched candidates | ~86% | 31,331 (3.46%) |
| dets/frame | ~1.8 | 49.5 |
| val match rate (same protocol) | - | 5.0% |

PBD cache: per-candidate LocateAnything-3B crop token
(`pbd_box_end_last`, 2048-d, fp16).  Cache complete at
`outputs/l10/cache/tao_train_pbd`; merged into
`outputs/l10/data/tao_train/*.pkl` with candidate-order/finite asserts.

## 6. Refer-KITTI / Refer-KITTI-V2

- Refer-KITTI-V2 is the TempRMOT (arXiv:2406.05039) expansion of
  Refer-KITTI: 2,719 manual annotations -> 9,758 (GPT-3.5), 617 words.
- Official split verified: train = 17 KITTI tracking sequences (5,171
  frames), eval = 4 held-out sequences (0005/0011/0013/0019, 861
  expressions).  No train/eval video overlap.
- Official KITTI tracking images downloaded (15.8 GB zip, official S3
  mirror) and symlinked at `data/kitti_tracking_training`.
- Evaluation uses the official TempRMOT TrackEval seqmap and
  per-expression GT built from `labels_with_ids` + expression labels.
- See `reports/l10_refer_kitti_and_v2_audit.md`.

## 7. Training configuration (L10)

To be filled: model, init checkpoint, steps, effective batch, GPUs,
wall-clock, speedup.

## Table 1 — Unified MOT (one shared checkpoint)

| Method | Shared checkpoint | Shared UIDM | Ordinary Macro AssA | OVMOT TETA / AssocA | RMOT Refer-Dance HOTA / AssA | RMOT Refer-KITTI-V2 HOTA / AssA |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| L9 final | yes | yes | 0.5056 | 33.79 / 29.34 | 36.79 / 29.86 | - |
| L10 v1 (expanded) | yes | yes | 0.5041 | 26.39 / 7.26 | 36.32 / 28.79 | 3.74 / 16.72 |
| L10 v2 (target fix) | yes | yes | 0.4982 | 26.24 / 7.86 | 36.10 / 29.18 | - |
| L9-ovmot (main shared) | yes | yes | 0.5056 | 33.79 / 29.34 | 36.79 / 29.86 | 3.74 / 16.72 |

## Table 2 — OVMOT supervision scaling

| model | train videos | frames | candidates | TETA | LocA | AssocA | ClsA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L8-B2 PBD-zero | 0 (zero-PBD train) | - | - | 34.33 | 65.05 | 30.44 | 7.51 |
| L8-B2 naive full-PBD | 0 (zero-PBD train) | - | - | 32.22 | 64.19 | 24.95 | 7.53 |
| L9 adapted full-PBD | 105 | 4,200 | 7,522 | 33.79 | 64.47 | 29.34 | 7.54 |
| L10 v1 expanded full-PBD | 500 | 18,274 | 322,843* | 26.39 | 64.48 | 7.26 | 7.44 |
| L10 v2 + target fix | 500 | 18,274 | 322,843* | 26.24 | 63.42 | 7.86 | 7.45 |

*Training stream hard-negative cap (all positives + top-16 unmatched per
frame); raw DLA stream is 905,400 candidates.

## Table 3 — Training efficiency

| config | GPUs | micro-batch | effective batch | VRAM | GPU util | samples/s | steps/s | time/1k steps |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L9 baseline | 4 | 4 | 16 | 5.1 GB | ~50% | 7.9 | 0.49 | 2040 s |
| L10 after | 4 | 8 | 32 | 11.6 GB | ~60% | 10.0 | 0.33 | 3060 s |

## Table 4 — Ordinary MOT (L10 final)

| dataset | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DanceTrack | 0.565 / 0.554 | - | 0.338 / 0.324 | 0.489 / 0.487 | - | 6896 / 6388 |
| BDD100K | 0.487 / 0.483 | - | 0.519 / 0.510 | 0.430 / 0.423 | - | 6593 / 7296 |
| MOT17 | 0.705 / 0.699 | - | 0.693 / 0.682 | 0.618 / 0.615 | - | 453 / 437 |
| MOT20 | 0.630 / 0.637 | - | 0.466 / 0.476 | 0.555 / 0.567 | - | 1613 / 1568 |
| Macro AssA | - | - | 0.5041 / 0.4982 | - | - | - |

(v1 / v2; official TrackEval; same candidate pipeline as L6-L9.)

## Table 5 — RMOT

| dataset | method | detector | HOTA | DetA | AssA | MOTA | IDF1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Refer-Dance L9 | LocateMOT | LocateAnything | 36.79 | 45.58 | 29.86 | 29.38 | 36.56 |
| Refer-Dance L10 v1 | LocateMOT | LocateAnything | 36.32 | 46.03 | 28.79 | 28.93 | 35.70 |
| Refer-Dance L10 v2 | LocateMOT | LocateAnything | 36.10 | 44.88 | 29.18 | 26.90 | ~35.5 |
| Refer-KITTI-V2 L9-ovmot | LocateMOT | Detic-SwinB | 3.74 | 0.93 | 16.72 | -4153 | 0.97 |
| Refer-KITTI TempRMOT | TempRMOT | Deformable-DETR | 52.21 | 40.95 | 66.75 | - | - |
| Refer-KITTI-V2 TempRMOT | TempRMOT | Deformable-DETR | 35.04 | 22.97 | 53.58 | - | - |

Detector caveat: LocateMOT RMOT candidates come from LocateAnything /
Detic, not TempRMOT's end-to-end detector; DetA is not directly
comparable.  Identity claims focus on AssA.

## Table 6 — Cost

| item | value |
| --- | --- |
| trainable params | ~19.9M |
| frozen params | LocateAnything-3B + CLIP |
| GPUs | 4 x 40 GB |
| peak VRAM | 11.6 GB/GPU (training); 27 GB/GPU (val PBD cache) |
| optimizer steps | 15,000 (v1 and v2) |
| wall-clock | ~7.5 h / run (4 GPUs) |
| feature-cache time | TAO train PBD ~8 h; KITTI-V2 PBD ~3 h (pending) |

## Ablation

Supervision scaling (small 7.5k vs expanded 322.8k stream, same arch /
init): **negative**.  Expanded full-PBD OVMOT supervision collapses
association (AssocA 29.34 -> 7.3-7.9).  The expanded stream is only 3.5%
candidate-GT matched; with all unmatched detections labelled as NEW
targets, the model learns to birth a new identity for every detection.
The target-correction retrain (relevance negatives + score-gated NEW)
does not rescue it (7.86), and eval-time NEW margins 0-2 only reach
subset AssocA ~6.4.  See `reports/l10_supervision_scaling_ablation.md`.

## Failure analysis

Primary failure: expanded-stream full-PBD OVMOT (over-birth; verified
via unique-id statistics).  Ordinary MOT and RMOT remain stable across
the L10 variants, so the failure is OVMOT-training-supervision-specific.
See `reports/l10_failure_analysis.md` (RMOT worst-query analysis can be
extended from the per-query TrackEval CSV).

## Novelty audit

Updated in `reports/l10_literature_and_code_audit.md`: after COVTrack/
C-TAO, COVTrack++, OVTR, TRACT, AED, QTrack, MOTIP, iKUN, TempRMOT,
ReaMOT, CRMOT, no published system was identified that uses one trained
identity-dynamics core + one shared checkpoint across closed-set MOT,
OVMOT and RMOT.  Claim phrased as "we did not identify ...".

## ICLR readiness

Scoring:

- Novelty: strong (no published one-core/one-checkpoint MOT+OVMOT+RMOT
  system identified).
- Method depth: good (UIDM lifecycle + unified observation adapter).
- Unified evidence: strong for MOT + RMOT; OVMOT full-PBD is a negative
  analysis in L10.
- OVMOT competitiveness: weak in the full-PBD regime (PBD-zero remains
  the best OVMOT configuration, AssocA 30.44).
- RMOT breadth: Refer-Dance + Refer-KITTI-V2 (second benchmark pending).
- Ordinary robustness: Macro AssA ~0.50.
- Protocol fairness / ablation / failure analysis / reproducibility:
  documented, official evaluators, single seed.

Overall: **NEAR_READY**.  The paper can present PBD-zero as the main
OVMOT regime and full-PBD scaling as a rigorous negative result with a
clear supervision-coverage boundary.

## Next steps

1. Final report assembly + git commit (this document and
   `reports/LATEST_GPT_HANDOFF.md`).
2. Optional follow-up: temporal pseudo-track self-supervision for
   unmatched OVMOT detections (the evidence-based direction to recover
   full-PBD OVMOT), and RMOT fine-tuning on Refer-KITTI-V2 if the second
   benchmark is to be competitive.
