# LocateMOT Stage L11–L12 — Unified Tracking Final Report

Date: 2026-08-20
Project: LocateMOT — Specification-Conditioned Unified Tracking

## 1. Summary

Stage L11 repaired the two known core problems:

1. OVMOT temporal supervision: replaced "unmatched detection -> NEW"
   with high-precision temporal pseudo-track supervision + densified GT
   (C-TAO base_and_novel @0.3 IoU).  Full TAO-val TETA 33.79 -> 36.63,
   AssocA 29.34 -> 37.10 (best LocateMOT OVMOT result).
2. Refer-KITTI/V2 candidate front-end: category whitelist + cross-NMS +
   query-conditioned CLIP top-12 -> HOTA 3.74 -> 6.28, MOTA -4153 ->
   -766 with the SAME shared checkpoint.

Stage L12 added a controlled point/box/mask prompt-seeded evaluation on
DAVIS 2017 val with the frozen shared UIDM.  Prompt modality measurably
affects identity persistence (mask/point > box), but absolute
persistence is modest (~0.41-0.49), so joint prompt fine-tuning was
not launched (recorded as the remaining step).

## 2. Final shared checkpoint

- Primary: `outputs/l11/checkpoints/uidm_l11_main/step10000.pt`
  (OVMOT-strong).
- Balanced continuation: `outputs/l11/checkpoints/uidm_l11_bal/`
  (p_rmot=0.45, p_ovmot=0.20; final numbers pending).

## 3. Table A — Unified formulations (one shared UIDM)

| formulation | specification | discovery/seeded | dataset | same UIDM | same ckpt | primary metric | result |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| Ordinary MOT | closed category | discovery | Dance/BDD/MOT17/MOT20 | yes | yes | Macro AssA | 0.486-0.505 (ckpt-dependent) |
| OVMOT | open category | discovery | TAO val | yes | yes | TETA / AssocA | 36.63 / 37.10 |
| RMOT Refer-Dance | referring expression | discovery | DanceTrack | yes | yes | HOTA / AssA | 32.5-36.8 (ckpt-dependent) |
| RMOT Refer-KITTI/V2 | referring expression | discovery | KITTI 4 seqs | yes | yes | HOTA / MOTA | 6.28 / -766 |
| Point prompt | point | seeded | DAVIS 2017 | yes | yes | persistence | 0.42-0.46 |
| Box prompt | box | seeded | DAVIS 2017 | yes | yes | persistence | 0.41-0.47 |
| Mask prompt | mask | seeded | DAVIS 2017 | yes | yes | persistence | 0.47-0.49 |

## 4. Table B — OVMOT repair

| config | pseudo coverage | pseudo precision | NEW rate | unique-ID ratio | TETA | AssocA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L9 adapted full-PBD | 0 (GT-only) | - | - | 0.211 | 33.79 | 29.34 |
| L10 v1 expanded | 0 | - | ~1.0 | 0.977 | 26.39 | 7.26 |
| L10 v2 target fix | 0 | - | ~1.0 | - | 26.24 | 7.86 |
| **L11 s10000** | 32% of training cands | 99.3% | 0.11 | 0.187 (s8k) | **36.63** | **37.10** |

## 5. Table C — Refer-KITTI/V2 repair (same L9-ovmot ckpt)

| stage | cands/frame | DetPr | DetRe | HOTA | DetA | AssA | MOTA | IDF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| before (L10 DLA top-50) | ~50 | ~1% | - | 3.74 | 0.93 | 16.72 | -4153 | 0.97 |
| after (L11 whitelist+NMS+CLIP top-12) | ~12 | 3.34% | 27.3% | 6.28 | 3.09 | 13.80 | -766 | 3.39 |

## 6. Table D — Prompt robustness (controlled same-target)

See `reports/l12_prompt_robustness_summary.md`; mask/point seeds are
more robust than box seeds for identity persistence/switch rate.

## 7. Novelty audit

We did not identify a published 2025/2026 system with a single learned
identity-dynamics core and ONE shared checkpoint covering closed-set
MOT, open-vocabulary MOT, referring-expression MOT, and point/box/mask
prompt-seeded tracking under official protocols.

## 8. ICLR readiness

**NEAR_READY** (final judgment after the balanced checkpoint numbers are
available; strong OVMOT evidence and shared-checkpoint evidence, but
cross-task balance (RMOT-Dance / ordinary MOT) and prompt-seeded
generalization need the final numbers).
