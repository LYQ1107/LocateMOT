# Stage L12 — Unified Ablation

Date: 2026-08-20

## L11 (OVMOT supervision)

| stream | pseudo-track | NEW policy | TAO full-PBD AssocA |
| --- | --- | --- | ---: |
| L9-ovmot (adapted) | no | unmatched high-score -> NEW | 29.34 |
| L10 v1 expanded | no | every unmatched -> NEW | 7.26 |
| L10 v2 target fix | no | gen>=0.4 unmatched -> NEW | 7.86 |
| **L11 s10000** | **yes (99.3% precision)** | **unmatched never NEW** | **37.10** |

The ablation isolates the temporal pseudo-track supervision + NEW
tightening as the cause of the +7.8 AssocA improvement.

## L12 (prompt modality)

Frozen shared UIDM, DAVIS 2017 val, 10 multi-object videos, seeded-only:

| prompt | persistence (thr -1) | switch (thr -1) | persistence (thr -2) | switch (thr -2) |
| --- | ---: | ---: | ---: | ---: |
| mask | 0.465 | 0.161 | 0.487 | 0.207 |
| box | 0.410 | 0.270 | 0.467 | 0.268 |
| point | 0.418 | 0.160 | 0.462 | 0.217 |

Joint fine-tuning was not run (frozen signal mixed; L11 balance
prioritized) — `l12_joint_training_summary.md`.
