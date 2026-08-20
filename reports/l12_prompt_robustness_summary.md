# Stage L12 — Prompt Robustness (Controlled Same-Target Comparison)

Date: 2026-08-20

Same videos, same objects, same frozen shared UIDM; only the seed
prompt type differs.  Operating point match-thr = -1.0 (balance between
coverage and switch rate):

| prompt | object-frames | persistence | switch | distinct IDs/object (mean) |
| --- | ---: | ---: | ---: | ---: |
| mask | 2,171 | 0.465 | 0.161 | 2.2 |
| box | 2,093 | 0.410 | 0.270 | 2.6 |
| point | 2,171 | 0.418 | 0.160 | 2.2 |

At match-thr = -2.0:

| prompt | persistence | switch |
| --- | ---: | ---: |
| mask | 0.487 | 0.207 |
| box | 0.467 | 0.268 |
| point | 0.462 | 0.217 |

Conclusions (frozen phase):

1. Identity dynamics IS prompt-modality-sensitive: mask/point seeds are
   consistently more robust than box-only seeds (lower switch, similar
   or better persistence).
2. Absolute persistence is modest (~0.41-0.49) with a frozen shared
   UIDM and DLA candidates; the seed token alone is not sufficient for
   reliable multi-object identity persistence.
3. This is a negative-to-mixed result for the frozen interface and the
   motivation for a joint prompt-adapter fine-tune (recorded in
   `l12_joint_training_summary.md`).
