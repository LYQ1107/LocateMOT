# J1/J2 training budget audit

## J1 current course

The active OneJev J1 course is
`trackjev_3task_J1_v4_birth` on physical GPU4/GPU6. It uses the real
43,533-row three-task multimodal corpus:

| Task family | Rows |
| --- | ---: |
| Identity/MOT | 21,168 |
| RMOT | 16,677 |
| OVMOT | 5,688 |

The latest durable state observed at the 2026-10-19 post-launch audit is
optimizer step 650, 10,368 rows consumed, epoch 0, with FP32 masters, Adam
moments, scheduler and all RNG state saved. Feature caching is disabled. This
is a partial-read budget, not a claim that all 43,533 rows were trained. The
registered horizon remains 1,200 updates; the held-out gate waits for its
terminal receipt.

## J2 registration rule

`J2_FULLDATA_IDENTITY_SFT` is a separate course after J1 completion and
independent evaluation. The candidate budget is 4,000--6,000 optimizer updates,
but the final number must be registered from the actual global batch, unique
video exposure, and replay/coverage statistics. J2 starts from pretrained
OneJev-0.8B plus the selected J1 state; it is not a from-scratch run.

J2 must add deployment-distribution supervision: real detector candidates,
global MATCH/NEW/REACTIVATE competition, assignment-conditioned memory writes,
RMOT native candidates, and OVMOT Base/Novel lawful supervision. GT-teacher
regions can remain an explicit control, but cannot be the only J2 candidate
source. No J2 quality claim or checkpoint promotion is made before the fixed
G0+J1, G1+J1, G2+J1, G2+J2 and projector-only comparison table is complete.

## Candidate-source gate audited 2026-10-19

The existing balanced `onejev_unified_matched600_v2/native/train.jsonl` was
stream-audited and rejected as the J2 candidate-driven manifest. It has 125,286
rows and SHA256
`136d029d212d831e1c72cf7c7fe538dc9b246e8edc0febb9bb14ac1abf662156`, but its
MOT portion includes `legal_training_GT_proposals_and_explicit_synthetic_fragments`
and its OVMOT portion is `registered train-Base GT teacher crops`. Only the
repeated 346-row RMOT component is sourced from native GroundAnything output.
It remains an explicit ablation artifact and is not launched as J2.

The corrected source inventory is retained at
`outputs/groundjev/training_data/j2_full_candidate_inventory_v1/inventory.json`:
131,214 native MOT rows (7 MOT17 sequences), 346 native RMOT rows from 72 fit
frames, and 430 OVMOT teacher-control rows. The J2 launch gate therefore still
requires the J1 final checkpoint plus native OVMOT candidate rows and broader
native MOT/RMOT coverage; no optimizer was started from the rejected manifest.
