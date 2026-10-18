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

The latest durable state observed before this audit is optimizer step 600,
9,571 rows consumed, epoch 0, with FP32 masters, Adam moments, scheduler and
all RNG state saved. Feature caching is disabled. This is a partial-read
budget, not a claim that all 43,533 rows were trained. The registered horizon
remains 1,200 updates; the held-out gate waits for its terminal receipt.

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
