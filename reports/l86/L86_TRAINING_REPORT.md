# L86 training report

## Contract smoke

The authoritative smoke is
`outputs/l86/audit/contract_smoke_attempt6/contract.json`. Earlier attempts are
preserved with their first tracebacks. The smoke used a real fit-only anchor,
kept duplicate target-bag rows, verified zero future-history rows, complete
candidate sets, finite loss, nonzero adapter gradients, frozen input assets,
and strict checkpoint reload. It recorded model size `2,061,413` parameters
and output shape `Q=5, N=41`. The reload check was strict and shape-valid; the
recorded `0.217` output difference came from comparing a dropout-enabled
training forward with evaluation mode and is not claimed as an exact-value
reload invariant.

## Registered full fit

The blocking 40-epoch run is
`outputs/l86/train/joint40/`. It completed all S/T/J phases with actual GPU
world size 3 (`CUDA_VISIBLE_DEVICES=0,2,3`) and effective clip batch 9 under
the registered world-size accumulation rule. The final checkpoint is
`checkpoint_l86_step40epoch.pt`, step 2,360, SHA256
`a87b076692798020857e86cb7d291103b84cb33c0355b8e2325f78ac09423552`.
Epoch checkpoints 2 through 40 are present and reloadable.

The run had 40 finite epoch summaries and 21,000 global group updates (525 per
epoch); every epoch saw both V1/V2 domains and the four registered categories.
Selected loss means were:

| epoch | phase | loss mean |
|---:|:---:|---:|
| 1 | S | 4.027949 |
| 8 | S | 3.335923 |
| 20 | T | 2.942559 |
| 40 | J | 2.625154 |

At epoch 40 the local sampling trace recorded V1=82 and V2=93 groups,
inactive=122, multi-positive=188, positive=199, present-uncovered=88,
2,440 positive rows, 58,090 negative target bags, and 124 temporal identity
pairs. The data contract retained all current rows with no deletion or
truncation.

Training provenance records `seed=20260829`, `groundingdino_lora_used=false`,
`z1_representation_changed=false`, `screening_gt_used=false`,
`official_test_labels_read=false`, and `ordinary_mot_ovmot_touched=false`.
The local cache was the compact, label-free L85 Z1 cache; no new raw/dense
cache or detector weights were written.
