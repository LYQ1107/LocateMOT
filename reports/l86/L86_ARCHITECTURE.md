# L86 architecture and faithful repair

L86 is an RMOT-only factorized repair on the frozen L84 Z1 representation and
the immutable L69 budget-40 bank. It does not alter the GroundingDINO
checkpoint, proposal bank, UIDM, tracker, or ordinary MOT/OVMOT paths.

## Three repaired contracts

1. Semantic supervision is target-bag aware. Duplicate rows carrying the same
   `candidate_gt` are grouped by unique target ID and scored with the maximum
   row energy for that target. Background rows remain singleton negatives. A
   present-but-uncovered target is masked from candidate membership loss.
2. Candidate energy, frame presence, and candidate-evidence-aware NULL energy
   are independent outputs. Presence is not added to candidate energy, and the
   NULL head receives both presence input and a softmax-weighted candidate
   summary.
3. Temporal learning uses causal target-identity pairs from real earlier
   observations. It does not use history length as a pseudo-target for a
   sigmoid value passed to `BCEWithLogitsLoss`.

## Inputs and forbidden inputs

The observation vector is 1,432 dimensions:
`clip[512] + history_clip[512] + uidm_h[384] + geometry[7] + motion[8] +
lifecycle[8] + objectness[1]`. Z1 semantic input is 256-D; text/frame global
inputs are 256-D each. A causal history contains at most eight observations and
never reads a frame after the current cutoff. `track_id` is used only by the
frozen data index to assemble history; it is not passed as a semantic value.

The model never consumes `source`, `pool_id`, `group_id`, `query_id`,
`state_key`, `track_id`, L29/L70/L75 scores, proposal rank, screening labels or
official-test labels. All current L69 rows remain in every forward and output.

## Trainable module

`L86FullRMOT` has hidden size 256, history length 8, and 2,061,413 trainable
parameters. It exposes `r_static`, `temporal_state`, `r_total`,
`candidate_prior`, `candidate_energy`, `presence_logit`, `null_logit`,
`temporal_gate_logits`, `temporal_gate`, `temporal_delta`, and `history_state`.
The prior is a centered query-independent auxiliary energy in
`[-0.5, 0.5]`; it is not a proposal filter. Candidate energy is
`r_total + candidate_prior`, without presence coupling.

Token/span-to-region and static/motion language alignment are not verified and
remain `UNALIGNED`.

## Objective and curriculum

The fixed total objective is

```text
1.00 semantic_target_bag(r_total)
+ 0.30 semantic_target_bag(r_static)
+ 1.00 membership
+ 0.50 presence
+ 0.50 candidate-evidence-aware NULL
+ 0.10 causal target-identity temporal loss
+ 0.01 temporal-delta L2 regularization
```

All losses are grouped within video/query/frame or video/query/target. All
positive target bags, including multi-positive rows and main/reserve fragments,
receive supervision. Same-class metadata is unavailable, so hard-negative
training uses the registered unique-target-bag all-negative fallback.

The run used S epochs 1–8, T epochs 9–20, and J epochs 21–40, seed `20260829`,
AdamW learning rate `2e-4`, weight decay `1e-2`, five-percent warmup, cosine
decay, gradient clipping 1.0, and BF16 after the contract smoke. Recurrent
history computation was kept FP32 where required.

## Source files

The implementation is isolated in:

- `locatemot/models/l86_full_rmot.py`
- `locatemot/rmot/l86_temporal.py`
- `locatemot/rmot/l86_losses.py`
- `locatemot/rmot/l86_clip_data.py`
- `tools/l86_train_full_rmot.py`

The fixed representation/cache and L69 bank are read-only inputs. No raw or
dense feature cache was created.
