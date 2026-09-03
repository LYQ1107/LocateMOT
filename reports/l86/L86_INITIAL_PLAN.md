# LocateMOT L86 Initial Plan

## Identity and source snapshot

- Project root: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`
- Luna thread: `01a02014-fce8-7f51-8414-e7ed6ab44745`
- Base: L85 commit `d54ffaa51a9c3e123fcd59fca0828a764a92ff3f`
- Working branch: `codex/l86-faithful-full-rmot-repair-20260903`
- Fixed manifest SHA256:
  `06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`
- Frozen semantic representation: L84 original-content `Z1`, fixed-reference
  GroundingDINO decoder layer 1. L84 layer selection and no-refPE comparisons
  will not be repeated.

L85 is the direct baseline. Its fixed semantic validation was recall
`.4193548`, precision `.1300`, FP/frame `3.625`, predictions/positive
`3.2258`, hard violation `.7692308`, multi-positive recall `.4861111`, and
inactive false acceptance `1.0`. Its internal full-video validation HOTA was
`25.0548` for V1 and `17.2924` for V2. L85 used the L69 budget-40 bank and
did not change ordinary MOT/OVMOT.

## Single L86 hypothesis

L86 tests whether L85's observed failure is caused by three implementation
errors rather than a missing Z1 signal: row-level duplicate-sensitive loss,
presence coupled to candidate energy/NULL, and a temporal gate trained with a
sigmoid output as logits against a history-length pseudo-target. The single
repair is a faithful unique-target-bag objective, independent candidate versus
presence/NULL energies, and causal temporal target-identity supervision.

No detector/representation change, LoRA, bank change, tracker change,
MLLM, motion-language pseudo-label, threshold repair, top-k, NMS, or candidate
deletion is permitted.

## Frozen inputs and model

The only candidate source is the query-independent L69 budget-40 feature and
dual-bank view. Rows are reconstructed by native `frame_ptr`/`frame_ids` and
all duplicates remain. The observation dimension is 1432:
`clip[512] + history_clip[512] + uidm_h[384] + geometry[7] + motion[8] +
lifecycle[8] + objectness[1]`. Z1, text summaries, L69 observations, and the
existing query-independent track IDs used only for causal history are frozen.

`L86FullRMOT` uses a shared semantic head for `r_static` and `r_total`, a
causal GRU history encoder with at most eight observations, a gated 256-D
state correction, a centered query-independent candidate prior, an independent
presence head, and an independent NULL head that also sees candidate evidence.
Candidate energy is `r_total + prior`; it does not include presence. The
forward contract will expose `r_static`, `temporal_state`, `r_total`,
`candidate_prior`, `candidate_energy`, `presence_logit`, `null_logit`,
`temporal_gate_logits`, `temporal_gate`, `temporal_delta`, and
`history_state`.

The loss is fixed at
`1.00 sem_total + .30 sem_static + 1.00 membership + .50 presence + .50
null + .10 temporal_id + .01 delta`. Target bags use `candidate_gt` and
`target_ids` only in the fit loss/evaluation attachment: max within each
unique target bag, background singleton negatives, all referred target bags,
and explicit `present_uncovered` masking. Temporal identity pairs use only
same-video/query/target positives across earlier frames and same-query
different-target negatives; target IDs never enter model tensors.

## Ordered execution

1. Record source/frozen-asset snapshot and preregistration.
2. Compile the new L86 namespace and run exactly one compact contract smoke:
   target-bag duplicate invariant, temporal-logit interface, and one real
   fit-only causal clip forward/backward with finite/nonzero gradients,
   future-frame count zero, and complete rows.
3. Run the requested GT-privileged semantic-oracle TrackEval for the legal
   internal validation videos, separately from learned results.
4. Train the joint V1/V2 model for exactly 40 epochs: S 1–8, T 9–20, J
   21–40. Use causal clips of length four where available, preserve complete
   sets, and use the actual available GPU world size/effective batch.
5. Score all epochs cheaply on the fixed video-disjoint internal dev split;
   run dev full-video HOTA only for epochs 8, 20, 40 and the best cheap
   shortlist (at most five), then freeze one checkpoint/rule.
6. Read the fixed 24 semantic validation units once after selection. Regardless
   of that gate, run full-video internal V1/V2 inference and local TrackEval.
7. Write the final decomposition and stop pending supervisor review.

## Pre-registered selection and gates

Cheap dev checkpoint selection uses, in order: lower target-bag hard
violation, higher target-bag hit@1, higher multi-target exact, lower inactive
false acceptance, higher candidate recall, earlier epoch. If dev full-video
HOTA is available, the selected emission rule is compared by higher HOTA,
then DetA, AssA, lower inactive false acceptance, and balanced-rule priority.
The threshold grid and Rule B/R/P definitions are fixed in
`outputs/l86/preregister/config.json`; fixed validation cannot change any
step, checkpoint, threshold, loss, or branch.

The semantic guardrail remains recall `>=.7233333`, precision `>=.0830189`,
FP/frame `<=11.125`, predictions/positive `<=4.069`, hard violation
`<=.8666667`, multi-positive recall `>=.7894444`, inactive false acceptance
`<1.0`, finite complete keys, and no deletion/truncation. This guardrail is
not a substitute for full-video HOTA.

The requested full-video scope is internal validation only: V1 videos
`0004,0018` (86 query sequences) and V2 videos `0016,0017,0020` (537 query
sequences). No screening or official-test labels will be read.

## Resource and provenance contract

Use at most GPUs 0–3, never GPU 4+, with one process per available GPU and
deterministic accumulation targeting effective clip batch 8. BF16 is enabled
only after the smoke; recurrent computation remains FP32 if needed. No raw or
dense cache is created, no frozen weights are copied, and no old asset is
overwritten. All outputs carry the required scope flags, including
`screening_gt_used=false`, `official_test_labels_read=false`,
`ordinary_mot_ovmot_touched=false`, `candidate_deletion=false`,
`candidate_truncation=false`, `z1_representation_changed=false`,
`groundingdino_lora_used=false`, `token_span_region_alignment=UNALIGNED`,
and `static_motion_alignment=UNALIGNED`.

If a technical error occurs, preserve its attempt, fix only the first
actionable cause, and run one targeted regression before resuming. A semantic
failure does not stop the authorized full-video HOTA sequence. After the
final report, stop at `STOPPED_PENDING_SUPERVISOR_REVIEW` and do not create an
automatic L86-R1.
