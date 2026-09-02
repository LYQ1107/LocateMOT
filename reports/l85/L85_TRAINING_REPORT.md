# L85 training report

## Scope and frozen inputs

This is an RMOT-only sidecar for Luna thread `01a02014-fce8-7f51-8414-e7ed6ab44745`.
It uses the query-independent L69 budget-40 bank and the label-free compact Z1
cache at
`outputs/l85/features/fit_dev_eval_full_attempt2/`.  The cache contains 1,623
complete groups and 115,651,514 bytes; it contains no labels, raw pixels, or
dense detector maps.  The fixed L19 manifest SHA remains
`06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`.

## Contracts before training

- `outputs/l85/audit/protocol/protocol.json`: all 17 registered L69 videos,
  native frame pointers, finite bank fields, and frozen-source checks passed.
- `outputs/l85/audit/candidate_oracle/oracle.json`: internal validation oracle
  only; unit coverage `0.8385300668151447`, target micro coverage
  `0.8792302587923025`.
- `outputs/l85/audit/memory_contract_attempt1/`: maximum candidate count 82;
  query tile 32 was selected by the label-free memory rule.
- `outputs/l85/audit/forward_loss_contract_attempt8/contract.json`: finite
  forward/loss, positive/negative/minimum-positive gradient, causal history,
  strict reload, and no deletion/truncation passed.  The compact model has
  1,795,174 parameters.

## Smoke and DDP checks

The initial corrected single-GPU smoke is preserved at
`outputs/l85/train/smoke100_retry3/metrics_l85_step100.json`: 100/100 finite
steps, 100/100 nonzero-gradient steps, both domains and all four registered
strata, detector-side parameters absent/frozen, strict reload, and bounded
memory.  The four-GPU one-step contract at
`outputs/l85/train/ddp_contract_step1/` also passed.  It is a contract check,
not the full training result.

The first full-cache attempt was stopped and retained in
`outputs/l85/features/fit_dev_eval_full_attempt1/INCOMPLETE.md` because it
reconstructed the frozen detector runtime for every group.  Runtime reuse was
then added and the complete attempt2 cache was generated.  No old evidence was
overwritten.

## Registered 40-epoch run

The authoritative run is
`outputs/l85/train/joint_curriculum40_gpu0/` with curriculum S=8, T=12, J=20
(40 epochs total), seed `20260829`, FP32 compact adapter and the registered
loss weights.  It completed 20,960/20,960 finite steps and
20,960/20,960 nonzero-gradient steps.  Every epoch checkpoint was written and
strict reload passed; the model has 1,795,174 parameters and the measured peak
resident GPU allocation was 76,101,120 bytes for the compact training graph.

The registered four-GPU world was not used for the long run because another
process occupied GPU1 when resources were checked.  The long run therefore
used one process on GPU0, which is a resource execution deviation, not a
hidden multi-GPU claim.  The independent four-GPU one-step DDP contract still
passed.  No full detector checkpoint was copied, no raw/dense cache was
created, and ordinary MOT/OVMOT files were not touched.

The implementation qualification is material: the trainer's S/T/J schedule
uses one current observation in S and the last four causal observations in T
and J for each row; it does not build an explicit batched clip-4 tensor. The
history encoder still enforces the registered maximum of eight and rejects
future observations. Results are therefore described as a causal-history
curriculum, not as evidence of a separate full clip-batch temporal model.

The dev-selected checkpoint and frozen emission rule are reported separately
in `L85_DEV_HOTA_SELECTION.md`.  No validation labels were used to select a
step.  This report is fit/implementation evidence, not a semantic gate or a
HOTA result.

## Flags

`candidate_bank_gt_conditioned=false`, `candidate_bank_query_conditioned=false`,
`screening_gt_used=false`, `official_test_labels_read=false`,
`ordinary_mot_ovmot_touched=false`, `token_span_region_alignment=UNALIGNED`,
`static_motion_alignment=UNALIGNED`.
