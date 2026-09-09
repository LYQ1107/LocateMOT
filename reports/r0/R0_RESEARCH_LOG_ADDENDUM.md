# R0 research-log addendum

This addendum is kept under the R0 allow-list because the R0 boundary contract
forbids modifying the existing `research_log.md` during the isolated reset.

## 2026-09-09 — R0 Track-Centric Grounding Hook

- Hypothesis: multi-scale query-independent GroundingDINO observations plus
  causal track geometry and a pure-text pair/set head could break the L89E
  compressed-observation local optimum.
- Inputs/outputs: isolated worktree at base
  `7fa201e4967f71a42ba669e90f330a6e3979e1cd`; static source audit passed;
  dense data audit written to `outputs/r0/data/dense_contract_attempt1/`.
- Failure: 5,314 fit rows form 2,594 repeated query groups; 651 groups (V1
  422, V2 229) disagree in frame-level `target_ids`. The R0 contract requires
  repeated rows to agree, so no union/majority repair was allowed.
- Retained: all R0 code, source audit, failed dense-index attempt, reports,
  and boundary evidence. No detector/cache/training/TrackEval ran.
- Unique next action: supervisor-approved frame-specific visibility-label
  contract repair or explicit re-registration, then rerun only the dense data
  contract audit before any visual cache or model run.
