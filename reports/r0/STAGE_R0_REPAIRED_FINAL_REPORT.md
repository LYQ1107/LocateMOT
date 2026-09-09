# R0A → R0 repaired execution — stopped at source boundary

## Status

`R0A_BLOCKED_L49_QUERY_SOURCE` / `STOPPED_PENDING_SUPERVISOR_REVIEW`.

R0A was authorized to repair the false canonical-target invariant, but the
first source-isolation audit found that the local V2 loader parses a complete
JSON containing official-eval records before filtering videos. R0A therefore
stopped before the 5,314-row audit, dense indexes, visual cache, training,
checkpoint selection, semantic diagnostics, and TrackEval.

## Verified facts

- R0 failed base: `19fcb5fd3dcf1d26adb60a5fcceb3121c6f58c5b`.
- R0A branch: `codex/r0a-frame-specific-target-contract-repair-20260909`.
- Local L49 source SHA256:
  `a1aae91cfbd8aadfba2e432302b06ff7b3e950fe486b48e1f5a4e80481819b3b`.
- V2 old source contains top-level videos `0005`, `0011`, `0013`, which are
  forbidden official-eval scope for this stage.
- `load_l49_queries` filters after parsing, so the source contract is not
  isolated before label loading.
- The fixed manifest remains the registered SHA
  `06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`.
- No model, detector, visual cache, training, TrackEval, HOTA, ordinary MOT,
  OVMOT, UIDM, tracker, or L69 asset was modified or executed by R0A.

The preliminary V2 loader inspection necessarily parsed the monolithic source;
this attempt is explicitly marked invalid with
`official_test_labels_read=true` in the audit JSON. No metric or label result
from this attempt is scientific evidence.

## R0A/R0 outputs

The source audit and stop marker are under
`outputs/r0/audit/r0a_l49_target_source/`. R0's historical
`reports/r0/STAGE_R0_FINAL_REPORT.md` remains unchanged. The R0 architecture
code was not evaluated after the new source blocker and no HOTA values are
available.

## Unique next action

Provide a supervisor-approved train-scope target source or isolation reader
that never opens official-eval records, then rerun only the R0A source audit.
Do not resume dense indexing or training before that audit passes.
