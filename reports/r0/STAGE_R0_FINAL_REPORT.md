# R0 Track-Centric Grounding Hook — final stage report

## Executive decision

`R0_DATA_CONTRACT_INVALID` — `STOPPED_PENDING_SUPERVISOR_REVIEW`.

R0 stopped before visual cache creation and training because the first
preregistered dense-data invariant is false. The active implementation did not
repair the labels or proceed with an ambiguous interpretation.

## Verified evidence

- Project identity was verified in the original LocateMOT checkout before
  creating the isolated R0 worktree.
- Worktree: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_R0`.
- Branch: `codex/r0-track-centric-grounding-hook-20260909`.
- Base: `7fa201e4967f71a42ba669e90f330a6e3979e1cd`.
- Fixed manifest SHA: `06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`.
- Static frozen-source audit passed at
  `outputs/r0/audit/source_interfaces_retry1/audit.json`.
- A first wrong-path source audit is retained at
  `outputs/r0/audit/source_interfaces/INCOMPLETE.md`.
- The R0 data audit read 5,314 fit rows, 2,594 repeated query groups, and
  found 651 target-contract violations (V1 422, V2 229).
- First violation: V1 video 0001, query 0, whose repeated frame rows have
  target sets `45`, `57`, `57+58`, `84`, and `9` for one sentence.
- `py_compile` passed for all implemented R0 modules and tools, and the R0
  boundary guard passed. These are implementation checks only.

## Why the later stages were not run

The R0 prompt requires repeated fit rows for one `(dataset, video, query_id)`
to agree on `sentence + target_ids`, and explicitly forbids unioning or
reinterpreting target IDs. The observed L49 rows are frame-level visibility
labels. Building a visual cache, training V1/V2 heads, selecting checkpoints,
or running TrackEval before resolving that semantic mismatch would invalidate
the experiment. Therefore no visual cache, dense training index, checkpoint,
dev selection, fixed semantic diagnostic, screening, or TrackEval evidence
exists.

The registered R0 architecture files were added only as isolated, unexecuted
sidecar code: four-level 3x3 inner/context visual tokens, causal geometry,
pairwise text/visual correspondence, and complete current-frame set reasoning.
No claim about their quality or the R0 final labels is valid.

## Frozen boundaries

No L69/L89E bank, UIDM, tracker, ordinary MOT, OVMOT/TAO, production entry
point, fixed manifest, old checkpoint, or historical report was modified. No
screening or official-test labels were read. The complete machine status is in
`outputs/r0/R0_STATUS.json`.

## One next action

Obtain supervisor approval for a frame-specific target-visibility label
contract repair or explicit re-registration that preserves each frame's
`target_ids`, then rerun the R0 dense-index audit only. Do not start visual
cache or training until that new contract passes.
