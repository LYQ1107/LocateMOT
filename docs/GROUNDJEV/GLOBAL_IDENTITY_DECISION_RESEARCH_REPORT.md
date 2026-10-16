# Global Identity Decision research report

Status: R1 audited; R2 actual CPU A1/A2 training running. The full GroundJEV goal
and the new `GOAL_GLOBAL_IDENTITY_DECISION_V1` milestone remain incomplete.

## Completed evidence

The isolated worktree is `/data2/user/LocateMOT_GlobalIdentity`, branch
`codex/global-identity-decision-v1`. The source branch was independently checked
against GitHub `4881bbc` / local `d2d4c00`. Original Full-source, paired temporal,
paired RMOT native training, Ground retention and legacy native evaluation jobs
remain on their original code and four physical GPUs 4–7. New work uses CPU only.

All 43 MOT TRAIN units / 1,056 frames / 48,051 candidates were audited. The sparse
whole-frame graph covers all 4,804 known cross-group complete competition sets.
6,087 candidate labels remain UNKNOWN, not invented background. Source live-ID
coverage is 28,507 candidates, with 22,040 covered by the old Top-K. The versioned
[label/action contract](LABEL_AND_ACTION_CONTRACT.md) and compressed per-candidate
labels preserve the exact denominators, ambiguous support and original provenance.

## Registered controlled training

A0 is the existing Shared Compact reference, not a repeated known-negative8000
training run. A1/A2 start from the same Shared Compact parameters and identical
new-head seed0 initialization, same immutable1056-frame stream, four epochs,
AdamW1e-4/weight-decay1e-4/clip1 and CPU4threads each. A1 trains masked identity /
NEW / REJECT CE; A2 adds sparse competition messages, duplicate BCE and quality
ordering. Physical inputs contain no task/query text. All live IDs are scored in
blocks; the O(NM) allocation matrix and sparse O(NK) competition are distinct.
The frozen task-conditioned semantic head is kept separate during runtime.

The first A1 prototype emitted a real positive-gradient update, then failed on an
empty source frame because an empty loss was detached. Its log/directory remain.
The correction attaches a zero-valued empty-frame loss to the model graph, records
the frame and performs no fabricated optimizer update. The final A1 objective also
excludes the A2 duplicate/ranking losses; the failed prototype is not a result arm.
No original training source, label, seed or acceptance criterion was changed.

## Required results still pending

|Comparison|HOTA/AssA/IDF1/IDSW change|Status|
|---|---|---|
|A0 existing Shared Compact|Existing registered five-scope reference|Retained|
|A1 identity + NEW/REJECT|Unmeasured|Training|
|A2 global candidate competition|Unmeasured|Training|
|A3 assigned-ID memory|Unmeasured|Pending|
|Own-state8/16/32-frame training|Unmeasured|Pending|
|Parameter/input-matched non-JEV control|Unmeasured|Pending|

There is no claim yet of fewer false births, duplicate identities, wrong writes,
cross-task improvement, ordinary-model parity, or an independent JEV contribution.
All five original scopes and official scorers remain required; a development video
will precede the complete registered five-scope comparison. Same final physical
checkpoint across MOT/RMOT/OVMOT is mandatory. Novel ClsA remains an unresolved
independent problem. Replay latency is not detector-inclusive end-to-end FPS.
