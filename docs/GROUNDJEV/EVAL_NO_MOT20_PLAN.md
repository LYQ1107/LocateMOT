# G2 final / Simple / J1 / Coupled evaluation, 2026-10-26

Goal: `GOAL_G2_4GPU_NO_MOT20_DETECTION_SIMPLE_JEV_EVALUATION`.

This worktree is an explicitly authorized independent evaluation checkout of
the GroundJEV project, based on fetched `1cf48ce9`. The original GroundJEV
checkout, its uncommitted work, training sources and all checkpoints are preserved.

1. Audit G2 Clean step18227, J1 step1200 and the loadable coupled readout.
2. Register every locally available, legally evaluable, video-disjoint input.
   Exclude all MOT20. Check cross-dataset copies, including RMOT26.
3. Run four bounded single-GPU DLM services and persist resumable per-request
   detection records. Keep the official parser, decoder and missing scores.
4. Score detection and run the existing online Simple tracker from each
   completed video cache on CPU. Freeze the official result before JEV replay.
5. Replay immutable J1 and compatible coupled checkpoints on exactly the same
   candidates. Report unsupported semantics and negative results explicitly.
6. Aggregate with official sequence-combination rules, report measured timing,
   preserve failures, and publish code and small reports at stage boundaries.

No training or checkpoint modification is authorized by this task. Preflight
results are functional measurements, not full validation results. A running
queue or a successful pilot does not complete this goal.

Initial leakage findings: every local DanceTrack val sequence appears in the
G2 RMOT26 training coverage. All labeled MOT17 training sequences were also
used by G2. These cannot establish independent generalization, even when a
historical half-val manifest calls them validation. SportsMOT's 45 val clips
have neither clip-ID nor source-video-ID overlap with G2 SportsMOT training.
