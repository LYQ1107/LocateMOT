# L83 initial plan

This is the supervisor-approved Stage L83 branch, created from immutable L82
commit `75e9f9cd0482645c07a9f71ad4419b0c5f57132b` on
`codex/l83-faithful-targetbag-20260902`.

The experiment will first audit the five L82 protocol mismatches, then run
target-bag layout/loss/metric tests, corrected read-only rescoring of L81/L59/
L82 controls, and one video-disjoint frozen-representation probe.  Target bags
use `max` over rows sharing one non-null `candidate_gt`; background rows remain
singleton negative bags.  Query-axis supervision uses only exact same-frame
real-label flips.  The independent ROC-AUC implementation is rank-based and
is not pair concordance.

The only new trainable probe is the same 66,561-parameter representation head
used by L82, with the new faithful loss and corrected target-bag metrics.  It
uses 524 fit groups for training and 138 video-disjoint fit-derived dev groups,
seed `20260829`, ten epochs, and compares L81 candidate evidence, L59 fused
ROI, and L82 candidate-reference features.  No fixed historical 16/24 labels
are read unless the faithful rank gate passes and all later gates are frozen.

Before any experiment, the branch records immutable asset hashes and an AST
audit of L82.  A source mismatch, target-bag data/metric contract failure, or
non-finite/reload failure stops the stage after one minimal targeted repair.
The faithful gate is fixed in
`reports/l83/L83_PREREGISTERED_PLAN.md`: G1 corrected bag-hard improvement,
G2 bag hit@1, G3 multi-target exact coverage, G4 query-swap accuracy, G5 V2
hard improvement, and G6 complete finite rows without deletion/truncation.
The decoder sharpness decomposition is mandatory for every representation
selection outcome, including a faithful-gate failure; it is a diagnostic and
does not authorize a primary semantic branch. Factorized energy and large
task-composition training are conditional and not pre-authorized by a smoke
alone.

All L82/L81/L59 banks, checkpoints, manifests, GT, ordinary MOT/OVMOT paths,
UIDM and TrackEval remain read-only.  Features remain process-local; no raw or
dense cache is written.  Token/span-to-region and static/motion alignment are
`UNALIGNED`.
