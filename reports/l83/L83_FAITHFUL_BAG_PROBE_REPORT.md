# L83 faithful target-bag probe report

## Execution

The authoritative run is
`outputs/l83/train/faithful_bag_attempt1/`.  It used the fixed
`20260829` seed, 524 fit groups, 138 video-disjoint fit-derived dev groups,
10 epochs, and four DDP workers.  Feature tensors remained process-local;
there was no raw/dense feature cache and no fixed 16/24, screening, or
official-test label read.  All traces contain 5,240 finite, nonzero-gradient
updates per representation.

## Corrected target-bag results

| representation | old bag hard | new bag hard | old hit@1 | new hit@1 | old multi exact | new multi exact | old swap acc | new swap acc | old V2 hard | new V2 hard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L59 fused ROI | 0.794872 | 0.794872 | 0.320513 | 0.285256 | 0.257862 | 0.270440 | 0.733553 | 0.725329 | 0.774869 | 0.806283 |
| L81 candidate evidence | 0.932692 | 0.919872 | 0.137821 | 0.157051 | 0.119497 | 0.132075 | 0.490132 | 0.457237 | 0.921466 | 0.942408 |
| L82 candidate reference | 0.807692 | 0.778846 | 0.288462 | 0.336538 | 0.207547 | 0.207547 | 0.754934 | 0.748355 | 0.832461 | 0.816754 |

The complete machine-readable gate is
`outputs/l83/train/faithful_bag_attempt1/faithful_gate.json`.  G6 (finite,
complete rows, no deletion/truncation) passed for all three representations.
G1--G5 did not pass for any representation: L59 hard did not improve, L81's
hard improvement was only 0.012821 and its V2 result worsened, and L82's hard
improvement was only 0.028846 with insufficient hit@1/multi-target/V2 gains.
Thus the status is exactly `faithful_target_bag_training_gate_fail`.

## Reload evidence

`outputs/l83/audit/faithful_reload_attempt1/reload_audit.json` strictly loaded
all three 66,561-parameter packages.  Synthetic output shape was `[2,5]`,
all outputs were finite, and maximum reload output difference was `0.0` for
each package.  This closes the implementation/reload gate but does not alter
the failed semantic evidence gate.

The first scientific root cause is
`grounding_representation_target_separation_insufficient` under faithful
duplicate-aware target supervision.  This is not a claim that every possible
raw representation or detector is exhausted.
