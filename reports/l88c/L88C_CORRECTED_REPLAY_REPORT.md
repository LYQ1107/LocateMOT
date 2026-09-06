# L88C corrected candidate-vs-NULL replay

## Scope

L88C is a zero-training continuation of L88. It changed only the candidate/
NULL gate used during replay; it did not update a checkpoint, run backward,
change the bank, or change the selection/threshold grids.

- Worktree: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L88C`
- Branch: `codex/l88c-candidate-null-corrected-replay-20260905`
- Base L88 commit: `c9b44c07b9b977de9d0f839fb2ff6363abb0386e`
- L88C code commit: `e8f5f62`
- Fixed manifest SHA256: `06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`
- Training: not run; old L88 checkpoints only
- Screening/official-test labels: not read
- Ordinary MOT/OVMOT/TAO: untouched

The corrected emission contract is:

```text
candidate_energy >= candidate_threshold
AND candidate_energy - null_logit >= null_margin
AND presence_logit >= presence_threshold
```

Candidate-only control uses only the first condition. The prior L88 helper
incorrectly used `presence_logit >= presence_threshold` as a gate before the
candidate-vs-NULL comparison. The old directories remain unchanged. The
first corrected selection attempt is preserved as an implementation failure
because the legacy TrackEval matrix used `scope` while the selector required
the equivalent `scope_key/full_video`; the provenance alias fix was pushed in
`c5a6685`, and the rerun is `trackeval_matrix_attempt2`.

## Dev refit and full-video replay

`outputs/l88c/dev/corrected_reselect_attempt2/` refit all 20 even checkpoints
with rules B/R/P on the registered candidate threshold and NULL-margin grids.
The corrected shortlist was epochs 8, 20, 40, 30 and 4. The complete replay
is `outputs/l88c/dev/fullvideo_corrected_attempt1/`: all five candidates, all
three rules, both Refer-KITTI domains, complete rows, and no candidate
deletion/truncation. Internal TrackEval is in
`outputs/l88c/dev/trackeval_matrix_attempt2/`.

The frozen dev selection is in
`outputs/l88c/dev/final_selection_attempt2/checkpoint_selection.json`:

| item | frozen value |
|---|---|
| epoch | 30 |
| rule | B |
| candidate threshold | 1.0 |
| presence threshold | -1.0 |
| NULL margin | 0.0 |
| dev V1 HOTA / DetA / AssA | 0.296615 / 0.216472 / 0.408302 |
| dev V2 HOTA / DetA / AssA | 0.265682 / 0.156622 / 0.451938 |
| selection timing | frozen before fixed validation |

These are internal dev TrackEval results only. They are not screening,
official-test, or final RMOT results.

## Fixed 16-calibration/24-validation replay

The authoritative fixed evaluation is
`outputs/l88c/eval/fixed_semantic_corrected_attempt1/`. Its preselection
schema audit confirms 40 native rows, 16 calibration then 24 validation,
forbidden labels absent before selection, finite scores, and no candidate
deletion/truncation. Calibration and validation labels were attached only at
their registered points.

| method / control | recall | precision | FP/frame | pred/positive | hard violation | multi-positive recall | empty | inactive false acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| immutable L29 | .733333 | .083019 | 10.125 | 8.8333 | .916667 | .819444 | — | — |
| L88 original final Rule B | .214286 | .166667 | 1.875 | 1.2857 | .909091 | .169814 | .375 | .636364 |
| L88C pure corrected gate | .257143 | .174757 | 2.125 | 1.4714 | .909091 | .215268 | .250 | .727273 |
| L88C corrected final, candidate-only | .419355 | .220339 | 1.9167 | 1.9032 | .846154 | .347222 | .2917 | .666667 |
| L88C corrected final, candidate+NULL rule | .354839 | .207547 | 1.750 | 1.7097 | .846154 | .305556 | .375 | .666667 |

The pure-gate row isolates the formula change on the historical L88 final
records. The corrected-final row also changes the frozen checkpoint and dev
selected thresholds, so it is not a pure causal attribution of the formula.

## Gate decision

`gate_decision.json` is `semantic_gate_fail`. Hard violation, precision,
FP/frame, predictions/positive and finite/key checks pass the registered
floors, but recall `.354839 < .7233333` and multi-positive recall
`.305556 < .7894444`. This is not a deployable emission and not a HOTA claim.

The corrected formula therefore fixes the accounting/gating contract and
improves hard-negative violation, but it does not provide recall-preserving
multi-positive correspondence.
