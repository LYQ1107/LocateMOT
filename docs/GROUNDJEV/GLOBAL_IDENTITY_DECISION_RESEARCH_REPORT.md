# Global Identity Decision research report

Status: R1 audited; A1/A2 and A3 CPU training complete. R2 five-scope evaluation, A3 four-policy controls and real own-state training are in progress. The full GroundJEV goal
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

## Completed full-video results (registered subsets, not full benchmark validation)

|Arm/scope|HOTA %|DetA %|AssA %|IDF1 %|IDSW|FP|FN|
|---|---:|---:|---:|---:|---:|---:|---:|
|A1_mot_full1203|11.037|27.561|4.447|11.051|137|316|3045|
|A2_mot_full1203|24.752|44.843|13.687|29.256|197|706|1948|
|A1_rmot_single800|0.000|0.000|0.000|0.000|0|30|18|
|A2_rmot_single800|0.000|0.000|0.000|0.000|0|17|18|
|A1_rmot_multi800|7.490|6.001|9.547|8.079|20|147|1945|
|A2_rmot_multi800|5.685|3.940|8.569|6.011|11|113|2008|

A0 registered HOTA: DanceTrack26.3157, RMOT single6.0247, RMOT multi19.8019.
A2 DanceTrack is 1.564 HOTA points below A0, with IDSW165→197, FP579→706,
FN1962→1948. A1 lowers IDSW/FP but sharply increases FN. Both A1/A2 regress on
RMOT, including zero single-expression HOTA. These are negative results; Stage A
helps relative to A1 on DanceTrack but does not establish a cross-task benefit.
TAO40 completed for both arms; MOT20 full1657 remains with the uninterrupted R2 runner.

## Actual training and runtime evidence

A1:4056 optimizer updates,386.37 CPU seconds. A2:4064 updates,409.87seconds.
Both consume192204 candidate exposures /4224frames from identical source streams.
Eight A2 extra steps arise from duplicate/ranking-only valid frames: optimizer/FLOP
counts are not matched. Physical core979883 parameters; frozen semantic head1146511.
Runtime checks find physical logits and original update probabilities identical for
four task/query inputs on one real frame; semantic readouts differ. This does not
prove invariance when detector candidates change. Official replay speed is roughly
19fps on DanceTrack and36–40fps on RMOT; detector-inclusive end-to-end FPS is unknown.
New decision GPU peak is0bytes (CPU), not the whole system GPU peak.

A3 completes3602 actual write-head updates (33281 trainable parameters). All A2
physical tensors are bit-exact. All2882 existing RMOT references are reused per
epoch:1428 ID references plus1454 write labels. Training uses real original actor
states, off-policy for A2. It is not A2 own-state training. Train-unit-held-out write
calibration5465 questions reduces BCE4.375→0.668 (scale0.1979/intercept1.9131).
This is calibration evidence only, not tracking improvement. Four actual policies
(always/confidence/original/assigned) now run on all five scopes with one A3 model.

## Failure analysis and versioned follow-up

On A1 DanceTrack, known supported MATCH events are1007 correct/490 incorrect;
entity-frame MATCH recall972/(972+2235). There are412 known wrong protected-long
writes and490 wrong recent observational writes. Of115 retirements,58 have the
bound GT still annotated. Correct existing-ID ranks3359/6938. The old-ID coverage
is6938/9462. See the immutable diagnostic for denominator limitations: unknown
birth bindings are not false-positive background and NEW recall can be undefined.

V1 identity CE conflates duplicate proposal rejection with entity identity, while
first annotation presence can precede any effective model birth. The additive V2
contract separates proposal competition from ID labels. Audit:28507 MATCH-supported
proposals,641 first-effective NEW proposals,6083 annotation-unknown and12820
retired/unbound unknown.3572 old duplicate-REJECT proposals receive positive entity
ID targets;40 old UNKNOWN become first-effective NEW. V1 files/results are retained.

A4 real model-in-loop8/16/32-frame training is now launched from A3 using V2 labels.
It uses actual predicted causal histories, detached per-frame gradients accumulated
with fixed episode weights, then one update per episode. It is not temporal BPTT.
TRAIN-derived birth cost is sqrt(28507/641), without held-out metric tuning. GT is
external supervision and never repairs predicted states. Final32-phase model is
registered for every task, with no best-metric checkpoint selection. Actual
completion, write calibration after joint updates and official gains remain pending.

## Still required

Complete all five A1/A2 scopes and20 A3 memory controls; evaluate A4 across all five
scopes with identity/write/recovery diagnostics. Fix and validate semantic transfer,
including Novel ClsA (currently unresolved at0). Add a parameter/input-matched non-JEV
control and targeted Full experiments when original GPU owners finish. There is no
claim of mature-model parity, common improvement across MOT/RMOT/OVMOT or independent
JEV superiority. Both milestone and full research goal remain incomplete.

## TAO40 completed (same original annotation-frame protocol)

A1 TETA25.293/LocA13.651/AssocA31.933/ClsA30.297. A2 TETA32.306/LocA26.066/AssocA42.014/ClsA28.838. A2 improves TETA by17.789 points over coupled A0 (14.517), but only1.488 points over the existing task-conditioned canonical readout (30.818). Both contrasts matter because physical and semantic structure differ from coupled A0. A2 Base TETA39.0145/ClsA43.257; Novel TETA18.889/ClsA0. Novel classification remains unsolved. RMOT regression prevents a unified improvement claim.

A0 all-five posthoc identity diagnostics are now complete and saved separately; source predictions remain read-only. For example A0 DT has1190 known wrong protected-long writes/1268 wrong recent writes, but the opportunities and trajectories differ from A1, so report counts and denominators together rather than treating count decreases alone as a causal gain.
