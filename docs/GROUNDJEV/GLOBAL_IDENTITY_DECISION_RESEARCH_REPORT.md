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

## A3 four real memory policies — DanceTrack1203 complete

All four use exactly the final A3 physical parameters, identical original candidates and task, with actual online trajectories.

|Policy|HOTA %|AssA %|IDF1 %|IDSW|FP|FN|UPDATE/HOLD flips vs original|
|---|---:|---:|---:|---:|---:|---:|---:|
|always|25.202|13.674|29.204|156|584|1894|392|
|confidence .7|21.062|10.974|23.505|132|469|2332|1001|
|original|24.752|13.687|29.256|197|706|1948|0|
|assigned|25.066|13.509|30.718|227|1229|1657|2937|

Assigned memory changes real decisions and adds0.314 HOTA points over original, but increases IDSW197→227 and FP706→1229, while FN1948→1657 and IDF1 improve. It does not beat always-update HOTA25.202 or A0 HOTA26.316, so this is a mixed/negative result for the proposed memory mechanism, not a tracking improvement claim. Remaining four scopes continue with the same fixed checkpoint; no task-specific selection. Exact scalar receipts and posthoc contamination/recovery diagnostics are retained.

A3 DT posthoc: original→assigned known wrong long writes974/1782→221/514; wrong recent writes1170→1462; recovered previous IDs after natural candidate gaps44→63 of252returns; extra accepted duplicate identities424→754. Long-memory protection improves its local measured behavior, but observational contamination and duplicate outputs increase. These trajectory-dependent denominators accompany the official FP/IDSW regression, rather than replacing it.

## Missing physical RMOT training and one registered continuation

All12 A3 DT/RMOT policy units completed. Single-expression HOTA remains0 for every policy. Multi-expression HOTA% always6.088/confidence6.078/original5.685/assigned7.425; assigned IDSW18 versus original11, still far below A0 19.802HOTA. A3 reused1428RMOT ID references only as diagnostics because the A2 physical parameters were frozen. A1/A2/A4 additional physical training is MOT-only, so shared initialization alone is not evidence of new multi-task physical learning.

The existing18 legal RMOT FIT source rollouts have now been audited with the original FIT exporter:3628query-frame exposures,10868candidates,4171uniquely supported/6697unknown or ambiguous and11016GT-object observations. No validation/reserved GT is opened. Register one A4b continuation from the fixed finalA4 with both43MOT and18RMOT units, complete units interleaved, same optimizer/schedule and TRAIN-derived birth weight. Each8/16/32phase processes4684frames; GT remains external and own predicted states are never repaired. RMOT write reliability uses IoU>=.6 as a spatial proxy, not measured visibility. Baseline A4 evaluation remains mandatory and unaltered; A4b is a separate model with its own fixed five-scope comparison. No outcome is claimed before actual training/evaluation.

Final A3 runtime validation on a real31candidate/31history TRAIN frame checks100existing assignments across MOT/two RMOT queries/OVMOT: physical probabilities/admission/global assignment/assigned write values are exact and checkpoint reload maxdifference0. Scope is one identical physical input, not changed candidate sets or whole-video accuracy. The original native Full3175/28690, short2875/6000 and pairedRMOTphysical600/6000 owners remain live at the2026-10-17 observation. Mixed A4b versus A4 also differs in additional training exposure; an eventual quality difference alone cannot isolate the causal effect of RMOT supervision from additional optimization.

## Storage correction (2026-10-17)

The initial2MiB stage reservation understated rendered video and official-scorer derived outputs. Actual new-stage conservative649602123bytes plus actual source22584176800 totals23233778923<25GiB, with scientific predictions/models separately protected. Two completed historical namespaces have absent producer PIDs and stopped owned services: FullRMOT allfiles886733321 under old2GiB cap, pairedE2E allfiles171395400 under old1GiB cap. Close only those unused reservations, deleting nothing; count even their scientific outputs conservatively. New stage2GiB reservation yields global formal upper26824992559<26843545600 (18553041headroom). Accounting-only V2 amendment supersedes old storage fields; running experiment configs/source code remain unchanged. The initial2MiB accounting was insufficient and is not presented as a valid current cap.

## A4 own-state training completed

Final A4 completes247actual updates/3168frames/144153candidate exposures. Actual episode counts8/16/32 are132/66/49. Compressed per-frame proof validates247resets and2921next-frame predicted-ID continuities, with no teacher history replacement; it does not independently verify every numeric feature value. Training includes11427predicted births and6796known wrong protected-long writes; these are training-trajectory observations, not official tracking results. Original five-scope evaluation and independent mixed continuation started automatically from the fixed finalA4. Early mixed RMOT0000 frames are mostly unknown-masked, and an early nonzero update is competition-only; actual RMOT ID-supervision evidence is still required. Do not equate3628processed FIT frames with3628valid identity labels.

## A4 first three complete official scopes: adverse birth/recall tradeoff

|Scope|HOTA %|DetA %|AssA %|IDF1 %|IDSW|FP|FN|
|---|---:|---:|---:|---:|---:|---:|---:|
|DT1203|16.007|31.037|8.257|13.892|599|7307|268|
|RMOT single800|6.666|3.285|13.622|4.494|1|155|13|
|RMOT multi800|15.568|26.902|9.177|14.670|88|480|1282|

DT badly regresses despite fewer FN; do not hide FP7307 or IDSW599. Single RMOT exceeds A0 HOTA6.025 slightly, multi remains below A0 19.802. A4 combines own-state training, V2 identity labels and cost-sensitive birth loss, so these results do not isolate any single mechanism. TAO/MOT20 remain pending. Exact HOTA/AssA/IDF1/IDSW/FP/FN deltas against A0 are saved in A4_first_three_deltas_vs_A0.json.

The paired A4c mixed continuation is now registered from exactly the same A4 model and same4684-frame/phase mixed source/seed/optimizer/8-16-32schedule as running A4b. Only birth-loss weight changes from fixed TRAIN6.6687854 to standard CE1.0; no runtime threshold, candidate, seed or scorer change. Implementation equality after removing that single factor plus output arm/status differences is checked. Own-state trajectories and effective gradients can diverge, so no FLOP/state equality claim. This is one prespecified causal control for the observed over-birth hypothesis, not a held-out threshold sweep; final five-scope quality remains required.

## Refined A4 failure diagnosis

DT has5979GT-supported extra duplicate proposals in every fixed-candidate arm. Accepted extra identities A2/A3-assigned/A4 are424/754/5184. Explicit learned graph duplicate rejection is0/0/55. A4 known-existing wrongly NEW is only24; much of the failure is several live IDs accepting proposals of the same entity, beyond an isolated NEW decision. Unique allocation of numeric IDs and uniqueness inside predicted graph components do not establish physical-entity exclusivity. Wrong protected-long writes become6580/8550known updates (recent wrong writes6580), against A3-assigned221/514 and1462recent wrong writes. Joint own-state training did not preserve A3's local memory protection, and its sigmoid was explicitly not re-calibrated.

TRAIN graph audit: MOT13036positive duplicate edges/2111172known other-entity negatives, positive micro fraction0.006137,476023unknown masked. RMOT3038positive/6744negative, fraction0.31057,28070unknown masked. These are edge-level micro priors; losses are averaged per frame, so they are not equivalent to the optimizer's effective positive weight or proof of causation. Whole-set retrieval coverage and label audit success are distinct from learned duplicate admission quality. The cost-neutral paired control remains one isolated factor test; it may not resolve the more direct candidate-admission/global-entity exclusivity failure. Further architecture changes require its and mixed-model official evidence, rather than further arbitrary threshold sweeps.
