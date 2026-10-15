# GroundJEV-MOT current evidence — 2026-10-15

Independent worktree `/data2/user/LocateMOT_GroundJEV`; original LocateMOT data
and preserved U2 results remain under `/data2/user/LocateMOT`. The full goal
remains active. These are registered legal validation scopes, one seed; no
complete-dataset result or successful unified-model promotion is claimed.
Earlier snapshots/failures remain in Git, `STATUS_HISTORY_20261015.md` and
`research_log.md`. Official artifacts remain authoritative.

## Complete DanceTrack video

Same original V1 Ground proposals, all1203frames of validation0004,
13664candidates/27parser-rejected frames; unchanged official TrackEval.
HOTA/IDF1 below are percentages, IDSW/FP/FN counts.

|Method|HOTA|IDF1|IDSW|FP|FN|
|---|---:|---:|---:|---:|---:|
|Simple|8.530|3.956|746|9334|256|
|Distilled Compact300|20.157|17.140|588|2129|951|
|Matched MLP300|11.197|15.118|161|549|3556|
|Same Set architecture, GT-only300|5.977|3.406|1322|5954|1796|
|New original U2 core/bridge300|1.243|0.625|250|455|4191|
|Shared Compact, broad native history|26.316|30.633|165|579|1962|

Shared Compact has1146511learnable resident parameters plus original frozen
OneJev CPU token lookup254279680parameters/508559360nominal bytes and external
Ground perception. It is not a replacement claim for the0.8B decoder. The
shared stage has a larger/different training budget than the matched300arms.
Lower IDSW alone does not establish improvement when FN rises.

Fresh pretrained Full V3 uses a different actual source: HOTA2.389 versus its
own matched Simple8.257. Actual unresumed native pipeline FPS.118560, under
concurrent workloads, excluding preloaded service/model initialization. Its
historical final-checkpoint timer definition is preserved. Trained Full native600/
MOT500/teacher600 full1203 controls continue on the original V1 source; do not
infer scores before completion or compare different-video speeds as paired.

## All12complete800frame RMOT controls

Both expressions cover all frames0..799 of legal Refer-KITTI-v2 validation0007.
Single has18target boxes/782empty-target frames; multiple2112/247. Same actual
native sources1412/2208candidates,36/19parser rejections and every empty/rejected
frame retained. Official parsing unchanged; strict-wire diagnostic failures are
not relabeled as valid or omitted. All12actual official/runtime/video receipts
validate; the original blocking owner exited0 and released its own services.

|Method|Single HOTA %|Single IDF1 %|Single FN|Multi HOTA %|Multi IDF1 %|Multi IDSW|Multi FP|Multi FN|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Simple|4.966|1.259|8|31.877|37.685|99|1076|980|
|Shared Compact|6.025|2.890|13|19.802|21.008|165|619|1054|
|Pretrained Full|16.119|17.391|16|8.621|7.635|215|214|1766|
|MOT500 Full|0|0|18|0|0|0|11|2112|
|Teacher-history additional600 Full|6.847|4.839|14|4.578|4.228|108|104|1963|
|Own-history additional600 Full|2.506|2.062|17|6.740|6.693|185|145|1859|

Both additional600stages start from the actual completed MOT500 weights and
have600further updates/2400examples each. They partly recover MOT500's RMOT
collapse but do not beat pretrainedFull here. PretrainedSingle's higher HOTA
still accompanies16/18misses. No training/unified-model success is claimed.

Compact fresh runs used live pristine DLM and real native2560ROI, association,
causal memory, archive/prediction writes, rendering/encoding and final export:
actual pipeline FPS.678005/.882942, wall1179.932/906.062s. Core initialization
5.232/5.362s and remote preloading excluded. Simple/Full in this table replay
these exact sources, so their replay FPS is not comparable to native Compact
FPS. Archive total197218901bytes<2GiB. Core Torch allocator peaks14970368/
14855168bytes and core RSS1467027456/1478381568bytes exclude external services.

|Empty-target frames with any output|Single|Multiple|
|---|---:|---:|
|Compact|200/782=25.575%|143/247=57.895%|
|Simple|747/782=95.524%|243/247=98.381%|

Compact reduces false alarms but loses multiple-target identity accuracy.
False-alarm frame counts and detection counts are separate. Live-video paths:
`multitask_runs/full_rmot_shared600_v1/{single,multi}800_shared_compact_fresh/tracks.mp4`.
Receipt: `multitask_runs/full_rmot_shared600_v1/all12_validated_summary.json`.

Paired fresh Fullnative600/Compact800same-video/same-GPU5 E2E has completed.
All800frame/image/box/label/native-score/score-kind/parser-rejection records are
exactly equal:1412candidates and36rejected frames. Official accuracy exactly
matches the prior same-candidate native600/Compact controls. Both decoded800.

|Fresh same-video method|Actual pipeline FPS|Wall seconds|Sampled owned VRAM GiB|HOTA %|FN /18|
|---|---:|---:|---:|---:|---:|
|Full native600|0.495914|1613.184|36.184|2.506|17|
|Shared Compact|0.687099|1164.316|19.166|6.025|13|

Observed pipeline FPS ratio1.3855x, wall reduction27.825%. Sampled total owned
GPU peaks37052/19626MiB (47.031%lower), five-second samples are lower bounds
on instantaneous peak, not exact maxima. Full core Torch allocator8426480128
bytes/RSS3305140224bytes; Compact14970368/1466036224bytes exclude external
services. Compact also has original frozen CPU text lookup; the1.146M core
alone does not describe whole-pipeline model/memory cost.

Actual pipeline includes imageIO/nativeDLM/ROIwhenrequired/association/state/
archive/prediction/render/video-encoder completion/final export. Private
service startup26.052/26.046s and core initialization8.010/5.004s are excluded
and separately recorded. Full uses real images/crops; Compact additionally
pays nativeROI cost. Per-frame inference p50/p95seconds Full1.810/2.930,
Compact1.202/2.136 exclude trailing output/state/video writes that are included
in whole-pipeline wall time. Both remain below1FPS with poor18-box recall.

Empty-target alarm frames Full67/782 versusCompact200/782; fewer misses/higher
HOTA here still trade against more false alarms. One single-expression pair,
fixed Full-then-Compact order, one seed, concurrent other CPU/project loads:
no counterbalanced/statistical speed or real-time claim. Historical other-video
or replay FPS is unchanged and not used for this ratio. Evidence:
`e2e/full_compact_rmot800_v1/validated_summary.json` and both run artifacts.
All prior owned groups exited before GPU5followup allocation; localization-only
training has now actually started on GPU5, seed42/200updates/identity weights0.
Snapshot: `ground_adaptation/localization200_control_gpu5_v1/registered_start_snapshot.json`.

## Same-source saved-output diagnosis

All2208candidate outputIDs/boxes reconstruct exactly from saved actions and
original membership>=.5, with no new model calls, corrected states, thresholds
or predictions. Same1132one-to-one aligned query-positive candidates:

|Method|No physical assignment|Assigned then membership rejects|Actually emitted aligned positives|
|---|---:|---:|---:|
|PretrainedFull|105|687|340|
|MOT500Full|263|869|0|
|Teacher600Full|544|445|143|
|Native600Full|330|555|247|
|SharedCompact|4|71|1057|

These disjoint candidate funnels are not an officialFN decomposition across
HOTA thresholds. Per-target proposal coverage1133/2112uses a different matching
rule. Unknown/unmatched candidates are not fabricated negativeGT. Full query
rejection and association decisions both need work; the membership gate alone
is not the whole teacher/native600failure. Different causal birth-anchor event
sets prevent strict same-event identity-accuracy comparisons.
Receipt: `diagnostics/multi800_full_compact_output_funnel_v1/summary.json`.

## Completed matched input-evidence study14.2

All12full-video controls complete: three modes on fullDT1203/rawTAO40/single800/
multi800. Two new1200step models and reused completed all_three1200 full-input
weights share actual initialization, all1200sample IDs, seed0,536source groups,
GT/native teacher targets and1146511resident parameters. Three-task reload
delta0. Full-input official scores exactly reproduce all four prior baseline
scopes despite changed state-storage implementation.

|Runtime evidence|DT HOTA %|RawTAO TETA %|Single800 HOTA %|Multi800 HOTA %|
|---|---:|---:|---:|---:|
|Geometry/motion, ROI/specification branches masked|9.144|13.568|0|14.462|
|Native appearance, specification condition masked|26.061|11.444|2.491|12.900|
|Native appearance plus specification condition|19.312|10.081|7.860|18.674|

Appearance helps this MOT scope; specification conditioning helps both RMOT
expressions but hurts MOT/TAO here. All NovelClsA0, and every multi800arm is
below same-candidate Simple31.877HOTA. Keep negative transfer; no universal
multimodal/JEV superiority claim or validation-threshold tuning.

Geometry-motion disables382208visual/history-projection parameters, so equal
resident count does not mean equal active capacity. All modes retain the native
semantic frontend, softteacher knowledge and output-only class-vocabulary text
prototypes. These are association-input ablations on common older-actor training
histories, not language-free perception or unique JEV proof. Full native-image
JEV differs in architecture/training budget and is separately reported.
Receipt: `compact/input_evidence1200_v1/controls/validated_summary.json`.

## Complete TAO annotation-frame scope and policy studies

One legal Base/Novel video:40annotation frames/160GTboxes, all1203LVIS input
categories. Raw source35228proposals, separate exact geometry-alias protocol
7343groups retaining every alias. Physical source FPS unknown; annotation-step
timing is not physical seconds. Official unchanged TETA, all NovelClsA0.

|Proposal protocol/method|TETA %|AssocA %|ClsA %|
|---|---:|---:|---:|
|Raw/Simple|18.640|28.618|20.290|
|Raw/shared Compact|14.517|5.148|30.314|
|Aliases/Simple|22.556|26.472|18.325|
|Aliases/shared Compact|21.773|9.627|32.419|

Raw versus alias changes proposal representation/representative labels, not
association alone. Coverage147/160atIoU.5. Frozen-trajectory correct-option
coverage32/64/128 is15.672/31.343/44.776% raw and53.097/60.177/74.336% alias;
these are ranking diagnostics, not counterfactual tracking scores.

Original five-policy cohort25units and both full800RMOT five-policy cohorts
10units complete. Same-input DT HOTA learned/IoU/cosine/cosine+learnedaux/
learned+fixedaux=26.316/7.488/13.311/13.009/24.283. But fullmulti800 gives
19.802/32.813/16.622/22.035/19.558. Current learned association loses to fixed
IoU plus identical semantic functions here. Histories/confidence/action priors
and filtering evolve by policy; not optimal retrained baselines or JEV proof.

Actual context study12units, four32references/eight new64/128trajectories:

|Context IDs|DT learned HOTA %|DT fixed cosine HOTA %|AliasTAO learned TETA %|AliasTAO fixed cosine TETA %|
|---|---:|---:|---:|---:|
|32|26.316|13.311|21.773|23.913|
|64|26.316|13.289|23.100|26.099|
|128|26.316|12.491|24.749|29.062|

Larger context helps this TAO scope but exceeds training32options and changes
semantic normalization/histories; fixed cosine still leads. No isolated JEV gain.

Hybrid32retrieval12units, six new/six geometry references, all complete:

|Scope|Learned geometry|Learned16geometry+16appearance|Fixed cosine geometry|Fixed cosine hybrid|
|---|---:|---:|---:|---:|
|DT HOTA %|26.316|26.316|13.311|12.811|
|AliasTAO TETA %|21.773|21.466|23.913|24.435|
|RawTAO TETA %|14.517|15.178|13.964|15.652|

Mixed gains; geometry remains the default. All receipts under respective
`compact/{fixed_affinity_controls_v1,context_sizes_v1,hybrid_shortlist32_v1}/validated_summary.json`.

## Completed task-subset and memory studies

Four new task subsets share MOT300+seed0semantic initialization,1200totalupdates,
536groups and1146511resident core parameters. Per-task exposures differ:
MOT1200; pair600each; triple400each. All24controls complete:

|Tasks|DT HOTA %|RawTAO TETA %|Single800 HOTA %|Multi800 HOTA %|
|---|---:|---:|---:|---:|
|MOT|19.639|12.421|0|7.673|
|MOT+RMO|6.760|19.284|0|18.255|
|MOT+OV|21.229|12.850|0|15.794|
|All three|19.312|10.081|7.860|18.674|

All NovelClsA0. No universal transfer; older shared26.316DT/19.802multi model
has different initialization and joint1200+native1200 budget. MOT400/600extra
controls HOTA13.317/13.979 have exact actual sample/loss prefixes versus1200,
but different total compute prevents pure gradient-interference attribution.
Receipt: `compact/task_subsets1200_cpu_v2_feature_sources/validated_summary.json`.

Complete DT1203 memory interventions HOTA11.441(frame-only),16.394(recent),
14.643(EMA),24.245(anchor/always),26.316(anchor/gated3s),18.552(gated30s).
These are frozen policy interventions; no successful >3sDTrecovery evidence.
SingleRMOT800has two natural >3s candidate returns, recovered0 by both Compact/
Simple under immutable correct-birth-anchor diagnostics.

Natural MOT20-05validation1657frames(1659..3315,25Hz) is now queued on GPU5
after pairedE2E and localization200 complete. OriginalGPU7retention continues
independently; this execution-device/dependency amendment occurred before any
natural inference in the superseded waiting owner. Eight same-source CPU memory/Simple
controls, packed FP32 archive3GiBcap. Registered ID854's29.76s low-visibility
interval retains743intermediate annotated frames; it is not a proven missing-
candidate interval. No synthetic occlusion/GT correction/visibility filtering.
Real21frame exact archive regression preserves266candidates/rejected empty
frame/full state/next-frame restore:14998172JSON-equivalent→3409159packed bytes.
No new accuracy/speed claim from this storage test.

## Training, adaptation and remaining work

Original One100real uncorrected rollout→500SFT and Compactjoint1200real72frame
rollout→native1200 provide offline iterative own-state training. Not per-gradient
on-policy resampling; the latter is a research direction, not an invented gate.
Both additionalFull600FP32/Adam/scheduler/RNG states saved,2400examples each,
nativeMM reload delta0. Teacher resumed450, native250 exactly; teacher replayed
475loss differs.001734814 and native has no saved-log overlap. Neither called
bitwise training replay; resumed-tail timings are not whole-stage throughput.
All12same-policy shortFull controls completed, but complete-video results above
show the short multiRMO gains do not establish full-video improvement.

Fit72update audit reproduces original labels:28positive,12unknown birth owners,
0two-known-identity-conflicting positives. One known conflicting write already
has target0. Initial unknown→negative diagnostic suggestion was invalid and
never applied. No pack/model changed; do not claim this label hypothesis proved.
Authoritative audit: `diagnostics/causal_update_supervision_fit72_v3_known_conflicts/audit.json`.

Ground200trained native27267584parameter projector, native dense/referring plus
identity losses, exported569tensors with remaining weights untouched. Ground320
adds actual absence supervision. Isolated original/200/200repeat/320controls
preserve request order/repeat raw outputs exactly, show mixed MOT/RMO behavior;
no promotion. Original/Ground200nativefull9retention each targets91124released
rows, not all30paper benchmarks; neither complete while native clients remain.
GPU6native CUDA failure/retry evidence retained; first failing kernel unknown.

Localization-only200 now follows the completed/released pairedE2E GPU5 owner: same112rows/56pairs/200actual draws/seed42/lr2e-6 as completedGround200,
identity coefficients0; then fresh export,96ordered native frames and18Full
visual×pretrained/native600composition controls. This is composed adaptation,
not joint-gradient training. All new frontend/factorial comparisons run on GPU5;
the actual reference tracking200 training ran on GPU6 (same A800 model), so no
bitwise cross-device training trajectory claim. No pending adaptation/retention
score inferred.

|Queue/device|Remaining actual work|
|---|---|
|GPU4|TrainedFull native600/MOT500/teacher600DT1203; `training/onejev_full1203_controls_v2_wait_fix`|
|GPU5|Paired E2E complete; localization200training now running, then96frames/18controls and naturalMOT20/8CPUcontrols; `gpu5_followups_v1`|
|GPU6|AdaptedGround200nativefull9 remains unchanged|
|GPU7|OriginalGroundnativefull9 remains unchanged|
|CPU|Policy25, tasksubset24, context12, hybrid12, input12 all complete|

At most four physical GPUs4/5/6/7combined; foreign jobs and stopped U2 queues
untouched. No new GPU/feature cache for completed CPU studies. Conservative
prior feature upper22471148986+3GiB+32MiBregression reserve=25725928890bytes
<25GiB(26843545600). Data/checkpoints/results retained. Historical old-cache
cleanup freed39.89GiB; resource-snapshot limits in `RESOURCE_CLEANUP.md`.

Compact batches FP64 geometry and caches same-timestamp predicted boxes in
new processes. Real591×2365=1397715overlaps/37groups32/64/128ordering and full
model outputs/lifecycle/metadata match scalar exactly. Three warm16candidate
post-state probe medians147.08→51.11ms geometry and784.82→675.81ms hybrid;
not causal-video or E2E FPS. Historical timings unchanged, actual childGit
revisions recorded. Receipt: `validation/dense_geometry_batching_v1/validation.json`.


Execution scheduling amendment: both old localization/natural owners were
verified still CPU-only waiters, no children/completed experimental units, and
superseded with preserved manager-handoff receipts. The old localization shell
returned241 from intentionalSIGTERM; this is a canceled wait, not a completed
measurement or hidden experimental failure. New owner2207841/start43963616
(blocking session36847) waits original paired owner1953808/start42350971 before
any GPU5 allocation. It verifies successful paired completion and all owned
process groups gone, then runs both existing studies serially. No active
training/inference/retention jobs stopped. Real112row/200draw/native-proposal
alignment matches, all other scientific plan fields and96/18/1657/8scopes
unchanged; GPU5/6/7 allNVIDIA A800-SXM4-80GB. Cache bound unchanged.
Receipts: `validation/localization200_contract_gpu5_v1/validation.json` and
`validation/gpu5_followup_registration_v1/validation.json`.
New plans: `configs/groundjev/gpu5_followups_v1.json` and referenced GPU5variants.


The GPU5handoff is now executed, not only queued: paired blocking session87824
exited0; its five recorded model process groups are all gone. New serial owner
2207841 started localization child2213633/start44011696. The actual trainer
reports physicalGPU5, native_dense_referring_localization_only, seed42,200target
updates, teacher/native identity coefficients0. This is a start/device receipt,
not training completion or new adaptation accuracy. Original retention6/7 and
Full1203owner remain independent.
