# GroundJEV-MOT current evidence — 2026-10-15

Project worktree: `/data2/user/LocateMOT_GroundJEV`; original LocateMOT data
and U2 results remain in `/data2/user/LocateMOT`. The full research goal is
active. Results below are legal development scopes, one seed; they are not
complete-dataset results or a successful unified-model promotion.

Earlier snapshots and failures are preserved in
[status history](STATUS_HISTORY_20261015.md), `research_log.md` and Git history.
This page summarizes current evidence rather than repeating the running log.

## Complete-video tracking

All DanceTrack rows use the same original native V1 proposals, all1203 frames
of validation0004,13664candidates and27rejected frames. Official TrackEval is
unchanged. HOTA/IDF1 are percentages; IDSW/FP/FN are counts.

|Method|HOTA|IDF1|IDSW|FP|FN|
|---|---:|---:|---:|---:|---:|
|Simple|8.530|3.956|746|9334|256|
|Distilled Compact300|20.157|17.140|588|2129|951|
|Matched MLP300|11.197|15.118|161|549|3556|
|Same Set architecture, GT-only300|5.977|3.406|1322|5954|1796|
|Newly trained original U2 core/bridge300|1.243|0.625|250|455|4191|
|Shared Compact, broad native history|26.316|30.633|165|579|1962|

The shared core has1146511learnable parameters, plus original frozen OneJev
CPU token lookup (254279680parameters/508559360nominal bytes) and external
Ground perception. It is not a replacement claim for the0.8B decoder. This
shared stage has a different training budget from the matched300 baselines.
Lower IDSW alone does not establish a gain when FN increases.

Fresh pretrained Full V3 uses a different actual native proposal sequence:
HOTA2.389 versus its own matched Simple8.257. Its uninterrupted detector-inclusive
FPS is0.118560 under concurrent workloads, excluding preloaded model/service
initialization. Do not compare this speed to replay or to other videos as a
paired speedup. Trained Full500/teacher600/native600 full1203 controls are now
running on the original V1 source; no scores are inferred before completion.

## Full800-frame RMOT and actual pipeline speed

Both expressions cover every frame0..799 of legal Refer-KITTI-v2 validation0007.
The fresh Compact runs use live original DLM, real native2560ROI, association,
identity memory, all archive/prediction writes, video rendering/encoding and
final state export. Both are unresumed. Remote preloading and local model
initialization are excluded. Simple controls replay exactly the same source.

|Expression/method|HOTA %|IDF1 %|IDSW|FP|FN|Live pipeline FPS|
|---|---:|---:|---:|---:|---:|---:|
|Single Compact|6.025|2.890|0|323|13|0.678005|
|Single same-candidate Simple|4.966|1.259|3|1402|8|not measured live|
|Multiple Compact|19.802|21.008|165|619|1054|0.882942|
|Multiple same-candidate Simple|31.877|37.685|99|1076|980|not measured live|

Single has18target GT boxes,782empty-target frames and10/18target proposal
coverage atIoU.5. Multiple has2112target boxes,247empty-target frames and
1133/2112coverage. All36/19native parser rejections remain in the800frame
denominators; official-scoring parsing is unchanged and strict-wire diagnostic
failures are not relabeled as strict-valid.

|Empty-target behavior|Frames with any output|False-alarm frame rate|
|---|---:|---:|
|Single Compact|200/782|25.575%|
|Single Simple|747/782|95.524%|
|Multiple Compact|143/247|57.895%|
|Multiple Simple|243/247|98.381%|

Compact reduces false alarms but loses multi-target identity accuracy. Its
single IDSW0 does not offset13misses out of18target boxes. No model promotion.
These are two expressions on one video, not the full RMOT dataset or real-time
performance. Absolute pipeline wall times are1179.932/906.062seconds.

Both sources and decoded videos were validated:1412/2208native candidates,
197218901combined archive bytes under2GiB. Core Torch allocator peaks are
14970368/14855168bytes and core-processRSS1467027456/1478381568bytes; these
exclude external DLM/ROI services. Cross-stage resource snapshots are not
per-video total-VRAM peaks. Paired fresh Full/Compact speed/resource comparison
on the same800frames/GPU5 is still queued.

Evidence under `outputs/groundjev/`:

- `multitask_runs/full_rmot_shared600_v1/fresh_compact800_validation_and_summary.json`
- `multitask_runs/full_rmot_shared600_v1/{single,multi}800_shared_compact_fresh/tracks.mp4`
- `multitask_runs/multi800_simple_cpu_v1/owner_state.json`

The early CPU multiSimple baseline leaves the original GPU5 Full queue intact.
Any later identical scheduled baseline is a duplicate, not another seed.

## Complete TAO annotation-frame scope

The legal Base/Novel video has40annotation frames,160GTboxes and all1203LVIS
input categories. Original native source has35228proposals; a separate exact
geometry-alias policy has7343groups while retaining every original alias.
Physical sourceFPS is unknown; annotation-step timing is explicitly not physical
seconds. Unchanged official TETA, no detector-inclusive replay speed claim.

|Proposal protocol/method|TETA %|AssocA %|ClsA %|
|---|---:|---:|---:|
|Original raw/Simple|18.640|28.618|20.290|
|Original raw/shared Compact|14.517|5.148|30.314|
|Exact geometry aliases/Simple|22.556|26.472|18.325|
|Exact geometry aliases/shared Compact|21.773|9.627|32.419|

NovelClsA is0 in all four. Raw versus alias changes proposal representation
and representative labels, so it is not an association-only improvement. Source
coverage remains147/160atIoU.5. Exact causal replay shows frozen-trajectory
correct-option coverage at32/64/128IDs of15.672/31.343/44.776% on raw data,
and53.097/60.177/74.336% on alias data. These are coverage diagnostics, not
counterfactual larger-context tracking scores.

## Controlled training and policy studies

All four new task subsets have the same MOT300+seed0semantic initialization,
1200total updates,536source groups and1146511core parameters. Per-task
exposure differs:1200MOT,600per paired task,400per triple task.

|Training tasks|DT1203 HOTA %|RawTAO40 TETA %|SingleRMO8 HOTA %|MultiRMO12 HOTA %|
|---|---:|---:|---:|---:|
|MOT|19.639|12.421|0|4.096|
|MOT+RMO|6.760|19.284|0|3.513|
|MOT+OV|21.229|12.850|26.494|9.622|
|All three|19.312|10.081|37.265|3.513|

All24registered scopes validate, including both complete800frame sources.

|Training tasks|Single800 HOTA %|Multi800 HOTA %|Multi800 IDF1 %|
|---|---:|---:|---:|
|MOT|0|7.673|7.984|
|MOT+RMO|0|18.255|21.471|
|MOT+OV|0|15.794|16.770|
|All three|7.860|18.674|19.299|

Triple training transfers to the single expression, but every multi800 arm is
below same-candidate Simple31.877HOTA/37.685IDF1. These matched1200update
arms differ in budget/initialization from the older shared broad native model.
 NovelClsA0 in every subset. Complementary MOT400/600 controls have
HOTA13.317/13.979 with exact actual sample/loss prefixes versusMOT1200; total
compute still differs from the multitask arms. No isolated gradient-interference
or universal transfer conclusion.

All five frozen same-input policies completed the original DT1203 source:

|Association/auxiliaries|HOTA %|IDF1 %|IDSW|FP|FN|
|---|---:|---:|---:|---:|---:|
|Learned/learned|26.316|30.633|165|579|1962|
|IoU/fixed|7.488|2.597|775|9335|257|
|IoU+cosine/fixed|13.311|9.841|553|9328|250|
|IoU+cosine/learned|13.009|9.688|545|9329|251|
|Learned/fixed|24.283|29.761|155|608|2139|

Same semantic functions/input information are retained, but confidence and
history evolve with each policy. Learned policy also suppresses many proposals;
this is not isolated identity-ranking evidence, an optimally retrained conventional
matcher, or unique JEV proof. The same five policies have now completed both
full800RMOT scopes; CPU learned-reference official metrics exactly match the
fresh GPU5 runs. All25original policy units now validate, including raw and
aliasTAO40. AliasTAO TETA learned21.773 versus IoU26.928; rawTAO learned14.517
versus IoU13.995. NovelClsA remains0. The two proposal protocols are separate.

The context study also completed:12units, four32references and eight new64/128
runs. These are actual independently evolving tracking trajectories, not frozen
ranking coverage estimates.

|Context IDs|DT learned HOTA %|DT fixed cosine HOTA %|AliasTAO learned TETA %|AliasTAO fixed cosine TETA %|
|---|---:|---:|---:|---:|
|32|26.316|13.311|21.773|23.913|
|64|26.316|13.289|23.100|26.099|
|128|26.316|12.491|24.749|29.062|

Larger context helps this TAO scope but does not establish a learned-association
advantage. Context64/128 exceeds the training32-option distribution and changes
semantic normalization and histories. Fixed32hybrid retrieval remains running. Hybrid
keeps16geometry IDs then fills32from native appearance; real251-ID differential
checks preserve old defaults exactly. No newGPU/feature sources or speed claims.

## Full fine-tuning and Ground adaptation

Original One500 and paired teacher600/native600 additional stages completed.
Both600FP32/Adam/scheduler/RNG states are saved; both real native multimodal
repeated reload checks have probability delta0. Teacher restored450 and native
250 exactly. Teacher replayed475loss differs by.001734814; native has no
unsaved-log overlap. Neither is called bitwise training replay. Resumed tail
timing is not full-stage throughput.

|Full checkpoint|MOT4 HOTA %|SingleRMO8 HOTA %|MultiRMO12 HOTA %|TAOBase4 TETA %|
|---|---:|---:|---:|---:|
|Before500|30.868|0|0|11.981|
|Teacher600|30.601|0|9.945|52.181|
|Native600|32.236|0|11.214|47.682|

All12same-policy controls validate. NativeNoul membership>=.5 applies to all
three RMO arms; they cannot be compared to older ungated scores. Base4 has
only the person/baby Basecanonical class; ClsA100 is not Novel recognition.
Native history is not universally better. These remain short scopes.

Ground200 trained the actual native projector (27267584parameters), with
native dense/referring plus identity losses; all569native tensors export with
other weights untouched. Ground320adds actual absence supervision. Isolated
original/200/200repeat/320 native controls preserved seeds/request order and
repeat raw outputs exactly, but showed mixed MOT/RMO behavior; no promotion.
Original and Ground200 full9native retention evaluates all91124released rows
per model; neither is claimed complete while native clients remain active.

Required localization-only200 control has actual112rows/56pairs/200indices
matched to completed Ground200, same seed42/lr2e-6, identity-loss coefficients0.
It waits only verifiedGPU6retention clients/service release, then trains/exports,
runs96ordered native frontend frames and18pretrained/native600Full composition
controls. Composing adapted weights is not joint-gradient optimization. No
localization-only accuracy/retention score is claimed yet.

## History/update supervision and long gaps

One100realrollout->500SFT and Compactjoint1200real72-frame rollout->native1200
provide offline iterative own-state training. Recorded states are causal and
uncorrected byGT. They are not per-gradient on-policy resampling; the latter is
a research direction, not an additional invented acceptance gate.

Fit72update audit reproduces every original label. Of28positive labels,12have
unknown GTbirth ownership; none has two known conflicting identities. The one
known conflicting write already has update_target0. Unknowns must not become
fabricated negative labels. The initial hypothesis that wrong positive update
labels explain failures is unproven; original packs/models are unchanged.
Authoritative audit: `diagnostics/causal_update_supervision_fit72_v3_known_conflicts/audit.json`.

Complete DT1203 memory interventions have HOTA11.441(frame-only),16.394(recent),
14.643(EMA),24.245(anchor/always),26.316(anchor/gated3s),18.552(anchor/gated30s).
They are frozen policy interventions. The full singleRMO800diagnostic has two
natural candidate returns beyond3s; bothSimple/Compactrecover0under immutable
birth-anchor diagnostics. This does not prove successful long recovery; GTgap
and candidate-gap scopes differ. The registered natural MOT20challenge still
requires actual inference. A single registered owner2126039 waits only original
GPU7 native-retention release. It will use every1657validation frame,25Hz, with
8same-source memory/Simple controls and a3GiB packed native ROI archive cap.
Registered ID854's29.76s low-visibility interval retains743intermediate annotated
frames; it is not yet a measured missing-candidate interval. No artificial
occlusion or metric filtering is used.

Real21frame packed-storage regression preserves266candidates, the actual rejected
empty frame, every FP32ROI value, predictions, full state and next-frame recovery
exactly:14998172JSON-equivalent bytes versus3409159packed bytes. This is a
storage functional result, not new model accuracy or a full-video speed claim.
Existing conservative cache upper22471148986+3GiB+32MiB regression reserve is
25725928890bytes, below25GiB. The default older JSON archive remains unchanged.

## Active execution and remaining work

At most physical GPUs4/5/6/7combined; other users and stopped oldU2 queues
remain untouched. Temporary feature cache cap is25GiB; all data/checkpoints/
valid results are preserved. Historical verified old cache cleanup freed39.89GiB;
see `RESOURCE_CLEANUP.md` for limits of the original resource snapshots.

|Device/queue|Work and authoritative owner output|
|---|---|
|GPU4|Fullnative600/before500/teacher600DT1203; `training/onejev_full1203_controls_v2_wait_fix`|
|GPU5|Full RMOT matched controls; `multitask_runs/full_rmot_shared600_v1`, then fresh pairedE2E|
|GPU6|Ground200nativefull9, then `ground_adaptation/localization200_control_v1`|
|GPU7|OriginalGroundnativefull9; summaries under `ground_retention/reattached_full9_owner_v1`|
|CPU|Original25policies, all24task-subsets and12context units complete; hybrid study running|
|GPU7 after retention|Natural MOT20-05 validation1657, then8CPU memory controls; `natural_memory/mot20_05_full_validation_v1`|
|Completed CPU|`compact/full_rmot_affinity800_v1`:all10full800matched-semantic controls|

The v1Full1203waiting manager failed before any inference when a PID handle
shadowed training handles. Minimal fix and actual completed-owner CPU regression
passed; v2nowruns unchanged data/checkpoints/metrics. Failure/logs remain.

Remaining work is to finish and validate these exact full-video, retention and
paired resource studies, complete the required visual-only/composed adaptation
comparison and natural long-gap experiment, and consolidate the scientific
input/budget controls without inventing missing evidence. Negative results are
retained. Pilot evidence never implies completion of the full research goal.

## Complete full800 matched-semantic policy comparison

Both CPU references exactly reproduce the corresponding freshGPU5 official
accuracy. All ten runs retain their800frames,1412/2208native candidates and
36/19rejections; same frozen semantic functions, no GT correction or speed claim.

|Association/auxiliaries|Single HOTA %|Multi HOTA %|Multi IDF1 %|Multi IDSW|
|---|---:|---:|---:|---:|
|Learned/learned|6.025|19.802|21.008|165|
|IoU/fixed|7.766|32.813|41.432|79|
|IoU+cosine/fixed|5.538|16.622|14.009|105|
|IoU+cosine/learned|6.009|22.035|23.803|119|
|Learned/fixed|6.636|19.558|21.064|157|

The same learned semantic functions plus fixedIoU outperform the current
learned association on this complete multiple-target expression. Raw native
appearance cosine does not help this geometry baseline. This remains a frozen
policy comparison, not isolated JEV uniqueness or an optimal conventional
training claim; states/action priors and future semantic outputs differ.
No model promotion. Receipt:
`outputs/groundjev/compact/full_rmot_affinity800_v1/validation_and_summary.json`.


Latest Full single800 controls are also complete (same1412native proposals,
800decoded frames/36parser-rejected frames, unchanged nativeNoul.5 semantics):
pretrained/500/teacher600/native600 HOTA16.119/0/6.847/2.506%,
IDF1 17.391/0/4.839/2.062%, FN16/18/14/17 out of18GTboxes.
Pretrained recovers only2GTboxes, so its higher HOTA does not imply adequate
recall. Training does not improve this full single-expression scope. Multi800
Full controls still run; all Full replay speeds exclude perception.

Curated official-artifact receipts: `outputs/groundjev/compact/` under
`fixed_affinity_controls_v1/validated_summary.json`,
`task_subsets1200_cpu_v2_feature_sources/validated_summary.json` and
`context_sizes_v1/validated_summary.json`.


Input evidence14.2 now has a registered matched retraining study,
`configs/groundjev/compact_input_evidence1200_v1.json`. Two new1200step CPU
models mask nativeROI and runtime specification conditioning, respectively;
the full-input arm reuses completed all_three1200 weights. All modes retain
1146511resident parameters and the output-only class vocabulary text head.
The same original native semantic frontend and native softteacher targets remain;
this is association-input isolation, not language-free perception or isolated
JEV proof. Initialization tensors and all1200sample IDs must equal baseline.
Three modes each evaluate fullDT1203/rawTAO40/single800/multi800. One serial
owner waits hybridCPU completion; no new GPU/feature cache. Real three-task
isolation/default-exact, two-step reloads and two real21frame scheduler controls
passed; no full-video accuracy is claimed for these new modes yet.

Actual completed single800Full timing logs show all3200requests use batched
forking. Pretrained/500/teacher600/native600 average prefix tokens are
1347/3987/8235/5474 and branches7.93/14.20/24.38/17.72; forward totals are
142.531/344.670/787.309/481.178s. Different causal histories and concurrent
loads prevent isolated causal speed attribution. No evidence of universal
sequential fallback; no native-engine optimization protocol changed.


PretrainedFull multi800 also completed: HOTA8.621%, IDF1 7.635%, IDSW215,
FP214, FN1766 out of2112target boxes, on the same2208native candidates and
19parser-rejected frames. Remaining trainedFull multi800 arms still run.
Replay speed is not full native perception-inclusive speed.

Saved-output diagnosis at the unchanged membership threshold.5, with the same
1132one-to-one physically aligned query-positive candidates:

|Completed method|No physical assignment|Assigned then membership rejects|Actually emitted aligned positives|
|---|---:|---:|---:|
|PretrainedFull|105|687|340|
|SharedCompact|4|71|1057|

All2208candidate outputIDs/boxes reconstruct exactly from saved actions and
membership, without new model calls or corrected states. Head recall is
31.625% versus93.728%. Full surviving-correct-anchor option coverage is99.149%
but existing-association accuracy10.207%; these causal event sets differ between
models. Candidate-aligned funnels do not decompose officialFN across HOTA
thresholds; unknown/unmatched proposals remain unknown. Wait registered600
controls before choosing a training change; no threshold tuning or extra runs.
Evidence: `diagnostics/multi800_full_compact_output_funnel_v1/summary.json`.
