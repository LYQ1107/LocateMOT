# GroundJEV-MOT current evidence — 2026-10-16

## Opt-in post-assignment native write path — 2026-10-16

Implemented `AssignedIdentityUpdateDecision` and explicit
`run_video.py --assigned-identity-updates` (Full v3/gated only, default OFF).
The original association/lifecycle pass and global matching run first. Before
any current-frame track mutation, the new path asks native Noul questions using
the SAME final-ID builder as the prepared613-row component. All assigned existing
IDs enter the request; there is no runtime GT filtering. These answers govern
existing long-memory writes at the unchanged.5threshold. Original observation
Q3 probabilities remain logged, and NEW initialization, boxes/recent memory,
association, lifecycle, RMOT membership and category policies remain as before.
The category wrapper forwards the callback for this explicit backend. Resuming
an existing run requires its recorded observation/final-ID protocol to match.
Current registered workers do not enable the new flag.

CPU execution validation completed(session64482exit0): the parent tracker blob
`9f092b04` and new default path reproduce all three saved complete Full MOT
trajectories exactly:1,203frames/13,664native candidates/all27rejected frames per
arm, identical outputs/per-frame state, and parent/current final memory equality.
Native original features and actual images are used; no new model forward.
Visible-track lifecycle probabilities absent from traces are not invented as
original full-memory truth. This is a frozen-decision compatibility check.

On the native600fit48original prefixes,96single-frame branches with explicitly
scripted gate answers0/1 exercise the new path:1,316questions,150pre-write calls,
all658existing assignments per gate value including all45unknown/unmatched.
All1,316training/runtime states/questions match after the declared identity
alias mapping;150historical crop-sheet pixel comparisons pass with saved aliases.
Long memory and last-reliable timestamps hold/update as scripted, original
association outputs remain identical, and category callback forwarding passes.
This is execution/schema/pixel evidence, NOT native second-pass model inference,
a causal continuous rollout, trained gain or a new tracking score.

The new questions still require matching SFT, balanced known positive/negative
support, fresh native-MM reload and controlled continuous three-task evaluation.
No new GPU job or feature archive was launched. All new small receipts fit the
already registered16MiBreserve; conservative upper26,443,764,493<25GiB. Evidence:
`outputs/groundjev/validation/assigned_identity_runtime_v1/{owner_state,validated_summary}.json`.

## Final-assigned-ID update component — 2026-10-16

Implemented a separate question builder, not enabled in the current tracker:
`locatemot/groundjev/assigned_identity_update.py`. It preserves the original
causal visual state and adds all actual existing-ID assignments from final
global matching, including unknown/unmatched assignments. Each new Noul question
explicitly asks whether this candidate should enter that specific identity's
long-term memory. No GT eligibility controls which assignments enter the input.
The builder is now shared with the opt-in post-global-matching/pre-write path
above; matching SFT and fresh causal tracking evaluation are still required.

Prepared the component from the already-completed native600actor's48legal-fit
frames/1,184native candidates. All candidates are registered:82NEW,444without
final ID,45known-consistent existing,568known-conflicting existing,15unknown
birth and30unmatched existing. Actor IDs map consistently to saved randomized
history/crop aliases using actual shortlist order. The613known assignments
supply29yes/584no labels, using immutable birth-ID consistency and the original
visibility>=.6proxy. The45unknown/unmatched existing assignments have no
supervised target and remain present in the input assignments. The300old
candidate-observation Q3yes/new assigned-IDno cases are different questions,
not silently changed labels. All original questions/targets and actor states
remain untouched; no GT state correction, new image artifact or model call.

All613new questions retain exact old state apart from the declared final
assignment, retain exact media references and name the actual selected input
alias. Twelve independent audit conflict examples reproduce exactly. Full CPU
native planning exits0:613rows,0drops,max27,234tokens,total13,305,265tokens,
actual plan_epoch17.393s;154updates is only a hypothetical4-row component epoch.
All component files are under16MiB; conservative total26,443,764,493<25GiB.
This tiny, highly imbalanced component is not trained or deployed and does not
establish accuracy, runtime/training equivalence or ordinary-model performance.
Current paired600training/questions/loss and four-scope control queues remain
unchanged. Broader balanced data and matched post-assignment runtime are needed
before a future causal comparison.

Evidence: `outputs/groundjev/training_data/assigned_identity_update_component_v1/{validated_component_summary,validation,files_manifest}.json`.

## Three-arm final-assignment memory diagnosis — 2026-10-16

Read-only full-video diagnosis now includes all three completed Full arms,
not just MOT500/native600. All13,664candidate events per arm reconcile with
saved final assignments and actual0.5-gated UPDATE/HOLD decisions; every
historical v1 statistic for the original two arms reproduces exactly.

| Completed arm | Q3 AUROC, known consistent vs conflicting assignment | Consistent writes / assignments | Conflicting writes / assignments |
| --- | ---: | ---: | ---: |
| MOT500 | .513176 | 192/214 (89.72%) | 594/696 (85.34%) |
| native additional600 | .485748 | 136/149 (91.28%) | 465/514 (90.47%) |
| teacher additional600 | .499661 | 56/82 (68.29%) | 152/216 (70.37%) |

Teacher's assigned-identity probability AUROC is.467395; its argmax-agreement
AUROC is.479336. Requiring final-ID=argmax on already-observed writes would
hold51/56known-consistent and127/152known-conflicting teacher writes, retaining
only5consistent/25conflicting writes. This is not an accepted policy or causal
rollout. These values describe each model's own endogenous known-anchor event
set, not a same-event training effect; unknown-birth and unmatched candidates
remain separate rather than being labeled negative. No universal memory-write
precision, threshold sweep, modified prediction or new HOTA is claimed.

All three trajectories support the diagnosis that the current pre-assignment,
candidate-observation Q3 does not discriminate correctness of the ultimately
assigned identity. Explicit final-ID-conditioned training/runtime supervision
remains a separate hypothesis; the ongoing paired history-refresh study keeps
its original questions, loss and runtime unchanged. Evidence:
`outputs/groundjev/diagnostics/full_mot_assignment_memory_v2_all_three/summary.json`.

## RMOT expression component — 2026-10-16

Prepared and validated all4,350legal-fit expression files on the same72
recorded native frames/165candidates, retaining all8rejected frames and34,800
query-frame contexts. The87,721questions comprise84,062candidate membership
and3,659empty-frame presence questions; candidate positives6,873 versus77,189
negatives. Only2,213expressions have a positive target in these sampled frames;
2,137do not. Per-expression positive/negative coverage is preserved for a
future balanced mixed-task curriculum. This expands language coverage, not
physical-image coverage or identity-history supervision. Proposals still come
from their original recorded primary-expression DLM requests; no detector was
rerun for the new expressions, so this is counterfactual membership supervision.

The initial generator stopped on missing same-frame boxes for expression IDs:
60contexts/15expressions/four unique video-frame-ID references, all in fit0017.
An early-preflight rerun correctly held before exporting any rows. Pinned
TempRMOT6a65640's unchanged `_pre_single_frame` was then executed on198actual
cases, including all60affected contexts: it iterates released boxes and marks
membership by expression ID; it does not invent boxes for unboxed IDs. Six
legacy empty-frame missing-label-file cases remain explicitly recorded. The
first official-reader verification failed because our symlink setup gave only
the image path the required KITTI token; fixing the label alias too selects
the official KITTI geometry branch. All failed attempts are preserved locally.

The final component explicitly records the official box-projection rule and
all60affected contexts; none were skipped, interpolated or silently relabeled.
All346legacy samples reproduce exact state/question/target/task, with136native
input-media pixel comparisons and200legacy query-media pixel pairs checked.
Full native CPU planning retains all87,721rows, max2,301tokens,120,061,085total
planned tokens,0overlength drops; actual `plan_epoch` took111.924s and peakCPU
RSS2,068,492,288bytes. The cached native reporter's0s and default4-rank label
are not full tokenization time or GPU use: actual world size was1onCPU; future
entry explicitly sets PLAN_WORLD to match. The21,931steps are a hypothetical
4-row accumulation epoch, not an optimizer run. Gzip training text3,156,280bytes;
all generated/failed-attempt real files total about7.98MB, within64MiB reserve.
Combined conservative temporary upper26,426,987,277bytes remains below25GiB.
No new image/feature archive or GPU training was created. Existing paired600
training and queued four-scope controls remain unchanged; ordinary-model-level
three-task performance and large-scale joint foundation training remain unproven.

Sources: `configs/groundjev/rmot_query_diversity_component_v1.json`,
`tools/groundjev/{prepare_rmot_query_diversity_component,check_rmot_release_box_projection,plan_rmot_query_diversity_component}.py`,
`outputs/groundjev/training_data/rmot_query_diversity_component_v3_official_box_projection/validated_component_summary.json`,
`outputs/groundjev/validation/rmot_query_diversity_full_cpu_plan_v1/owner_state.json`.

Independent worktree `/data2/user/LocateMOT_GroundJEV`; original LocateMOT data
and preserved U2 results remain under `/data2/user/LocateMOT`. The full goal
remains active. These are registered legal validation scopes, one seed; no
complete-dataset result or successful unified-model promotion is claimed.
Earlier snapshots/failures remain in Git, `STATUS_HISTORY_20261015.md` and
`research_log.md`. Official artifacts remain authoritative.

User clarification2026-10-15: the final purpose is ordinary-model-level
MOT/RMOT/OVMOT performance, not completion of the Pilot or experiment queue.
That target is NOT achieved. ByteTrack,TempRMOT-v2 andOVTrack are the initial
conventional references; see `ordinary_model_targets_v1.json` andFINAL_GOAL.
Published figures require matching scope/splits before formal use. In
particular, released TempRMOT training includes all three localV2validation
videos, so its released weights cannot establish unbiased held-out parity
there. The old TempRMOT V1 sanity report is neither a currentV2validation
result nor proof that the new performance goal has been reached.

Currentnative600 actual own-history refresh completed with its original
blocking owner exit0 after the successful Full1203cohort. All48fit frames/
1184original DLM candidates and perception records exactly match the original;
5741supervised rows,165806581exported bytes, no new DLM calls or GT-corrected
actor states. No automatic further training; export cap512MiB keeps the registered
combined upper26,262,799,802bytes below25GiB. Review actual existing-identity
and conflicting-anchor support before changing Q3 supervision. Plan:
`configs/groundjev/native600_fit_rollout_refresh_v1.json`.

Fit-only audit now reproduces all1184observation Q3 targets on refreshed
histories. Actual existing-ID assignments rise from0(original100actor) to658:
45known-consistent,568known-conflicting,15unknown-birth and30unmatched. Of
these, actual existing-memory UPDATE counts are12/41/12/20respectively.
There are82NEW initializations and444events without an assigned identity;
all counts conserve1184. The original observation-question target is yes for
300known-conflicting assigned events, which does not make that original
candidate-only question an assigned-ID correctness label. Unknown/unmatched
cases are not relabeled negative. Audit:
`diagnostics/full_fit_update_supervision_native600_v1/audit.json`.

Refreshed versus old actor history SFT is now running serially onGPU4,
retaining RMOT/OVMOT content and the existing runtime policy. Each arm adds
600updates/2400sampled rows from the same native600FP32 masters:320tensors/
752393024trainable parameters, first-arm initialization difference0. Each
arm uses800MOT/800RMOT/800OVMOT rows; only400native MOT histories change,
while400teacher MOT and all semantic rows remain equal. Actual row-to-step
membership and question families match; native length ordering within steps
can differ. CPU plans drop0rows, max25998/27960tokens; not token/FLOP matched.
New JSONL48712542bytes; combined registered upper25957225229<25GiB.
The first owner stopped before loading models because plan-only had created
args-only directories; these are preserved under the CPU validation, and
the corrected live owner is `training/full_history_refresh600_pair_v2`.
Original blocking session21963, PID2448456/start45150423; no subsequent
tracking gains are claimed. Explicit assigned-ID Q3 would be a
separate declared training/runtime protocol, rather than silently replacing
the current observation labels.

Preregistered post-training controls are queued through two CPU pidfd owners:
GPU4old worker waits for both training arms and native-MM reloads; GPU5worker
waits for the original natural-study queue to exit, fills the missing initial
native600 Full TAO40baseline, then waits for training and measures refreshed.
Each trained checkpoint covers DT1203, RMOTsingle800/multi800 and rawTAO40;
the initial checkpoint's already-completed exact MOT/RMOT units are reused.
All13664/1412/2208/35228native candidates and27/36/19/0rejected frames retain
their original denominators. TAO uses all1203inputclasses/UNKNOWN and the
annotation clock, not physical video seconds. No new DLM/ROI archive or
per-task checkpoint selection. Plan `full_history_refresh600_controls_v1.json`,
CPU source/real-baseline-validator evidence under
`validation/full_history_refresh600_controls_registration_v1`.
No post-training model calls or new accuracy results exist at registration.

## Complete DanceTrack video

Same original V1 Ground proposals, all1203frames of validation0004,
13664candidates/27parser-rejected frames; unchanged official TrackEval.
HOTA/IDF1 below are percentages, IDSW/FP/FN counts.

|Method|HOTA|IDF1|IDSW|FP|FN|
|---|---:|---:|---:|---:|---:|
|Simple|8.530|3.956|746|9334|256|
|Official ByteTrack association, explicit constant-score inputs|16.473|13.903|367|5234|395|
|Distilled Compact300|20.157|17.140|588|2129|951|
|Matched MLP300|11.197|15.118|161|549|3556|
|Same Set architecture, GT-only300|5.977|3.406|1322|5954|1796|
|New original U2 core/bridge300|1.243|0.625|250|455|4191|
|Shared Compact, broad native history|26.316|30.633|165|579|1962|
|Full MOT500, V3 native heads|7.848|5.056|3530|6387|765|
|Full MOT500 + own-history600, V3 native heads|8.703|5.826|3631|7516|416|
|Full MOT500 + teacher-history600, V3 native heads|7.577|5.015|3839|7683|384|

Shared Compact has1146511learnable resident parameters plus original frozen
OneJev CPU token lookup254279680parameters/508559360nominal bytes and external
Ground perception. It is not a replacement claim for the0.8B decoder. The
shared stage has a larger/different training budget than the matched300arms.
Lower IDSW alone does not establish improvement when FN rises.

ByteTrack uses untouched official tracker code
[`d1bf019`](https://github.com/FoundationVision/ByteTrack/tree/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99),
CPU-only with no weights/CUDA. All13664native boxes enter with explicit
constant1because DLM scores are absent; the low-score branch is empty.
Official CLI defaults .6track threshold/30buffer/.9match threshold are fixed,
giving20lost frames at20Hz (1s), versus GroundJEV3s. Native Kalman output
boxes and all returned active tracks remain, with no extra area/aspect filter.
All1203frames/27rejects and the exact unchanged input geometry are validated,
and MOT prediction text agrees with every saved frame output. This is a
same-proposal association control, not the full YOLOX+ByteTrack pipeline,
published-score reproduction, matched training budget or proof of unique JEV
benefit/parity. Compact has fewer FP but more FN; Fullnative600 is weaker
than this conventional association reference. Receipt:
`baselines/bytetrack_unscored_dancetrack1203_v1/validated_summary.json`.

Fresh pretrained Full V3 uses a different actual source: HOTA2.389 versus its
own matched Simple8.257. Actual unresumed native pipeline FPS.118560, under
concurrent workloads, excluding preloaded service/model initialization. Its
historical final-checkpoint timer definition is preserved. Full native600 and
MOT500 and teacher600 controls above each have all1203decoded frames, exact13664V1candidates,
all27parser-rejected frames, unchanged official metrics and final memory
validated. All three registered units are complete, without failed/pending
arms. Teacher600 lowers HOTA versus MOT500 despite fewer FN and has more
ID switches/FP; neither600branch establishes stable multi-task improvement.
These are replay measurements with no E2E FPS claim. Receipt:
`training/onejev_full1203_controls_v2_wait_fix/completed_units_validated_summary.json`.

Same-source posthoc diagnostics align4341/4586GT observations (94.658%) to
candidate boxes atIoU.5. MOT500/native600 correct-option coverage is98.603%/
95.021% of their respective4008/2912existing-anchor events, but actual correct
assignments are214/149 (5.339%/5.117%). Per-candidate argmax is correct254/178
times; global matching loses222/145 and rescues182/116, giving the same214/149
actual assignments. Low native identity ranking and global competition both
contribute; different endogenous event sets prevent a strict causal comparison.

Completed teacher600 has1588correct options among1833existing-anchor events
(86.634% coverage), but82correct final assignments (4.474%). Raw argmax is
correct77times; matching loses69and rescues74, yielding82. The same candidate
GT recall is94.658%; proposal misses alone do not explain this weak ranking.

Existing-memory UPDATE breakdown, excluding NEW initialization:

|Birth-anchor diagnostic|MOT500|Own-history600|Teacher-history600|
|---|---:|---:|---:|
|Matched GT, known consistent anchor|192|136|56|
|Matched GT, known conflicting anchor|594|465|152|
|Matched GT, unknown birth anchor|2545|2984|2962|
|Unmatched candidate, not universal negative truth|5812|6648|4977|

Known conflicting writes show real memory contamination in each run. Unknown
birth anchors and unmatched/ignore-region candidates are kept separate; they
are not relabeled wrong identities or supervised negatives. Counts preserve
the original analyzer, metrics and uncorrected causal tracking states.

Saved-decision diagnosis of the memory gate: native Q3 is asked per candidate
before global matching and does not name the final assigned identity. On
known-consistent/conflicting existing-ID assignments, Q3 AUROC is0.513(MOT500)
and0.486(native600). Native600 mean Q3 is0.852/0.849 for those two classes;
mean probability of the actually assigned identity is0.04269/0.04234. The
observation gate does not distinguish those identity outcomes here.

Requiring assigned-ID/argmax agreement on these frozen histories would hold
173/192correct and518/594conflicting MOT500writes, and111/136correct and399/465
conflicting native600writes. It discards most correct writes too, so this
heuristic is not adopted. These counts do not simulate future causal states
or establish HOTA gains. Receipt:
`diagnostics/full_mot_assignment_memory_v1/summary.json`.

Fit-only audit reproduces all1184original native Q3 labels on48legal training
frames and all2460eligible family-matched copies. The actual old100-step
native actor produced74NEW and1110events with no assigned identity, with zero
existing-ID assignments. Thus these native actor states contain no actual
conflicting existing-ID write contexts. Association/lifecycle supervision
targets still exist; this is not absence of association labels throughout the
training corpus. Q3 labels express visibility/correct-option availability,
not final assigned-ID correctness. These observations motivate refreshing
own-history support and an explicit assignment-conditioned gate; no existing
label, checkpoint or policy is silently changed, and no causal training-failure
claim follows from this eligible-corpus audit alone. Receipt:
`diagnostics/full_fit_update_supervision_v1/audit.json`.

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
training and its96native-frame/18Full controls have completed on GPU5. The
serial queue has advanced to fresh naturalMOT20; see the completed comparison
below. Original start/device evidence is preserved.

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

Broader legal TRAIN-source availability is now indexed without new model
calls/features: MOT40904frames/43videos (2664MOT17,33772Dance,4468MOT20),
RMOT1814local frames/9fit videos/4350expression files, andTAO18274annotation
images/500train videos. All indexed image paths exist. TAO provides52978
eligible Base annotation references across175frozen-vocabulary classes,
versus only9classes/64images in the existing category SFT. All175class
synsets/frequencies and52978image/annotation links validate against actual
train-only sources;6172nonBase and50outside-vocabulary annotations are not
supervised or converted into negatives. All1251images with no eligible Base
label remain indexed. Full1203input vocabulary stays unchanged.

The index uses33.85MB of metadata under a128MiB export reserve; conservative
registered upper26091442957bytes<25GiB. It is source availability, not rendered
JEV SFT, native proposals, actual training exposure, new large-scale training,
or joint Ground+One gradient optimization. RMOT local-frame listing is not an
external sequence-length completeness proof. Current600paired runs continue
unchanged. Index/coverage/validation live under
`training_data/broader_legal_scale_index_v1`; generator and source plan are
`build_training_scale_index.py` and `training_scale_index_v1.json`.

The175Base category component is now prepared and input-validated:52978train
objects produce105956native root/conditional-leaf questions across500videos.
Expanded JSONL1005240705bytes is stored as25820145gzip bytes; real-image crop
sheets are generated only when requested through the unchanged native
`NativeCategoryHead.crop_sheet`, with no PNG/ROI dataset cache. Typed questions,
1203input categories and conditional-root teacher forcing match the earlier
category component. No Novel/unknown negative labels are added; all18274images
are registered, including1251without eligible Base labels.

Actual regressions reproduce all215earlier Base object pixels/62sheets, native
question/pixel evidence on175class representatives and exact processor tensors
for root/leaf at both768x216/768x432sizes. All105956annotation/image/class/box
targets validate. Official CPU planning drops0rows, max4238tokens and total
315781843planned tokens; these are not training exposure. The reported26489
steps are hypothetical at4rows/update for this category-only component, not
an optimizer launch. Text rows still load into RAM (actual CPU peak3746652160
bytes); media generation is on demand. The native cached-plan reporter's1s
message is not complete tokenization timing or throughput. Existing normal
rows/images and native optimizer/rendering math remain unchanged; the separate
streamed-input entry records its IO version in checkpoint resume protocols.
Use this component within a future balanced MOT/RMOT/OVMOT curriculum, not an
automatic category-only training stage. Original600paired training continues.
Receipt: `training_data/base175_streamed_component_v1/validated_component_summary.json`.

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

Available completed full-task artifacts are now frozen independently of the
still-running9task owners: original6/9tasks (31,986native rows), adapted2002/9
(1,810rows). Every completed task has original=effective sample count and
provider-policy-filtered0. These are complete individual tasks, not completion
of either full9run or a result extrapolated from active response prefixes.

|Common completed grounding task|Rows|Original F1@.5 %|Adapted200 F1@.5 %|Delta pp|Original F1mIoU %|Adapted200 F1mIoU %|
|---|---:|---:|---:|---:|---:|---:|
|Dense200|200|78.986|78.679|-0.307|66.644|66.054|
|VisDrone|1610|46.125|42.145|-3.980|29.662|27.870|

Dense200 recall76.086→74.350% and precision82.116→83.544%; VisDrone recall
43.534→39.578%, precision49.044→45.068%. These measured grounding-retention
results do not establish tracking gains or overall retention. Originalqueue32
versus adapted-sync execution and recorded CUDA recovery prevent a bitwise/
paired-request or exclusively weight-causal interpretation. OriginalCOCO is
reused through its existing registered resume receipt. All original native
files, owner/service processes and metrics remain unchanged. Receipt:
`ground_retention/partial_completed_tasks_20261015_v1/summary.json` and the
eight exact native metric/sample-count extracts under that directory.

Localization-only200 and all96native-frame/18Full controls are complete and
official-artifact validated. Same112rows/56pairs/200actual draws/seed42/lr2e-6
as completedGround200, both identity coefficients0. All200logged total losses
equal native loss;27,267,584trainable parameters,16,164,180BF16elements changed,
checkpoint reload delta0. Export569native tensors; all other weights unchanged.
Localization repeat reproduces all raw-wire candidates/outputs on4/8/12frames.

|Native frontend|MOT4 HOTA %|Single8 HOTA %|Multi12 HOTA %|
|---|---:|---:|---:|
|Original|75.069|30.385|22.664|
|Localization200|84.099|18.776|23.355|
|Tracking200|83.467|24.876|23.675|

Full policy HOTA %, each frontend's exact candidates reused in bothOne arms:

|One weights + frontend|MOT4|Single8|Multi12|
|---|---:|---:|---:|
|Pretrained + original|33.913|0|17.636|
|Pretrained + localization200|36.858|0|1.754|
|Pretrained + tracking200|35.199|0|6.697|
|Native600 + original|31.932|0|7.605|
|Native600 + localization200|34.709|0|7.313|
|Native600 + tracking200|34.934|0|5.504|

ShortMOT improves after either visual adaptation; single-expression native
results worsen and Full misses all5single-scope GT boxes. Composition has no
general RMOT gain. These4/8/12frame controls do not establish complete-video
or dataset quality, full9retention, Novel capability or model promotion.
Separately adapted weights are composed, not jointly optimized. Frontend
adaptation changes proposals; exact input pairing holds within each twoOne
comparison. All new frontend/factorial runs useGPU5; reference tracking200
training usedGPU6 (same A800), so no bitwise cross-device trajectory claim.
Receipt: `ground_adaptation/localization200_control_gpu5_v1/validated_summary.json`.

|Queue/device|Remaining actual work|
|---|---|
|GPU4|Paired old/refreshed history sharedFull600SFT running; queuedold four-scope worker follows successful training/MMreload|
|GPU5|Fresh naturalMOT20 then8CPUcontrols; queuedbeforeTAO/refreshed four-scope worker follows original queue release|
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


The GPU5handoff executed after paired blocking session87824 exited0 and all
five owned model process groups were gone. Serial owner2207841 completed
localization child2213633 and advanced to naturalMOT20. Original retention6/7
and Full1203owner remain independent.

Progress visibility: `tools/groundjev/stream_owner_events.py` uses blocking
Linux inotify/pidfd events for the three existing owners and their registered
child state files. It prints actual state changes and process exits without
periodic filesystem/process polling or model calls. Original blocking commands
retain authoritative exit codes; an exit event alone is not scientific success.
Registration snapshot: `progress_events/remaining_v1/registration.json`.

Pilot documentation is now reconciled with the actual12execution deliverables
in usertask17; see `FOUR_GPU_PILOT.md` and
`pilot_milestone_20261015/evidence_audit.json`. This is a limited-development-scope
execution milestone, not completion of the full research goal or model promotion.
The durable original cleanup snapshots directly show14042MiB net all-device
memory decrease; concurrent other jobs prevent exclusive U2 attribution. The
historically reported17584MiB owned-worker inventory is kept as historical,
not reconstructed from those per-device tables. No old model is restarted.
