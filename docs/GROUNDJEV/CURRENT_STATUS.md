# Current status — 2026-10-15

This page reports current evidence; earlier running-state snapshots are retained
in Git history and research_log.md. The full GroundJEV research goal is active.
Independent worktree: /data2/user/LocateMOT_GroundJEV. Original LocateMOT data
and U2 results remain in /data2/user/LocateMOT.

## Completed complete-video evidence

All rows below evaluate all 1203 frames of legal DanceTrack validation video
0004 with unchanged official TrackEval. Fractions are displayed as percentages.
This is one video and one training seed, not a complete-dataset result.

| Original V1 native proposals | HOTA % | IDF1 % | IDSW | FP | FN |
|---|---:|---:|---:|---:|---:|
| Compact distilled set, 300 steps | 20.157 | 17.140 | 588 | 2129 | 951 |
| Same set architecture, GT only | 5.977 | 3.406 | 1322 | 5954 | 1796 |
| Matched distilled feedforward | 11.197 | 15.118 | 161 | 549 | 3556 |
| Simple reference | 8.530 | 3.956 | 746 | 9334 | 256 |
| Newly trained original U2 core + native ROI bridge,300steps | 1.243 | 0.625 | 250 | 455 | 4191 |
| Original pretrained Full V1 | 2.145 | 0.483 | 3792 | 9300 | 241 |

The three trained models preserve identical original native candidate boxes,
labels, scores and all 27 rejected frames. The Compact 20-frame resume is
explicitly retained. ROI replay and resumed runs have no detector-inclusive
end-to-end FPS claim. Compact and feedforward miss more GT than Simple;
lower IDSW alone does not establish better tracking.

The fresh uninterrupted native Full V3 run and its own exact-candidate Simple
control are also terminal and validated: HOTA 2.389% / 8.257%, IDSW 3665 / 764.
Fresh V3 detector responses differ from original V1; cross-group comparisons
are not identical-candidate controls. Full V3 measured end-to-end FPS 0.118560,
including detector RPC, frame reads, identity memory, writes, rendering and
encoding; preloaded server/model initialization is excluded. Hardware was
A800-SXM4-80GB under concurrent project and foreign workloads. This establishes
an actual slow pipeline measurement, not real-time performance.
Evidence: outputs/groundjev/compact/dancetrack0004_native300_v2_empty_roi_fixed/
aggregate_results.json and both full-video owner/official-metric receipts.

## Shared semantic Compact joint stage

The same persistent track core now has trained expression membership and
full-vocabulary category heads. The 1200-step balanced stage completed with
1,146,511 parameters, real nonzero updates and exact checkpoint reload on all
three tasks. Text inputs use the actual frozen original OneJev token embedding
means; this is not language-decoder reasoning or a replacement claim for Full.

Training retains all 149 native MOT groups and adds 48 fit-v2 RMOT teacher
frames plus 64 TAO train Base teacher frames. There are 256 object-group
examples. All 112 teacher frames remain declared, including five empty frames
without a supervised region example. Teacher GT proposals/history and ideal
update/reliability proxies are explicit; they are not native detector rollouts.
No Novel GT is supervised and no unannotated TAO region is made a negative.
All 1203 supplied LVIS categories remain available at inference.

| Registered short native-proposal scope | Before | After |
|---|---:|---:|
| MOT17 4-frame HOTA % | 66.911 | 71.556 |
| MOT17 IDSW | 1 | 3 |
| RMOT single 8-frame HOTA % | 8.932 | 0 |
| RMOT multiple 12-frame HOTA % | 0 | 0 |
| TAO Base 4-frame TETA % | 14.013 | 39.272 |
| TAO Base ClsA % | 0 | 58.286 |
| TAO Base AssocA % | 21.350 | 17.971 |

All paired candidates and native ROIs match exactly; all registered empty and
rejected frames and decoded video frames were validated. These are short
functional controls, not full RMOT/OVMOT or Novel-generalization results.
The stage is not accepted as a successful unified model: RMOT regresses and
OVMOT association declines despite the category gain. Evaluation-only diagnosis
finds all 6 single-query and all 8 multi-query covered candidate observations
suppressed by the learned membership head. Their probabilities are below .008;
the registered .5 threshold is unchanged. The training-fit diagnostic classifies
all96teacher candidate observations correctly (45positive/51negative); swapping
the training expression changes probabilities, but its target correctness was
not scored. Training fit does not establish native validation generalization.
These counts are candidate observations,
not unique GT coverage counts. No GT correction or threshold tuning was applied.
Evidence: outputs/groundjev/compact/joint1200_controls_v1/owner_state.json and
validation_and_diagnostic.json. Full 1203-frame joint-core retention evaluation also completed with all candidate
and ROI rows identical and1203decoded video frames. HOTA20.157->22.103%,
IDF1 17.140->21.413%, IDSW588->346, FP2129->1653, FN951->1485. The association
gain comes with more missed targets. All27original rejected frames remain in
the official denominator. This is a same-video retention result, not complete
RMOT/OVMOT success or a full-dataset claim. ReplayFPS22.747 excludes detector
and ROI extraction and must not be compared to nativeFull E2EFPS. Evidence:
outputs/groundjev/compact/joint1200_dancetrack0004_v1/owner_state.json.

## Current training and services

The original500 replay and all16 short official controls are complete (see
below). The earlier250BF16-only interruption remains preserved; it was not
used to fabricate an exact optimizer resume.

Both paired shared Full600 arms completed from actual500 FP32 masters with an explicit
new-stage optimizer reset and matched semantic exposure. The first owner and
children subsequently disappeared without a captured traceback or exit code.
The interrupted saved cursors were450teacher/250native; both final cursors are now600. Detached recovery restores the
fullFP32/Adam/scheduler/RNG/cursor state exactly: both restore receipts report
FP32 maximum difference0 and exact paired-index-shuffle-v1 protocol. Unsaved
teacher475 log rows remain in the interrupted archive and are replayed from450.
Execution-only changes detach children and write direct persistent logs.
Current owner: training/onejev_unified600_pair_v4_detached_resume.

A separate blocking owner waits for the verified teacher child to exit before
GPU4 reuse. It will validate real600 native MM reloads and score the same
native semantic policies for before500/teacher600/native600 on four registered
short scopes. The before500 MOT result is reused because its policy is already
identical. RMOT uses native Noul membership with unchanged .5 threshold; OVMOT
uses all1203 supplied categories and explicit UNKNOWN. Teacher600 same-policy four-scope scores have completed (table below); native600 scores remain pending.
Plan: configs/groundjev/onejev_unified600_controls_v1.json.

TAO Base/Novel63 perception is complete: all40frames,520native queries,
35228boxes,317scored/203parser-rejected query responses. Rejection denominators
remain unchanged. Simple tracking failed on native frame755 when a positive
subpixel-height box produced an actual59x0pixel PIL crop that PNG cannot encode.
The corrected memory format preserves the exact empty dimensions without
changing the box, score, metric or invented pixels. The real frame755 regression
passes. CPU recovery replays all28 complete causal decisions and must finish
all40 frames for both Simple and the existing shared Compact checkpoint.
Simple now completes all40 decodedframes and unchanged official TETA18.640,
Base23.3725/Novel9.1737, with all35228proposals and160GTboxes. NovelClsA0,
so this is not demonstrated Novel recognition. SharedCompact has now completed
all40 frames too: TETA14.517,AssocA5.1478,ClsA30.314; classification improves
overSimple while association and overall TETA regress. BothNovelClsA remain0.
Owner: multitask_runs/tao_base_novel63_full_v3_empty_crop_cpu.
No native perception is repeated; no CPU/replay speed is claimed as E2E.

The original and Ground200 native full9 client runners survived loss of their
outer owners. A new detached blocking owner follows those exact PID/start-tick
identities, starts no inference, and validates all native task summaries only
after terminal. It then stops only their verified owned service groups.
Owner: ground_retention/reattached_full9_owner_v1. Neither full9 run is claimed
complete before all9 native summaries and all91124 released rows validate.
Original GPU7 and Ground200 GPU6 remain separate native services; training uses
GPU4/GPU5. Foreign jobs and the stopped oldU2 queue are untouched. At most four
physical GPUs are allowed; temporary feature/text caches remain bounded to25GiB.
Datasets, model weights and valid optimizer-state checkpoints are preserved.

## Completed RMOT breadth and native-history intervention

Plan: configs/groundjev/rmot_native72_20261014.json. All9frozen fit-v2 videos,
8frames each. Actual native Ground supplied165candidates,49one-to-one GT-aligned
regions and8rejected frames; all72frames were retained. The Compact actor had
28contaminated history observations and was never corrected byGT. Counter
expressions reuse actual causal candidate/history snapshots without another
actor or detector update. Both comparison packs retain all original256groups;
their new groups are138broader teachers and142native own-history/counter groups.
Their serialized files total326,197,818bytes. Teacher proposals and IoU-derived
update/reliability quality proxies remain explicit, never native confidences.

Both models initialize from joint1200 and train1200additional balanced steps,
seed0/lr2e-4, with the same1,146,511parameters and exact3task checkpoint reload.
This compares broad teacher with broad teacher+native own history; initial
Compact300 has a different training budget/semantic parameter count.

| Scope (percent unless counts) | Initial joint1200 | Broad teacher | Teacher + native |
|---|---:|---:|---:|
| MOT17 4-frame HOTA | 71.556 | 71.359 | 63.077 |
| RMOT single8 HOTA | 0 | 20.921 | 0 |
| RMOT multi12 HOTA | 0 | 17.740 | 14.846 |
| TAO Base4 TETA | 39.272 | 46.279 | 56.733 |
| TAO Base4 ClsA | 58.286 | 67.000 | 100.000 |
| Complete DanceTrack1203 HOTA | 22.103 | 20.401 | 26.316 |
| Complete DanceTrack1203 IDF1 | 21.413 | 23.832 | 30.633 |
| Complete DanceTrack1203 IDSW | 346 | 151 | 165 |
| Complete DanceTrack1203 FP | 1653 | 615 | 579 |
| Complete DanceTrack1203 FN | 1485 | 2276 | 1962 |

Native own history helps this complete MOT video relative to the matched
broader-teacher comparison, while degrading short MOT and single-query RMOT.
Neither is a final unified-model success. All candidate boxes/native ROIs,
GT, frame denominators and official scorers stay identical. RMOT membership
threshold remains.5. Native mixed training produces6/6covered single-query
probabilities below.5 (range.362–.498), whereas broader teachers suppress2/6.
No threshold tuning or GT prediction correction was performed. TAO clip has
only the Base person/baby canonical category: Novel remains unevaluated.
The emitted labels include several categories; this is not a claim of100%
general open-vocabulary classification.

Native full-video inference failed at frame799 with asynchronous CUDA launch
failure. Preserved798complete rows and causal memorytimestamp39.9. A fresh
GPU7process passed actualframe799with finite outputs; CPU/GPU probability
difference1.15e-4 is recorded without claiming numerical identity. Accessible
NVML ECC counters were0/recovery actionNone; privileged kernel records were
unavailable, so the external root cause is not established. No GPU reset,
driver/model/GT/metric change occurred. Existing strict prefix replay restores
all798prediction/state rows identically, resumes799, and validates all1203rows
plus1203decoded video frames and27rejected native frames. Original failure
remains in owner_state.json; recovery_owner_state.json is separately terminal.
No end-to-end FPS claim on the resumed candidate/feature replay.
Evidence: outputs/groundjev/compact/rmot_native72_stage_v1/aggregate_results.json.

## Original Ground dense/referring benchmark inputs

Downloaded6.929GB of complete required archives directly without a proxy,
with3parallelCPUdownloadworkers. Pinned official Grounding-EvalData revision
772e34bb1505053ed4857bba4405ee7b308bbfb6; unmodified native extraction functions
and original data. Raw inputs live in/data3/user/Grounding-EvalData-data;
annotations/archive bundle in/data3/user/Grounding-EvalData-bundle. These are
raw datasets, separate from temporary model-feature budgets.

All4dense and5referring canonical task entries are prepared (91,124declared
evaluation rows), not the other21paper tasks. Unmodified native evaluate.py
dry-run with limit:null reports full scope and missing_inputs:[]. Its own
checkout needs a relative local recipe; copy the tracked recipe to a unique
local config. Resolve the venv executable lexically, not to its base-Python
symlink target. Original model files are immutable hardlinks within the native
evaluator checkout. No retention accuracy score has yet been computed and
none of these benchmark labels is adaptation supervision. Recipe:
configs/groundjev/ground_retention_dense_ref9_original.yaml. Evidence:
outputs/groundjev/ground_retention/dense_ref9_data/input_validation.json.

## Native Ground tracking200 execution and full retention baseline

Registered112native SFT rows/56two-image pairs from32MOT17/DanceTrack train
images and40fit-v2KITTIimages (five sequences, two expressions). Native-space
coordinate preflight keeps every selected target without clipping; MOT20
window69out-of-grid annotations are outside this explicitly new SFT corpus,
with existing evaluation labels/denominators unchanged. All112rows fit the
original native collator. First dense step proves separate nativeDLM,
GTidentity and real decoded-proposal identity gradients. Only27,267,584native
projector parameters train; original native language/vision remain frozen.

The borrowed servingTorch2.9.1runtime fails on varying-length batches: v1GPU7
cuBLAS at2; v2GPU6validsteps1–3, saved fullstate1, thenfailure4; synchronization
v3restoresactualstate1, validthrough20, localizesillegalmemoryaccess immediately
after noisycompiledFlexAttention at21. All failures/logs/states retained.
No GPU reset, seed/loss/data/mask change or sample skip. The isolated native
Py312 Torch2.10cu128/TF5.7/Triton3.6 runtime is installed; serving and OneJev
environments remain unchanged. v5 resumed1, saved55, failed before58. v6
specialized the same native FlexAttention operator to concrete shapes, saved185,
and failed before186. v7 restored185 with CUDA_LAUNCH_BLOCKING=1 plus boundary
synchronization and completed186–200. Asynchronous failure root cause remains
unestablished; passing synchronized replay is not proof that it is fixed.

The actual checkpoint chain has exactly200updates (100dense/100referring),
all56registered pairs visited, finite logged losses/gradient norms, complete
FP32/Adam/allRNG/cursor state at200, and native checkpoint reload difference0.
All failed attempts and the execution/library changes are retained; this is
not an uninterrupted or bit-identical cross-runtime run. FP32changed27,267,353
and BF16-visible changed16,204,590 projector elements. The complete569tensor
TrackProj200 bundle replaces only five projector tensors; all other native
weights, tokenizer and configuration are unchanged. Training completion alone
does not establish dense/referring retention or tracking improvements.
Existing112rows include0absent expression targets;40realfit absent-target rows
on the sameimages remain separately registered for a later explicit curriculum.
Evidence: native_tracking200_v7_boundary_trace/complete200_validation.json,
complete200_history.jsonl, owner_state.json and export_receipt.json.

Original benchmark nativeCPUeval environment has been installed. First full9
baseline had9client initialization failures from inherited HTTPproxy configuration,
before any model score. Task-local proxy vars are now cleared. Fresh v2owner
has launched all original canonical4dense+5ref tasks, limit:null (91,124rows),
with unchanged original scorer and native decode profiles. PhysicalGPU7owned
originalGroundservice; local evaluatorCPU-only. Originalfailedrun remains.
The exported TrackProj200 full9retention owner usesGPU6/API8102 and separate
rendezvous ports22345+, with the identical canonical tasks/data/limit:null/
concurrency2/originalscorers/native decode profiles. ExistingOne training4 and
fresh isolated checkpoint controls5 keep the project within4/5/6/7. Both full-suite results are pending. Fresh48frame MOT/RMOT functional
controls through these services also run with the unchanged Simple associator,
parser/tracking_dense/seed42 and all empty/rejected frames; this remains
separate from full original benchmark retention.
Evidence roots: outputs/groundjev/ground_adaptation/native_tracking200_v*;
outputs/groundjev/ground_retention/original_dense_ref9_v1 and v2_direct.

## Earlier verified foundation and remaining requirements

Old U2 owners/queues were stopped and 39.89GiB of regenerable features removed;
raw data, models and results were preserved. Both official original model
bundles and isolated DLM/OneJev environments are installed and verified.
OneJev SFT100 completed with actual changed weights and exact multimodal
reload, but its controlled tracking results regressed. The separate genuine
native-Ground own-history corpus completed 48 train frames and 5661 rows; the
current OneJev500 replay still uses its declared teacher-proposal corpus.

Native Ground projector adaptation completed three warmup steps and 48-frame
fresh dense/referring retention checks; formal tracking adaptation and full
original Ground benchmarks remain incomplete. The frozen visual temporal
adapter completed200 steps without proving a tracking gain. Native category
and earlier Full controls remain preserved, including failures and collapses.

Still required: complete formal OneJev500 and actual reload/before-after
tracking evaluation; adaptation using native detector-rollout training data;
formal Ground tracking adaptation and full dense/referring retention;
shared model RMOT generalization, Base/Novel OVMOT and complete legal scopes;
long-memory/reappearance/NO_MATCH and permutation ablations; controlled joint
negative-transfer studies; matched Full/Compact/MLP/SetTransformer accuracy,
end-to-end speed and resource comparisons; and continued GitHub publication.
Pilot and these short controls do not complete the research goal.

## Native Ground200 actual short tracking results and next curriculum

All48native frames and six rendered videos validated with unchanged official
TrackEval, Simple associator, parser/tracking_dense/seed42. This is a fresh
before/after native detector comparison; changed candidates are expected.

| Scope | Original HOTA % | TrackProj200 HOTA % | Original IDSW | Adapted IDSW |
|---|---:|---:|---:|---:|
| MOT17 four frames | 75.250 | 84.553 | 2 | 0 |
| Referring single eight frames | 26.866 | 25.277 | 2 | 0 |
| Referring multiple twelve frames | 24.780 | 23.959 | 0 | 2 |

MOT FP16->2/FN24->13, but single referring empty-frame output count rises3->8
on the same three absent frames. Multiple referring FP7->8/FN25->22. All
empty/rejected frames stay in official denominators. No full-dataset accuracy
or concurrent-service speed advantage is inferred from these short controls.
Evidence: projector200_tracking_controls_v1/aggregate_results.json.

Hypothesis: missing native absence supervision contributes to referring false
positives; not proven as the sole cause. A separate120step stage initializes
exactFP32 projector200, explicitlyresetsAdamW atlr1e-6/seed42, retains all112
existing nativeSFTrows and adds40previouslyregistered realfit absences on the
sameimages (152rows/76pairs). Everypair is visited. No synthetic emptyimage,
validation-label training, threshold change or silent original200protocol edit.
Actual noisy FlexAttention forward/QKVgradient checks on the first real dense,
positive referring and absent pair use PyTorch's eager reference and consume
no RNG; they do not cover every layer/step. Original200model and bothfull9
retention owners are preserved. Initialv1 stopped beforefirstupdate because detached diagnostic backward
inherited native activation-checkpoint hooks. Only this probe graph is now
isolated with identity saved-tensor hooks; production checkpointing and
acceptance tolerances remain unchanged. Failure is preserved. Currentstage
owner: native_absence120_v2_reference_hooks. Three actual reference checks
passed: maximum BF16 output difference .005859/.003906/.005859 and Q-gradient
relativeL2 difference .043767/.033754/.036040 (K/V below .007), within the
unchanged registered criteria. These checks cover the first real layer of
each supervision type; additional120training/results remain pending.

## Completed absence120 and effective-seed isolation

Additional120finished from native200: all152rows/76pairs,40realfit absences,
finite logs, projectorFP32changed27,266,797/BF16changed8,256,449 relative to200,
reload difference0, peak18,414,616,064bytes. Full569tensor native320bundle
preserves all non-projector tensors. The original200full9owner remains intact.

Fresh shared-service200/320 HOTA: MOT83.737/83.402%, single18.406/20.046%,
multi23.496/24.817%. Empty-frame outputs stay7/3, singleIDSW1->2/multi2->4.
All48frames and six videos validate. Absence correction is not established.

Original native sampling uses a monotonically incremented request_index in
addition to suppliedseed42. Shared full9services consume unrelated requests;
effective stochastic seeds in preliminary checkpoint comparisons are therefore
unpaired. Preserve those real scores as descriptive observations, not causal
proof that adaptation produced their change. Newisolated96owner runs fresh
original/200/200repeat/320 replicas sequentially onGPU5, same24request order
and unchanged source/math/seed/scorers. Repeated200raw wire/candidates/outputs
must agree. Original and200full9owners stay running and unmodified; these full
benchmark results also need sampling-variance qualification before causal claims.

A finite OneJev500completion owner now waits for the specifically verified
current training process. After terminal500receipt it validates actual native
multimodal2reload and runs fourway Simple/pretrained/init100/500 matched-native
controls acrossMOT4/RMO8/RMO12/TAOBase4, reusingGPU4 only after training exits.
No oldU2queue or newtraining restart. CompleteDanceTrack before/after and
Novel-diverse/fulltask evaluation remain required subsequent work.
Evidence: native_absence120_v2_reference_hooks/training_validation.json,
projector320_tracking_controls_v1/aggregate_results.json, isolated96_v1,
and training/onejev500_completion_controls_v1. Live owner handles are preserved.

## Authoritative isolated native checkpoint comparison (completed)

All96native frames and12decoded videos validate. Freshordered200repeat agrees
exactly in rawwire, candidateboxes and causaloutputs for all24frames. Native
source and requestedseed42 unchanged. SameGPU5/server/orderedrequests, Simple
associator/parser/tracking_dense, originalGT and empty/rejected denominators.
Original24validunit reused after transientportfailure; allother variants fresh.

| Registered short scope | Original HOTA % | Native200 HOTA % | Native320 HOTA % |
|---|---:|---:|---:|
| MOT17 four frames | 75.069 | 83.467 | 83.402 |
| Referring single eight frames | 30.385 | 24.876 | 20.046 |
| Referring multiple twelve frames | 22.664 | 23.675 | 24.817 |

Single empty-frame outputs3->7->7, multiple2->3->3. MultipleIDSW0->2->4;
AssocA41.678->33.457->30.669%. MOTIDSW1->0->0, FN24->15->13 and FP16->4->2.
Thus detection improves in this shortMOTscope, singleRMOTregresses, and the
realabsence120intervention does not fix absent-target false positives. Multiple
HOTA/recall gain comes with weaker association. Neither adaptedmodel is
promoted as the shared default foundation. These are single-short-scope
observations, not full-dataset or generalization claims. Shared-service results
remain descriptive; use this controlled aggregate for these checkpoint effects.
Evidence: isolated96_v2_port_release/aggregate_results.json and repeat_checks.
All isolated replicas have stopped. Original/full9GPU7 and native200/full9GPU6
owners and originalOne500GPU4training remain intact. The new completion owner
waits the actual trainer before native500validation and matched-core controls.

## Original full9 terminal partial result and bounded-queue repair

Originalv2full9owner has exitedfailed: gam_coco completed all4952/4952canonical
rows (zero provider-policy filtering), native recallIoU.5=.744359 andF1mIoU=
.479781. Other8tasks failed HTTP503 because officialservice maxrunning1/
maxqueued2 cannot accept9clients*concurrency2. Originalservice stopped;
completedCOCOscores and partial responseSQLite caches remain intact. This is
one original canonical task result, not all9or adaptation retention gain.

Task-owned serve wrapper now permits boundedmaxqueue32 while preserving native
running1/source/model/decode/math, and full9client concurrency remains2.
Separateactual18concurrent HTTPregression is running onGPU7; its ownedservice
will stop before freshoriginal8unfinished-task resumption. NativeCLIrejects
reusedoutputroots; a task-local wrapper calls its same nativefunctions in a
freshroot, copies only verified all-sample completedtask result and SQLite
backup caches, spawns only unfinishedtasks and aggregates all original9.
No changedsamples/GT/metrics/limits/seeds, no claim of exactstochasticrequest-
counter continuation. Adapted200full9owner remains live; One500latestreport350.

## Verified queue32/whole-service cleanup and full9resumption

Actualqueue32v1regression failed because its APIreadiness probe reached old
orphanfrontends after newserverEADDRINUSE. Oldlauncher-only stoppedflags were
insufficient: API775339(group775337) and1051641(group1051638) remained. Their
commands/cwd/checkpoints and isolatedPGIDs were verified; priorTERM hadfailed,
so onlythoseownedgroups were killed. Newhelper waits/verifies allgroupmembers,
usesSIGKILLonlyafterfailed20secondTERM, and verifiesnonealive. No foreignjob
or driver reset. Regression now checks portavailability beforelaunch.

Freshqueue32v2all18simultaneous clients returnedHTTP200. Its entireownedgroup
1339199 and botholdgroups are verifiedempty. Original/evaldata/model math/
seed42/concurrency2 unchanged. Firstrecovery gates and a localNamespace naming
failure are preserved. Verifiedv3blocking owner launches original8unfinished
and adapted200all9unfinished units with boundedqueue32, freshnative resultroots,
SQLitebackup responses, and verified all4952/4952COCO result reuse. Native
wrapper uses originalpreflight/runner/scorers; no sample skip or metric changes.
Full9results still pending; exactstochasticcountercontinuation and adaptation
retention gains are not asserted. One500training/completion owner continues.
Currentrecovery owner: outputs/groundjev/ground_retention/full9_queue_recovery_v3_verified.

## Fixed-core memory intervention study (running)

Nativebroad1200 semanticCompact checkpoint is frozen for six full1203frame
DanceTrack0004variants onGPU5: priorreferenceanchor/gated3s, immediateframe-only
identities with recentfeatures, recent-observationfeatures with3sretention,
unconditionalEMA(.1)features, anchor/alwaysreplace3s, anchor/gated30s. Same
790,667,425byte originalnative candidate/ROI log, all27rejectedframes and
originalTrackEval/GT/thresholds. These are frozenmodel inference interventions,
not separately retrained policy competitors or full-dataset evidence.

Long-memory writes and reliabilitytimestamps are separate: forcedwrites do not
falsify lastreliabletime. EMAuses convexmixtures of actual nativefeatures,
not synthesizedimages. Twoactualnative frames verify expectedEMA, immutable
anchor and identical state-save/restore nextoutputs. Outputlogs omit only
sharedfeaturearray duplication; allboxes/labels/decisions and originalsource
reference remain. Resume rehydrates vectors from the retained exactsource.

Posthocdiagnostics add natural candidate-gap identityrecovery by timebucket,
correct-to-wrong/globalcompetition effects, and matchedGT-onlyNO_MATCH
precision/recall under explicitmissing-correct-shortlist rule. Undefined
support staysnull; ignored/unmatchedregions do not become falseNoMatchtruth.
These diagnostics never correct runtime state or replace officialHOTA/IDF1.
Owner: outputs/groundjev/compact/memory_full1203_v1; plan:
configs/groundjev/memory_full1203_20261015.json. Nativefull9recovery and
One500/completion-controls owners remain live and unchanged.

Prior preserved referenceposthoc has4341/4586one-to-one nativeGTcoverage but
only575/1768existingbirth-anchor associations correct; globalcompetition moves
94correctmodelchoices towrong and18wrongchoices tocorrect. Naturalcandidate
returns: ≤1s9/177samecorrectbirthIDs,1–3s0/4, >3ssupport0(null). Long30svariant
therefore testsretention/competition, not provenlong>3srecovery in thisvideo.
NO_MATCHdiagnostics are conditional on the stated immutablebirth-anchor rule,
not claims about ignoredregions or a replacement for officialIDF1/HOTA.

FullTAO Base/Novel video63 registration has40annotationframes and160GTboxes:
Base120(person/baby+knife), Novel40(peeler), allsynsets available within the
unchanged1203fixedvocabulary. LowestcompleteeligiblevalID selectedbeforemodel
outputs, allimagegeometryverified. NovelGTis evaluation-only. A finiteowner
waits currentGPU5memoryjobexit, runs all520nativevocabularyqueries/realnative
ROIs, then exact-proposalSimple/sharedCompact underoriginalTETA. Explicit
scored-query aggregation retains every nativeoutcome/rejectedquery and all
classes/frames, no GT filter/NMS/topK. SourceFPSunknown; annotationstep time,
no physicalFPSor E2Espeedclaim fromreplay. Owner:
outputs/groundjev/multitask_runs/tao_base_novel63_full_v1.

## Real native-proposal OneJev continuation preparation

Existing original500uses declaredGTproposals/fragments. Prepare a matched
additional200teacher-only/native-mixture study to isolate proposal-domain
history effects from extra trainingbudget. Equalinitial real500FP32masters,
seed0/lr2e-6/200steps/4rows/32768context/officialloss/frozenvision; explicitnew
optimizer/scheduler reset. No change to running500 and no stage200launch yet.
Native48trainframe corpus contains5661unique actualDLM/uncorrectedOne100history
rows, includingMOT20. Preserve all originals; identityweight4inbothsources.
Native/teacher eligibility is exactlymatched bytyped-question family and50/50
sourceweight (41762rows). v1preparation withunmatchedfamilyratios retained as
unused; v2is selected. No Novel/evaluation supervision or correctedactorstates.

Actualfull375checkpoint probe validates320FP32tensors/752,393,024trainable
parameters, maximuminitializationdifference0. It restores498,430,914elements
withadditionalprecision relative toBF16inference-export initialization; frozen
vision/config/tokenizer unchanged, CPUonly, oldlive500andweights untouched.
Officialv1CPUplan-only41762rows max26028/cap32768 andzero drops. The exact
familymatchedv2 gets its own officialpreflight. Launch requires actual500
completion and existing matchednativecontrol review. Nativecorpus actor is
One100; do not claim newlyadapted500onlinehistory in this preparedstage.

## Full memory intervention results (complete)

All7218nativeinput frames/six videos verified; samecheckpoint/candidates/ROIs,
all27rejectedframes pervariant and unmodifiedTrackEval. Freshdefault reproduces
priorbaseline predictions and probabilities exactly (maximumdelta0).

| Frozen-model memory intervention | HOTA % | IDF1 % | IDSW | FN |
|---|---:|---:|---:|---:|
| Anchor + learned gated update,3s | 26.316 | 30.633 | 165 | 1962 |
| Previousframe-only identity memory | 11.441 | 8.461 | 295 | 1548 |
| Recentobservation features,3s | 16.394 | 18.202 | 141 | 2754 |
| UnconditionalEMA features,3s | 14.643 | 16.876 | 119 | 2972 |
| Anchor + unconditionalupdate,3s | 24.245 | 29.818 | 154 | 2143 |
| Anchor + learnedupdate,30s | 18.552 | 19.849 | 148 | 2484 |

Keep currentgated3s default. It improvesHOTA/IDF1relativealwayswrite here, with
moreIDSW but fewerFP/FN. No generalization/independentlyretrainedEMA superiority
claim. Natural >3srecovery support remains0/null, so 30srun is retention/
competition sensitivity, not proof of longtermrecovery. ExcessiveNoMatch and
weakshortgap recovery remain problems. Evidence: memory_full1203_v1/
aggregate_results.json, validation.json and memory_comparison.svg. FullTAO
Base/Novelowner now reusesGPU5 after thisfiniteownerexit; no extraGPU added.

Selected exactfamilymatchedv2officialCPUpreflight completed:41762rows rendered,
maximum26028tokens/cap32768, zero droppedrows, PLAN_OK. Bothdomains have
identicaltyped-question eligibilitycounts; no nativeunique row omitted. Actual
newstage training remains pending500completion/controlreview. Full375FP32
initialization probe is numericalcorrectness evidence, not a newtracking gain.

## 2026-10-15: native full9 CUDA failure and synchronized resumption

The adapted200 queue32 evaluation terminated after the native scheduler reported
CUDA unspecified launch failure at 2026-10-15 00:04:41. Subsequent client
APIConnectionError failures are downstream; the asynchronous trace does not
establish the failing CUDA kernel. No complete task metric is claimed from
this failed attempt. All response SQLite caches and failure logs are retained.

The separate trackproj200_dense_ref9_v3_sync owner is confirmed live through
its existing blocking session. It resumes all nine unfinished tasks from the
latest valid response caches on GPU6 with task-local CUDA_LAUNCH_BLOCKING=1.
The checkpoint, native operators, scorer, data, task limits and seed are
unchanged; this is a diagnostic recovery, with no GPU reset or claim of exact
stochastic request-counter continuation. The original full9 GPU7 owner
continues independently; its parent may eventually report the earlier adapted
child failure, so that parent must not be used to restart the original task.

OneJev500 last observed progress is 450/500, with original training protocol
unchanged. The prepared matched additional200 study has not been launched.
TAO Base/Novel63 v2 remains live on GPU5 after the explicit None expression
argument repair; its metrics are pending. Six full1203 memory interventions,
all predictions and the comparison figure were published and remote-tree
verified in commit 085125569af96d47b89f2fd269c6501f02b0b28f.

## Real expression response diagnostic and Full semantic training gap

CPU replays of all20registeredRMOTframes/27native candidate observations
for joint, broadteacher and broadnative checkpoints reproduce recordedGPU
boxes/IDs exactly. MembershipGPUCPUmaximum difference is7.15256e-5.
Within captured groups and identity shortlists, inverse-aligned candidate/history
permutations differ by at most6.55651e-7 across all six probability heads;
this does not establish full-pipeline invariance across candidate group boundaries.

On the six primary-onlyGTcovered redcoatwoman observations, broadnative
correct-expression mean membership is .432221, versus .772220 for the wrong
arranging-automobiles expression. The unchanged .5 threshold rejects all six
correct expressions and accepts all six wrong counterparts. Posthoc counter
labels confirm these women are not targets of that car query. Language response
is present but direction is wrong here; numerical/order effects do not explain
this failure. These are conditional diagnostic observations, not official
accuracy metrics or evidence of broad generalization. Sources:
outputs/groundjev/compact/rmot_query_response_cpu_v2_geometry/report.json and
counterfactual_labels.json. The first diagnostic accidentally included floating
query scores in its geometry comparison; correctedv2 preserves v1 and reports
score differences separately.

The actual original50020881row curriculum and prepared41762row native mixture
contain only state.task=mot, person vocabulary and no expressions. Mixed question
families are not mixed tracking tasks. One500 is an identity/memory adaptation
stage; neither it nor the prepared200domain experiment proves unified Full
semantic training. Scope audit: validation/onejev_sft_task_scope_20261015.json.

Full now has an opt-in native-rmot-membership flag: an explicit pretrained
OneJev Noul question supplies expression membership, and the existing .5
output rule retains physical identity memory before filtering. The default
protocol is unchanged. Its genuine native-kernel4frame functional probe has one
blocking owner waiting for the existing500matchedcontroller to exit and pass;
GPU4 is reused only then. Owner: validation/onejev500_native_rmot_probe_v1.
Native execution and semantic adaptation evidence remain pending.

New native RMOT instantaneous supervision has346rows: all165recordedDLM
proposals under both registered expressions give330membership rows (58yes,
272no), plus16presence rows for the eight rejected/empty proposal frames.
All72fitframes/144query contexts remain; GTnever enters state and no new
One500actor-history claim is made. Initialv1 failed before its first row
because the independent worktree has no data symlink; v2 uses the registered
release root without modifying data or omitting samples. Official plan-only
accepts all346, max2293/32768tokens, zero dropped,457637tokens total.

New Base-category supervision uses the earlier registered64TAOtrainframes:
215GTteacher crops over9Base classes yield430root/leaf questions, with all1203
input categories retained and noNovelGTsupervision. GTteacher proposals and
conditional correct-root teacher forcing are explicit; runtime chooses its
root using the model. No unknown negative supervision was added. Official
plan-only accepts all430, max4233/32768tokens, zero dropped,1283259tokens total.
Shared category request builders reproduce prior root/leaf request text and
real crop pixels exactly on a real six-object input; capture-only routing
answers are not native inference or accuracy evidence. These776semantic rows
are prepared data, not trained weights or completed unified evaluation.

## Completed original One500 and registered paired Full600 training

Actual original500 finished500updates/2000examples at2026-10-15 01:30:56,
elapsed33469.233s. Its9,029,140,249byte fullFP32/Adam/scheduler/RNG/cursor
checkpoint and final BF16 model are saved. RealnativeMMtworeloads agree exactly.
All16matched short controls and112decodedvideo frames now validate, with
identical source proposals and unchanged official scorers.

| Scope | Simple | Pretrained | Init100 | One500 |
|---|---:|---:|---:|---:|
| MOT4 HOTA percent |74.881|34.394|19.740|30.868|
| RMOTsingle8 HOTA percent |24.228|0|0|24.886|
| RMOTmulti12 HOTA percent |24.315|22.272|0|21.425|
| TAOBase4 TETA |24.171|13.243|11.732|11.981|

MOT IDSW Simple/pretrained/100/500 is1/47/0/57. TAOClsA is0 for allfour
under this legacy native-label protocol. These are short pilot comparisons,
not full-dataset or Novel results. One500 improves over100 but does not
consistently improve over pretrained/Simple; no model promotion follows.
Review validation_and_summary.json in training/onejev500_completion_controls_v1.

Complete500CPUstage initialization verifies all320trainable tensors/752393024
parameters restored exactly from real masters.498431025FP32 elements retain
precision beyond BF16 export. All153visual tensors stay unchanged. Safetensors
contain two identical tied embedding/head exports; removing that alias gives
852985920unique model parameters (752393024trainable language,100592896frozen
vision),333632768changed BF16unique elements versus initial100. Legacydefault
checkpoint protocol remains exactlycompatible with the real500 state.

NewFullsemantic training is an explicit additional600step paired study, not
a change to the completed500 or preparedMOT-only200 protocols. Each arm has
125286eligible rows:41762MOT,41762RMOT,41762OVMOT. Sharedsemantic sources are
the346nativefitRMOT rows and430Basecategory rows; noNovel/eval GTtraining.
Both begin from immutable500FP32masters with explicit newAdam/scheduler/RNG.

V1native length-sorted sampling changed the finite semantic exposure despite
equal eligible counts:686common repetition IDs out of1558/1553. Bothv1plans
and rawtrainingdata remain. Selectedv2adjacent teacher duplication matches
everyMOTquestionfamily to the native-mixture row, and paired-index-shuffle-v1
selects identical indices and questions per step. This optional new-stage
sampler is recorded in resumable checkpoint protocol; defaultoldsampler,
loss/AMP/kernel/seed/context/4rowstep planning are preserved.

Bothv2officialCPUplans accept all125286rows with0dropped, max22817teacher
and26028native tokens versus32768limit. Each600prefix has2400rows:
811MOT/749RMOT/840OVMOT, with identical per-step question families and1589
semantic repetition rows. DirectJSONcomparison confirms all83524eligible
sharedsemantic payloads, including those1589selected rows, byte-identical.
Prefix tokens differ15774003teacher versus17216498native, so not FLOPmatched.
Conservative text/features including both retained dataset versions total
16,289,535,076bytes before newruntime logs, below25GiB.

Actual native One500 Noul4frame inference succeeded (150..153,6candidate
answers, allfinite .172542..176372), but firstowner misread GT_read from
per-frame metadata. The correct GT/future flags already exist in result.json.
Originalfailure and completed outputs remain; fresh validation checks those
run-level flags and reuses all4nativeframes. No metric, numerical criterion,
source frame or model changed. FutureOne metadata now also declaresGT_read.
The firstpairedowner correctlyfailed before training on the failedprobe flag;
its logs remain. Fixedprobe: validation/onejev500_native_rmot_probe_v2_validated.

Original paired owner v3 started both arms after verified TAO service exit,
then disappeared. Current detached recovery v4 and exact restore receipts are
described above. Oldsource500 state is read-only; new600 paths remain distinct. After600, actualnative reloads, same semantic-policy500/600
tracking comparisons, completevideos and native E2E/resource measurements
remain required. No600accuracy, Novelgain or completed unifiedgoal claim.


## Association diagnostics and natural challenge registration

Posthoc diagnostics now cover all16 original500 controls. RMOT uses all
physical release GT for identity analysis and separately scores query membership
only when that native head exists. TAO annotations are non-exhaustive; unmatched
proposals are not silently treated as universal FP truth. Unknown TAO physical
FPS uses registered observation-step buckets, never fabricated seconds. Sparse
annotation numbering alone is not considered an occlusion gap. The complete
DanceTrack reference diagnostics reproduce all prior counts exactly; official
metrics and inference state are unchanged.

An annotation-only audit of36 registered MOT validation sequences finds647
natural challenges, including46 GT annotation gaps over3 and up to30seconds.
The deterministic longest challenge selects the complete registered MOT20-05
validation half, including a29.76second interval between reliable GT visibility
observations. This is low visibility, not proof of candidate absence or identity
recovery. No synthetic occlusion, proposal removal, GT feedback or visibility
filtering of official metrics was introduced. Native challenge inference and
long recovery evidence remain required.
Evidence: recovery_scope_20261015/audit.json and registered_mot_challenge.json.


## Original U2 core baseline now running

B2 uses the unchanged original SharedTrackCore(hidden_dim256,max_age30 source
frames), original Hungarian margins/lifecycle/identity GRU/motion and original
six MOT losses. A newly trained LayerNorm2560+Linear2560to256 bridge supplies
actual shared DLM ROI features. This is newly initialized seed0 training,
not a nonexistent pretrained U2 tracking checkpoint. CPU training reuses all48
existing native training frames from MOT17-02/Dance0001/MOT20-01, with original
GT ignore/distractor handling, all45 adjacent observation pairs and300updates.
Source frame gaps are retained; adjacency does not imply contiguous frames.
Original scheduled GT forcing and query augmentation remain explicit training
operations; evaluation uses neither GT nor teacher forcing. The registered
owner will validate a four-frame state roundtrip and then the complete1203
DanceTrack video with exactly the existingV1candidates/ROIs and officialTrackEval.
No additional GPU, detector query, cache, or oldU2queue is started.

Comparison limits: same300optimizer updates as earlyCompact controls only.
Core parameters, loss, optimizer, frame exposure and memory max_age differ;
this baseline alone cannot establish an isolated JEV advantage. B2 is MOT-only,
with no claimed trained RMOT/OVMOT semantics. Results are pending.
Plan: configs/groundjev/u2_core300_baseline_v1.json.
Owner: baselines/u2_core300_cpu_v1/owner_state.json.


B2 update: the actual300step CPU training completed (5,942,795 parameters,
4,361,991 changed elements,17.325seconds) and exact model reload plusfour-frame
video/memory/evaluator checks passed. Fullvideo failed at199: a finite retained
track with coordinates near7.6e19 overflowed FP32 squared-sum norms, producing
inf/inf center distances. Model parameters and actual input features were finite.
The smallest accepted numerical repair recomputes only undefined ratios inFP64;
all defined FP32 distances remain unchanged. A broader norm repair was rejected
because it changed frame198 logits. The accepted fix reproduces everypair/new/
no-match/alive logit andID in the entire198frame prefix exactly(maxdifference0),
and actualfailedframe199 is nowfinite. Weights, native candidates, thresholds,
assignment, lifecycle and source-frame max_age are unchanged. The original
failure and rejected regression remain. A newowner resumes the validated198state
and renders all1203frames; no training restart and no skipped frames.
Currentowner: baselines/u2_core300_cpu_v2_numeric_recovery/owner_state.json.
Officialfullvideo metrics remainpending until terminal and validated.


B2 fullvideo is now complete: all1203decodedframes,13664identical native
candidates, all27rejectedframes retained and unchanged official TrackEval.
HOTA1.2434%,IDF1.6255%,IDSW250,FP455,FN4191. The complete geometry/order,
per-frame output ID uniqueness and emitted-box membership in the original
candidates were validated. This newly trained U2 baseline substantially
underperforms Simple8.5298%HOTA and misses most GT; fewerIDSW cannot be
interpreted as successful association under that loss of recall. This does
not isolate JEV causality because losses, parameter counts and exposure differ.
Weights and all198prefix logits/IDs remain unchanged through numerical recovery.
A metadata error initially labeled resumed processing asassociation replayFPS;
raw receipts are retained, the speed claim is removed, and unchanged official
accuracy metrics reproduce exactly. The restored198frames are rendered but
skipcorecompute, so no uninterrupted or detector-inclusive speed is asserted.
Evidence: baselines/u2_core300_cpu_v2_numeric_recovery/validation_and_summary.json.


## Complete RMOT and fresh Compact end-to-end queue

A new detached owner follows the exact existing teacher/native600 child
PID/start-tick identities, waits for both to exit and validates their real600
masters/2400examples before GPU5 reuse. It does not duplicate training or touch
GPU4 short controls, native full9 services on6/7, or the TAO CPU control.
The existing frozen validation0007 single/multiple expression manifests each
contain all800frames0..799. No GT or test labels are read during inference.

The first pass for each scope is a genuine fresh shared Compact pipeline:
originalDLM4B detection, real2560ROI RPC, shared identity/semantic decisions,
memory serialization, candidate archive, prediction writes, rendering and
encoding. The actual wall measurement includes all of these; preloaded remote
service/model initialization is excluded under the prior Full convention.
Every original parser-rejected/empty frame remains in the denominator.
The archive stores its actual native inputs once and following Simple,
pretrainedOne,500,teacher600,native600 controls replay exactly that source.
All four Full checkpoints use identicalv3/adaptive groups/nativeNoul.5 policy.
Cached controls do not receive an end-to-end FPS claim.

The new archive path passed an actual4frame/13nativeROI regression: all geometry,
outputs and full memory state agree exactly after replay,2560dimensions remain,
and replay end-to-end FPS staysnull. This is dataflow validation, not a fresh
native RPC/speed result. Both800frame manifests pass current legal split/image
checks without GT reads. New source features are capped at2GiB across both
videos; repeated control logs omit copied ROI vectors. Conservative prior,
TAO native source, newcap and742680byte probe total20,323,665,338bytes<25GiB.
Only GPU5 is reused after training; no additional GPU or oldU2cache queue.
No fullRMOTdataset, successful transfer, Novel result, or FullOneJev reasoning
claim is made for the small frozen-token-mean Compact text encoder.

Plan: configs/groundjev/full_rmot_study_v1.json.
Owner: multitask_runs/full_rmot_shared600_v1/owner_state.json.
Current status: queued; fresh full800 inference and official results pending.


## Completed raw TAO40 controls and exact-box alias diagnosis

Both raw-candidate controls are terminal and validated on all40annotationframes,
35228same native proposals,160GTboxes and unmodified official TETA. All520native
query outcomes/1203categories perframe are preserved, including203parser-rejected
query responses. This is one Base/Novel video, not a fullTAObenchmark.

| Method, raw native proposals | TETA | AssocA | ClsA | Base TETA | Novel TETA |
|---|---:|---:|---:|---:|---:|
| Simple |18.640|28.618|20.290|23.3725|9.1737|
| Shared Compact |14.517|5.1478|30.314|18.93545|5.679|

BothNovelClsA0. Compact improvesclassificationbutfailsassociation andoverall
transfer; it is not promoted. Posthoc candidateGTrecall.91875 isidentical;
correct birthanchor option coverage in Compact's32identityshortlist is.1567,
versus1 forSimple. This diagnostic is not GTfeedback or an officialmetric.
Evidence: multitask_runs/tao_base_novel63_full_v3_empty_crop_cpu/
validation_and_summary.json.

A GT-free audit finds7343exact-coordinate groups among35228nativecandidates:
27885aliases(79.16%), up to188identicalboxes inoneframe. Rawsource andscores
remain intact. A separately registered CPU proposal-policy study consolidates
only exact unrounded coordinates, keeps the first native box/label/score,
checks identical ROI features and retains every raw index/label/score asaliases.
There is noIoUthreshold,NMS,topK orGTselection. BothunchangedSimple andshared
Compact use the same newgroupedsource; originalrawresults remain the reference.
This changes proposal representation andSimple's selected native label, so an
improvement cannot be attributed solely to association or JEV reasoning.
All40frames/originalqueryoutcomes and all35228alias observations remain.

The new source cap is1GiB; priorfeatures+TAO+reserved2GiBRMOT+probe+aliascap
remain conservatively21,397,407,162bytes<25GiB. No GPU/detector/training is added.
Both exact-alias controls have now completed; detailed results follow below.
Plan: configs/groundjev/tao_exact_alias_control_v1.json.
Owner: multitask_runs/tao_exact_geometry_alias40_v1/owner_state.json.


## Exact-alias controls complete; task-sharing matrix trained

All40frames,7343exactboxgroups and35228originalalias records validate, with
identical nativeROI arrays inside each group. Source file402723754bytes fits
its1GiB cap. A separate posthoc check confirms identical box sets and unchanged
one-to-one IoU.5 coverage147/160(.91875) forrawandgrouped candidates; GTdidnot
select groups or representatives.

| Proposal policy / method | TETA | LocA | AssocA | ClsA |
|---|---:|---:|---:|---:|
| Raw / Simple |18.640|7.0114|28.618|20.290|
| Raw / shared Compact |14.517|8.088|5.1478|30.314|
| Exact aliases / Simple |22.556|22.871|26.472|18.325|
| Exact aliases / shared Compact |21.773|23.273|9.6266|32.419|

Compact correct birthanchor option coverage improves.1567->.5310, but remaining
missing options and association errors stillmatter. Both NovelClsA remain0;
Compact still trails Simple in TETA andassociation under the newpolicy. This
changes proposal representation andSimple's representative label, so the gain
isnot isolated JEV evidence. Originalrawresults remainunchanged.
Evidence: multitask_runs/tao_exact_geometry_alias40_v1/validation_and_summary.json.

The explicitly required task-sharing matrix now has four actual1200update CPU
models: MOT, MOT+RMOT, MOT+OVMOT, allthree. Each has1146511parameters, the same
actualMOT300initialization plusseed0newsemanticheads, sameoptimizer/losses and
existing536group pack (149MOT/325RMOT/62OVMOT,142actualnativeRMOgroups). None
starts froma jointlytrained checkpoint. Independenttaskstreams give identical
actualexample-ID prefixes (400commonMOT/RMO/OV examples whereapplicable).
Allthree task reloads agree exactly for everymodel; checkpoints andfullAdam/RNG
snapshots remain preserved. Fixedtotalupdates changespertaskexposure:
1200MOT,600eachinpairedarms,400eachintriple; notFLOP/exposurematched andnotan
isolated negativeinterference attribution. Older frozenactor histories and
GTteacher Base crops remain explicit; noNovel/evaluation GTtraining.

Full1203DT/raw40TAO/shortRMO8/12 controls are running, followed by twofull800RMO
scopes when the existing freshsource owner completes them. FirstMOT-only
full1203HOTA19.6386%,IDF120.7863%,IDSW348,FP1089,FN1765; othernewarm scores
arepending. Legacy26.316 has differentjointinitialization/additionaltraining
budget andisnotapaired task-subset result.

A plan wiringerror used rawshortRMOlogswithoutROI. Existing enriched logs were
verified tohaveexactlythesame frames/images/boxes/labels/scores and2560features.
Only the verified CPU manager was replaced; its activeMOTinference andffmpeg
were left running, and the newmanager waits for their processgroup to finish.
Allfour1200step checkpoints/completedvideo units are reused, no retraining or
validinference restart. The oldplan, managerstate andhandoff receipt remain.
Currentowner: compact/task_subsets1200_cpu_v2_feature_sources/owner_state.json.
Plans: configs/groundjev/compact_task_subset1200_v1.json and correctedv2.
Evidence: compact/task_subsets1200_cpu_v1/training_validation_and_summary.json.


TeacherFull600 has now completed600retainedupdates/2400examples with its
9,029,141,401byte fullstate. The GPU4 before/after controller has begun the
same native semantic-policy controls; before500single/multiRMOT HOTA areboth0
under explicitNoulmembership. Legacyungated500scores24.886/21.425 aredifferent
runtimepolicies andmustnot be compared aspairedaccuracy. Nativearm and600
three-task officialmetrics remainpending.

Resume timing isscoped explicitly:3595.081seconds covers onlythe150updates/
600examples afterrestore450. Nativeforkdone-rate.66758 usescumulative2400,
so itisnotvalidtail/whole-stage throughput; actualtailrate.166895examples/s.
Atleast25loggedunretainedupdates werealsoexecuted beforetheinterruption and
replayed (totalobservedwork>=625). Full450masters/protocolrestore isexact,
butactualoverlap475loss differs.001734814 withsame.78slotaccuracy; subsequent
CUDAtrajectory isnotbitwiseproven andspecificnumericalcauseisunknown. No
kernel, seed, numericaltolerance oracceptance criterion waschanged. Rawlogs
remain, andthecompletionreceipt reportsthislimitation ratherthan hidingit.
Evidence: training/onejev_unified600_teacher_v2/resumed_completion_validation.json.

Tasksubset parameter counts above refer to the trainableCompactcore. Its text
inputs stilluse the originalOneJev frozen token-embedding table onCPU; theyare
not free, nativeOneJevdecoderreasoning, or a1.146Mparameter totalend-to-end system.
The table andnativeGround perception/ROI services arecommonwithinthiscohort.


## Four task mixtures: complete MOT video comparison

All four new models have completed the same 1203-frame DanceTrack0004
validation video with the same 13664 native candidates, all 27 rejected
frames retained, and unchanged official TrackEval. These are one-video,
one-seed results, not a full dataset benchmark.

| Training tasks, 1200 total updates | MOT updates | HOTA % | IDF1 % | IDSW | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| MOT |1200|19.639|20.786|348|1089|1765|
| MOT + RMOT |600|6.760|9.884|80|136|3865|
| MOT + OVMOT |600|21.228|20.081|414|2461|1006|
| All three |400|19.312|17.827|422|3363|1372|

MOT+RMOT misses most GT; fewer identity switches are not a success under that
collapse. MOT+OVMOT improves HOTA but has worse IDF1 and more switches. The
three-task model is also mixed. The total update budget is fixed, but MOT
exposure differs, so these measurements do not isolate task interference.
Two additional MOT-only 400/600-update controls now use the same initialization,
constant optimizer and exact per-task sample stream. Their logged loss/sample
prefixes must agree with MOT1200 before full-video evaluation. They complement
the fixed-total-budget study, and do not replace its original arms. No new GPU,
detector query or candidate feature file is required. RMOT/OVMOT scores for the
four-task mixtures are still being measured.

Evidence: compact/task_subsets1200_cpu_v2_feature_sources/
mot_full1203_comparison.json. Exposure control plan:
configs/groundjev/mot_exposure_controls_v1.json.

## Genuine paired Full/Compact end-to-end measurement queued

The separate paired measurement waits for the verified GPU5 full-RMOT owner
to finish all its work and release its owned service groups. It then runs Full
native600 and the current shared Compact on the same legal 800-frame RMOT
video and the same physical GPU5. Each arm gets a newly started original DLM
service and its own native request counter. Compact includes its actual native
ROI RPC cost; Full uses its actual V3 image crops. All reads, detector/ROI
requests and queueing, association, memory, source writes, rendering and encoder
completion are timed. Preloaded model/service initialization is recorded
separately, under the earlier Full timing convention. This is a fixed-order
single pair under concurrent machine workloads, not a counterbalanced benchmark.

All 800 frames, including rejected and empty frames, remain in the metrics.
Canonical frame/image/box/label/native-score/parser-rejection equality is
required before accepting the controlled association comparison. A mismatch
retains both real runs and fails pairing; it does not silently change decoding,
seeds, frames or criteria. Process-group-filtered GPU memory is sampled every
5 seconds, alongside the core allocator peak and core process peak RSS. The
sampled total is a lower bound on instantaneous peak. Frozen text-table and
native ROI service components are reported separately from the small core.
The source archive cap is another 1GiB; conservative reserved sources remain
22,471,148,986 bytes below the 25GiB limit. No current GPU job is restarted.

The candidate contract passed on four real frames/13 regions. A separate
actual Compact CPU replay probe confirms zero GPU allocation, positive process
RSS, 1146511 core parameters and the frozen 248320x1024 BF16 lookup table
(508559360 nominal tensor bytes, not resident RSS). The probe is not fresh
native inference or an end-to-end speed result.
Plan: configs/groundjev/paired_tracking_e2e_v1.json.
Owner: e2e/full_compact_rmot800_v1/owner_state.json. Measurements are pending.

## Matched MOT exposure and frozen affinity controls — 2026-10-15

The complementary MOT400/600 controls both completed and validated all1203
frames,13664 native candidates and27 rejected frames. Their actual example-ID
and logged-loss prefixes equal the original MOT1200 exactly; same shared
MOT300+seed0 semantic initialization and constant AdamW/loss.

| MOT-only update count | HOTA % | IDF1 % | IDSW | FN |
|---|---:|---:|---:|---:|
|400|13.317|17.757|174|3483|
|600|13.979|13.411|312|2566|

At600 MOT updates, M+RMOT HOTA6.760 and M+OV21.229; at400 MOT updates,
all-three19.312. These complement the fixed-total1200 comparison; total
updates differ, so they do not isolate gradient interference. One video/seed,
no model promotion or full-dataset claim. Receipt:
outputs/groundjev/compact/mot_exposure400_600_cpu_v1/validation_and_summary.json.

Registered next: configs/groundjev/fixed_affinity_controls_v1.json. Same
frozen shared semantic checkpoint, same native visual/text inputs and shortlist,
five learned/fixed association and auxiliary policy combinations. Complete
DT1203, both short RMO scopes and raw/alias TAO40, CPU only, no new features.
This is an inference intervention, not a retrained optimal conventional baseline
or unique JEV proof; actual confidence/history changes with each policy.

Actual4-frame preservation regression passes: semantic/auxiliary functions on
the same state unchanged, JSON versus lossless binary full states and next-frame
predictions exact. New controls retain per-frame decisions and final full binary
state; no speed comparison to older JSON/every-frame exports. Defaults remain
JSON/every-frame. Final full state export is now included in the pipeline timer;
older timing receipts excluded that duplicate final export and remain historical.
The queued fresh paired E2E runs will use the corrected boundary for both arms.

## Same-policy Full teacher600 scores and context research — 2026-10-15

Real pretrained OneJev500 versus600 additional shared teacher-stage updates,
unchanged native candidates and native semantic policies:

| Registered short scope | Before500 | Teacher600 |
|---|---:|---:|
|MOT4 HOTA %|30.868|30.601|
|RMOT single8 HOTA %|0|0|
|RMOT multiple12 HOTA %|0|9.945|
|TAO Base4 TETA %|11.981|52.181|
|TAO Base4 AssocA %|18.399|28.308|
|TAO Base4 ClsA %|0|100|

All8 actual controls retain their original frames/candidates and decoded videos.
Both RMOT phases use native Noul membership>=.5, unlike the legacy ungated
500 controls. All1203categories remain supplied; Base4 has only the Base
person/baby canonical category, so ClsA100 is not broad or Novel recognition.
Teacher600 resumed exact450 FP32/Adam/RNG state but not bitwise trajectory;
that discrepancy remains explicit. No model promotion/whole-video/E2E claim.
Receipt: training/onejev_unified600_semantic_controls_v1/
before500_teacher600_validation_and_summary.json under outputs/groundjev.
Native600 trainer and its registered controls continue independently.

TAO exact-alias40 causal replay reproduced every original output and state,
including all7343candidates. With113 surviving-birth-anchor observations,
frozen-trajectory correct-option coverage at32/64/128 is53.097/60.177/74.336%.
Maximum420live tracks; best correct rank reaches395. This is a coverage
diagnostic, excluding already terminated/never correctly born anchors; it
does not establish larger-context tracking accuracy. No GT modifies runtime.
RawTAO40 rank audit is running in its original blocking session86629.

New context32/64/128 experiment is registered in
configs/groundjev/compact_context_sizes_v1.json. Same frozen model, native
inputs, group16 and semantic functions; actual64/128 runs exceed the training
context32 and change pooled context/softmax and the entire subsequent history.
Dense functional check uses251live IDs before frame2, all requested option
counts valid and default32/explicit32 predictions/states exact. Owner2003663
(session5404) waits for affinity owner1986817 to finish and exit before
running8 additional full-video controls (learned and fixed-cosine64/128 on
DT1203 and aliasTAO40); it revalidates existing matched CPU32 references.
No additional GPUs/new feature source or speed claim. Full goal remains active.

## Both paired Full600 checkpoints complete — latest 2026-10-15

The original blocking training shell returned exit0. Both retained600 updates /
2400 examples and real9,029,141,401byte FP32/Adam/scheduler/RNG state files.
Both native multimodal repeated reload checks completed with probability delta0
and474 BF16 weight entries; selected language weights changed versus500.
Native arm restored250 exactly and completed350 updates/1400examples in
8767.184s (.159686examples/s for resumed tail only). Raw cumulative2400
examples divided by resumed-process time is not whole-stage throughput.
There is no native unsaved-log overlap after250, so bitwise trajectory equality
is untested; the separate teacher475 discrepancy remains recorded.

Existing GPU4 control owner continues native600 same-policy scoring. The
verified GPU5 handoff has occurred: full_rmot_shared600_v1 is now executing
single800_FRESH_COMPACT_NATIVE800, not waiting for training. This first genuine
full800 query run uses original live DLM and real ROI/memory/video path, and
its subsequent matched controls consume the same bounded archive. Full/Compact
paired E2E owner still waits for this entire registered RMOT owner to finish.
No full800 accuracy or new E2E result is claimed before terminal validation.

RawTAO40 rank audit has also completed (original blocking shell exit0):
all35228 candidates and every original causal output/state reproduce exactly.
Maximum2670live tracks,134existing-anchor events, frozen32/64/128 coverage
15.672/31.343/44.776%. Context size alone cannot establish correct retrieval
or accurate association; these remain frozen-trajectory diagnostics only.

Registered fullMOT follow-up: configs/groundjev/onejev_full1203_controls_v1.json.
Three actualFull native600/before500/teacher600 arms on the identical original
V1 DT1203 source, genuine nativeChoice/Noul/Score and realV3crops/motion,
unchanged context limit/adaptive grouping, all27 rejected frames retained.
GPU4 owner2024544/session88330 waits current short-control owner1870814
success+exit before allocation. Existing600 nativeMM validation is reused;
all three new arms retain binary final state/complete per-frame decisions
and share the existing source, with no newfeatures or detectorE2E claim.
These full-video fine-tuning comparisons remain pending.
