# Current status — 2026-10-14

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

The original corrected OneJev500 trainer stopped without a terminal receipt.
It saved BF16 weights at 250 and last logged 275, but no FP32 master weights,
Adam state or RNG state. Its external termination cause is not established.
The old intermediate is preserved and is not declared a completed 500-step run.

A new finite replay of the original unfinished 500-step protocol starts from
its original valid 100-step initialization, with unchanged corpus, seed0,
lr5e-6, 32768 context, native fork tolerance and zero dropped rows. It runs on
physical GPU4; the latest persisted training log was step300/500, with an actual9.029GB
FP32/Adam/scheduler/RNG/cursor state checkpoint saved. It does not
pretend to reconstruct missing Adam state at old step250. Checkpoint handling
now atomically saves FP32 masters, Adam, scheduler, all RNG and exact planned
step cursor every125 steps, retaining one latest full training-state checkpoint.
A real-native-ROI functional resume regression gives identical next-step loss
and parameters; this small probe is not itself an actual OneJev training result.
Actual intermediate125checkpoint contains all474weights, with millions of
BF16-visible changes in three language probes versus initialization100. Two
independent native multimodal reloads agree exactly(probability max delta0),
with native fork auto-check unchanged. This establishes genuine intermediate
checkpoint correctness, not500completion or a tracking accuracy gain.
Config: configs/groundjev/onejev_own_history_v3_exact500_replay_stateful.json.

The idle owned native visual-feature RPC onGPU5/port8111 was stopped after
all feature consumers finished and no established clients remained. GPU5 was
reused for the completed absence120stage and now runs isolated checkpoint
controls; each owned replica stops at terminal.
The native72pack and all Compact training/control/recovery owners are terminal.
The later originalGround full-retention service now runs onGPU7; formalGround200 training has finished and its full dense/ref9 evaluation now ownsGPU6. GroundJEV does not own the fufu/UDMT/ywh jobs and has not stopped
them. At most four physical GPUs are allowed. No old U2 queue is running.
Temporary feature caches remain bounded to25GiB; datasets and training-state
checkpoints are separate preserved artifacts.

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
