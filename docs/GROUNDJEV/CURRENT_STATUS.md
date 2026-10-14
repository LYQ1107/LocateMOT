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
all feature consumers finished and no established clients remained. GPU5 is
now reused for the separately registered absence120training stage.
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
new absence-training5 keep the project within4/5/6/7. Both full-suite results are pending. Fresh48frame MOT/RMOT functional
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
