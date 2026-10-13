# Current status — 2026-10-14

Real OneJev tracking SFT completed100steps/400examples, saved all474weights,
and passed two exact reloads of real multimodal decisions (max probability
difference0). Three BF16-aligned language-weight probes changed. This proves
training correctness, not tracking improvement.

## Four-frame same-candidate results

All results use exactly the native DLM candidates for MOT17 frames302–305
and unchanged TrackEval; this is a short functional scope, not full video.

| Model / evidence | HOTA | IDF1 | IDSW | FP | FN |
|---|---:|---:|---:|---:|---:|
| Simple | .74881 | .83650 | 1 | 17 | 24 |
| Pretrain / V1 | .42163 | .42748 | 60 | 16 | 24 |
| SFT100 / V1 | .32834 | .31179 | 75 | 17 | 24 |
| Pretrain / V2 | 0 | 0 | 0 | 0 | 135 |
| SFT100 / V2 | 0 | 0 | 0 | 0 | 135 |
| Pretrain / V3 | .34394 | .35193 | 47 | 11 | 48 |
| SFT100 / V3 | .19740 | .30120 | 0 | 6 | 110 |

Zero IDSW in collapsed runs reflects missing predictions. Retain these
negative results. Correct-identity-option coverage is100% in V1 for both
models, but fixed-anchor existing association accuracy falls17.95% to8.11%
after SFT100. Candidate GT recall.82222 is identical. Post-hoc diagnostics
never correct runtime memory and do not replace official metrics.

V2 adds current indexed crops, historical crops and causal geometry matrices,
but its NO_MATCH definition overlapped NEW and collapsed all128 proposals.
V3 preserves those real inputs while requiring a previously observed object
for NO_MATCH and omitting it with empty memory. V1 remains unchanged for the
complete native video baseline. V3 has not demonstrated improvement.

## Current running jobs

PhysicalGPU4: the corrected finite500-step full language-model own-history
adaptation owner has resumed from checkpoint100. The initial run was stopped
for an audited lifecycle-label error before any new valid save.

The audit found203 ACTIVE/REACTIVATE labels whose identity is absent from
current supplied proposals (164student100,39pretrain). They are now LOST.
All5951 lifecycle rows passed current-proposal consistency regression; actor
states, image evidence, other targets, IDs and all registered frames unchanged.
Original corpora/partial training logs remain. The label repair initially
failed on an ID underscore; that failed derived fragment and failed preflight
owner are preserved. Corrected export passed and a fresh owner was started.

Current config: configs/groundjev/onejev_own_history_v3_lifecycle_fixed2_500.json.
The new stage uses the same official full language-model configuration,
initialized from the valid100-step checkpoint, frozen vision, seed0/lr5e-6.
13,501 unique supervised rows from two real actor rollouts become20,881
training rows with explicitly registered4x identity repeats. The real
histories include526 contaminated observations,96-observation total across
MOT17/DanceTrack/MOT20 train clips,35-option exposure, synthetic false
fragments and real reappearance. GT proposals are explicitly declared;
actual GroundAnything-proposal training rollout is still pending.

All20,881 rows fit32768context (max22817,p50/p90/p99 15520/22158/22544),
none dropped. First two500-step losses3.2479/8.0658, gradients299.979/513.188,
peak allocator18.1GiB, measured75.8/76.2s per optimizer step. This is a
running finite stage; no final checkpoint or improvement claim yet.

PhysicalGPU6: one blocking owner processes all1,203DanceTrack0004 validation
frames with the real frozen DLM + pretrained V1 core, then the exact-candidate
Simple control and official metrics. Both earlier attempts stopped at frame21;
the dense request profile does not ensure valid model output. Preserve those
attempts. The current explicitly registered empty_and_record measurement
policy keeps the unchanged scorer: rejected output becomes an empty prediction
set and the real tracker takes that causal frame step. Raw error and elapsed
time are recorded; every frame remains in the official denominator. Completed
20-frame memory/decisions were restored exactly. The job is still running;
its resumed timing cannot be claimed as uninterrupted end-to-end FPS.

GroundJEV now uses physical4/5/6 within allowlist4/5/6/7; other projects untouched.
Physical5 runs native visual-feature RPC and finite RMOT/OVMOT/native-evidence checks.
Crop evidence: teacher59,665,520bytes, SFT100 student38,165,373bytes and
pretrained student44,023,768bytes. Temporary feature limit25GiB; no full
source-dataset feature extraction or old U2 queue is running.

## Preserved foundation and remaining scope

Old U2 owners/queues stopped;39.89GiB regenerable features removed, original
data/checkpoints/results preserved. Independent worktree
/data2/user/LocateMOT_GroundJEV, branch codex/groundjev-4gpu-20261013,
inherits U2 remote51614cb. Both official model bundles are complete with
matching LFS digests. Separate environments: DLM Python3.12/Torch2.9.1/custom
SGLang; OneJev Python3.10/Torch2.10/Transformers5.17.0. Its original.05 fork
check remains unchanged and passes.

Native hidden-feature export is verified, and a frozen-backbone temporal
adapter completed200steps with actual gradients/checkpoint reload. It matches
the unadapted calibration retrieval131/134, so no tracking gain is established.
DLM decoder adaptation and dense/referring retention benchmarks remain pending.
Real RMOT short loops and same-candidate controls completed; TAO exhaustive requests are active; V6 native-feature OneJev checks completed
with negative tracking evidence. Full-video results, formal500
before/after, true detector rollout, full RMOT/OVMOT, Compact/matched controls,
long-memory ablations and joint-task research remain incomplete. Pilot completion will not complete the
full unified online tracking goal. Old U2 perception recall is not MOT HOTA.


## Native features and legal multitask checks

Official DLM MoonViT/projector428,482,048parameters,170weights; real image
features31x2560 and local region tokens31x2560x3x3. Exact agreement with the
unmodified serving visual function and checkpoint reference; ROI coordinate
ramp error9.54e-7. Language decoder is not loaded by the feature RPC.

Temporal adapter200steps/seed0,693,632changed weights, loss4.20862 to.09104,
finite nonzero gradients, reload difference0. Training:legal MOT17/DanceTrack/
MOT20 clips; calibration:disjoint train DanceTrack0006. Native and trained
adapter retrieval both131/134=.977612. No HOTA or detection improvement claim.
Checkpoint:/data2/user/LocateMOT_GroundJEV/outputs/groundjev/ground_adaptation/
native_temporal200/checkpoint.pt,8,331,829bytes, SHA256
9ad17570ca88e972b837389373ca94b3668ee433033f6f9c8c0044489b62e35b.

Legal Refer-KITTIv2 validation0007, release expressions with empty ignore
fields; unchanged TrackEval on every registered frame, GT read only after
predictions. These are explicit short clips, not full-dataset metrics:

| Scope / backend | HOTA | IDF1 | FP | FN | Native target recall IoU.5 |
|---|---:|---:|---:|---:|---:|
| Single/empty8frames Full | 0 | 0 | 0 | 5 | 4/5 |
| Same candidates Simple | .242284 | .375000 | 7 | 1 | 4/5 |
| Multi/empty12frames Full | .194792 | .217391 | 11 | 25 | 5/30 |
| Same candidates Simple | .243151 | .217391 | 11 | 25 | 5/30 |

Single-query V1 wrongly suppresses correct proposals; multi-query proposal
coverage is low. See the authoritative GT-contract correction below. Strict wire errors
remain in logs; scorer rejects frame152/single and34/multi, both retained in
metrics. Multi Full complete processing FPS.579804 includes video encoding,
output serialization and RPC queueing;9.18752s OneJev initialization separate.
Single legacy frame-compute FPS.627761 is not complete end-to-end FPS.

TAO registers all1203fixed LVIS-v1 classes in96-class chunks; the first
run stopped at a rejected prefix. The corrected exhaustive request owner is active. A rejected chunk yields an explicitly logged rejected
whole frame; allframes/classes remain requested. Source physical FPS absent:
registered sparse functional scope uses declared annotation-step time, not
verified source seconds. Original categories/GT/frequency split retained for
unmodified TETA; no Novel evaluation GT in training.

Native memory2real-frame sanity uses actual saved DLM proposals, no GT:
31persistent IDs, exact reload, fixed anchors, held long features unchanged,
past timestamps rejected before mutation,10,682,927byte serialized memory.
V1 behavior stays unchanged; native mode adds region/temporal memories,
causal feature history, lifecycle survival probability and identity entropy.
V4 first real history request exceeded32768with56426tokens and was preserved.
V5 uses declared CSV milli-units for matrices without dropping candidates or
identity options; its actual4frame controlled evaluation is pending.


### Completed native-feature control and measured limitations

V5 still exceeded context at34313tokens. V6 compresses native-history rows
in addition to all-pair similarity matrices; frame304 then reached34419tokens.
The smallest further fix is context-adaptive candidate grouping only when the
unchanged official native preflight rejects a group before inference. Every
proposal and up-to32 shortlisted identity options remains. The two valid
V6 prefix frames restore exact native outputs/state; resume completes304/305.
Original V4/V5/V6 failed artifacts remain, no sample skipped, no token-limit
relaxation and no retraining/checkpoint change.

MOT17 same128native candidates,4frames, unchanged TrackEval:

| Decision | HOTA | IDF1 | IDSW | FP | FN |
|---|---:|---:|---:|---:|---:|
| Simple IoU | .748813 | .836502 | 1 | 17 | 24 |
| Fixed IoU + native2560cosine | .738898 | .813688 | 4 | 17 | 24 |
| Fixed IoU + trained temporal128cosine | .746248 | .828897 | 2 | 17 | 24 |
| Pretrained OneJev V6 native/crop/motion | .359328 | .330709 | 61 | 17 | 33 |

Native/temporal fixed controls use exactly the exported features, coefficients
.5/.5, NEWscore.3,32-identity geometry shortlist and16-candidate groups. They
are untrained association controls, not JEV/Compact. Temporal-vs-native is
also a dimensionality change; random128 and larger video controls remain
required before attributing the difference solely to training. Native-feature
OneJev still underperforms Simple; no isolated feature or foundation gain
claim. Existing-ID diagnostic accuracy9/75=.12, option coverage72/75=.96,
mean correct native mass.062495; candidate GT coverage remains111/135.
Its8,504,115,712byte allocator peak and.042028association replay FPS exclude
live detection; resumed timing does not support an end-to-end speed claim.

TAO first4frames have zero predictions and TETA/LocA/AssocA/ClsA0 under the
unmodified official source b498aa87b252bfb75d7cc0d8d6435c3e1e74e4de. There
are18GT boxes, source canonical class baby/person.n.01, Base. Novel is null
because this scope has no Novel GT; never report it as a Novel benchmark.
Two evaluator formatting configuration errors were repaired by restoring
its official defaults OUTPUT_PER_SEQ_RES=True and OUTPUT_SUMMARY=True;
metrics/source/GT/predictions unchanged, failures retained.

The first vocabulary run registered1203classes but stopped on a rejected
96-class chunk:192/192/192/288classes actually requested across the4frames.
This is recorded in vocabulary_coverage.json, not claimed as exhaustive
1203-class inference. A new finite owner requests ALL13chunks on every frame,
records every native outcome, and only then applies the same explicitly
registered whole-frame-rejection policy. It is active at publication.
First-prefix run processing FPS.020682 is not full-vocabulary inference FPS.
Original raw failures show many class-conditioned hallucinated boxes. The
next grounding research must address proposal quality and negative classes
through the official spatial/DLM training route, with dense/referring retention.

Visualization integrity verified8/12/4frames; KITTI display edge padded from
1242x375 to1242x376, predictions/GT coordinates unchanged. Formal500-step
SFT and fullDanceTrack original blocking owners continue. Pilot and final
unified research goal remain incomplete.


## Pre-publication RMOT GT-contract repair (authoritative results)

The new RMOT evaluator incorrectly read release top-left x/y as centers.
This was found before updating GitHub's branch. Existing established
read_gt in tools/unified/u1_formal_expression_gate.py defines normalized
top-left xywh. Corrected transport agrees for all70target box comparisons
to5.68e-14pixels. Predictions, target sets, frames, source extents, seeds,
metrics and thresholds unchanged. Original metric receipts are preserved
as invalid_center_format_metrics.json and explicitly marked invalid;
local invalid GT views remain but raw GT is excluded from GitHub.

Authoritative corrected results:

| RMOT scope / backend | HOTA | IDF1 | IDSW | FP | FN | Proposal target recall IoU.5 |
|---|---:|---:|---:|---:|---:|---:|
| Single/empty8frames Full | 0 | 0 | 0 | 0 | 5 | 4/5 |
| Exact candidates Simple | .242284 | .375000 | 2 | 7 | 1 | 4/5 |
| Multi/empty12frames Full | .194792 | .217391 | 0 | 11 | 25 | 5/30 |
| Exact candidates Simple | .243151 | .217391 | 0 | 11 | 25 | 5/30 |

The earlier zero-proposal-coverage diagnosis is invalid. On the single
query, four correct observations are present but V1 OneJev suppresses them
as IGNORE; association/birth/input evidence is the first failure. On the
multiple-target query, proposal coverage5/30 still limits every tracker.
Next controlled check should compare V1 against current-crop V3 on the
same RMOT proposals, while preserving the DLM multi-target recall deficit.
Dense/referring adaptation and original retention remain required, but
not every RMOT failure can be attributed to the perception front end.

The new all-class TAO owner remains active. A800-SXM4-80GB hardware was
verified for4/5/6; foreign UDMT processes are unchanged. Original OneJev
500step and fullDanceTrack owners remain responsible for their jobs.


## Exhaustive TAO functional control completed

All52native requests completed:13chunks and1203fixed categories for each
of4registered annotation frames. Unchanged TETA yields0TETA/LocA/AssocA/
ClsA and0predictions under the recorded whole-frame-rejection policy; Base0,
Novelnull because this video's GT contains no Novel class. No full-video
or complete-dataset result is claimed. Actual complete processing
FPS.0037577955 includes all calls, association, file/video writes and writer
completion; preloaded remote DLM startup and9s-class OneJev loading excluded.

Rejected/scored chunks per frame:11/2,10/3,7/6,10/3. Independently scored
chunks contain201/441/777/354boxes. Evaluation-only class-agnostic target
coverage is10/18 atIoU.5 before whole-frame rejection. This is a localization
diagnostic, not TETA or classification correctness. Whole-frame rejection
discards useful native outputs; retain this negative control and preregister
a query-level empty-prediction control that retains independently scored
chunks, all class requests, all GT and frames. Never label invalid chunks
valid or change the official parser/metrics. Raw native responses remain local.

At this completion check, the original blocking SFT500 owner is live, last
reported50steps/loss2.8577/20.6GiB allocator; original fullDanceTrack owner
is live, completed769of1203frames. Dataset physical clocks independently
agree with seqinfo:MOT17=30Hz,DanceTrack=20Hz,MOT20=25Hz. TAO sourceFPS
remains unavailable and annotation-step units explicitly declared.
The69-file native-feature/GT-contract stage was verified on GitHub at
1724df9de3465fce5fd1de96ccf81510029cd7f1 with exact local tree equality.
No Pilot or final-goal completion claim.


## 2026-10-14 — Registered task and inference controls

All six controls completed on A800 GPU5 with original OneJev weights or
actual SFT100. Predictions use exactly the same saved native candidates;
GT uses verified top-left Refer-KITTI release coordinates and unchanged TETA.
Official scopes remain8/12RMOT frames and4TAO annotation frames.

| RMOT / V3 | HOTA (fraction) | IDF1 | FP | FN |
|---|---:|---:|---:|---:|
| Single pretrained | 0 | 0 | 0 | 5 |
| Single SFT100 | 0 | 0 | 3 | 5 |
| Multi pretrained | .222720 | .177778 | 10 | 25 |
| Multi SFT100 | 0 | 0 | 1 | 30 |

Current crops alone do not fix cold IGNORE. MOT-only SFT100 has negative
transfer; it is not evidence of useful RMOT fine-tuning. Original finite
MOT SFT500 continues, and mixed-task/actual-proposal training remains needed.

| TAO scored-query retention | TETA (%) | LocA | AssocA | ClsA |
|---|---:|---:|---:|---:|
| Fixed native candidates Simple | 24.171 | 29.452 | 43.061 | 0 |
| Same native candidates OneJev Full | 16.815 | 27.459 | 22.987 | 0 |

Failed query chunks remain empty; independently scored chunks retained,
without re-running the detector, filtering classes/GT, repairing invalid
syntax, altering metrics or dropping frames. This is a registered combiner
control, not an output-format pass. Full remains slower and less accurate.
Both classification terms0 expose missing/wrong class decisions; the native
shared JEV class head is still required. Novel is null in this Base-only scope.
No detector-inclusive or full-dataset speed claim:replay FPS.838284Simple
and.00352281Full, with all1773label/box rows kept.

Exact-coordinate object-hypothesis grouping preserves every distinct native
box and all label alternatives:201/441/777/354rows become54/58/109/85.
No overlap threshold/NMS/topK/GT filtering. Independent Simple/Full controls
completed; first native label remains an explicitly unlearned class baseline.
A future genuine typed classification head must use the full dynamic vocabulary.

Isolated FLA0.5.2 + einops0.8.2 addon (original environments untouched):
GDN kernels2.29/3.24/8.43x faster at257/1024/4096tokens after both sides'
warmup, maxoutput difference.0004883, finite gradients. First import lacked
einops and was fixed only in the addon. First unbalanced warmup is preserved
and does not support63x speed. Actual3multimodal OneJev requests differ up
to.021211 in typed answers, exceeding the preregistered.001. Choices unchanged
but probabilities/material memory gates can change. Accelerator rejected;
original automatic fork check/.05 tolerance remains unchanged, training
and original video owners stay on the reference backend. No end-to-end gain.

Native DLM training preparation reconstructs the actual official
GAMQwen3DLM wrapper, MDM.5+causal.5/block32, with all569tensor keys/shapes
matched. The released JSON has152670AR rows; real checkpoint/tokenizer have
152671including MASK152670. Native reconstruction uses real header size; no
weight is added or randomized, no new AR download required. Initial mismatch
retained. This is a key audit, not gradient proof.

A finite3step native projector-only gradient check on legalMOT17TRAIN209/210
combines original DLM spatial loss and same-ID contrastive ROI loss. Language/
ViT frozen, projectorFP32masters, seed42, all32targets each frame with no
clipping or dropping. It is a custom native-adapter policy, not the stock full
language SFT recipe or completed adaptation. Meta nonpersistent rotary buffers
initially caused allocation failure; native Accelerate initialization fixes
that without creating learned values. The strict native cached-image schema
then rejected a path-only dict; real original bytes/path are now supplied.
All failed units preserved; the final v6 real-gradient check passed onGPU7. Four
physical GPUs4/5/6/7max; foreign UDMT jobs untouched. Dense/referring retention,
exported bundle inference and formal adaptation remain required.


### Exact object geometry and native gradient runtime

Exact geometry grouping controls completed:Simple TETA33.042,LocA48.539,
AssocA50.587; Full23.320,49.095,20.866; bothClsA0. Recorded replayFPS
9.82260Simple/.0197772Full. Every distinct box and all class hypotheses
retained, first native label is an explicitly unlearned category baseline.
The native class decision head and mixed-task supervision remain necessary.
All8new video units have exact registered frame counts, no runtime GT and
no detector-inclusive FPS claim.

Initial native DLM projector checks failed before gradient acceptance; the
final v6 unit now passed (authoritative receipt below).
First runs failed on meta-only nonpersistent buffers, cached-image schema,
projector per-image list output and CUDA-graph lifetime replay. Corrected
with native buffer initialization, actual original image bytes, tensor-list
concatenation and the official GAM_DLM_FLEX_CUDAGRAPHS=0 flag. Independent
loss-gradient probes further require task-local AOT donated_buffer=False.
Original failures preserved; native MDM/causal/block32/seed42/GT unchanged.
Final3-step native check passed on physical7; exported-replica before/after
controls completed. Formal adaptation remains incomplete.


### Actual native DLM spatial and temporal gradient proof

Final v6 native training unit completed3optimizer steps on legalMOT17TRAIN
209/210, all32targets each frame and32same-ID positives. Official native
GAMQwen3DLM MDM.5+causal.5/block32 retained, with.1identity ROI contrastive
loss. Language and native ViT frozen;27,267,584native projector parameters
useFP32masters. Independent dense-gradient norm8.75947 and temporal-gradient
norm2.13338, finite combined norms8.77141/9.25557/9.02194. Native losses
1.61708/1.66780/1.65477 include different sampled noise; do not claim training
improvement from this warmup.26,766,531FP32elements and1,968,676BF16elements
changed. Delta reload weight difference0,327,218,423bytes, SHA256
bf8537a97675e465d80dfb08cb5b21454ee37711f622510fe8181b86370e2b3e,
peak18,455,674,368GPUbytes. This proves actual localization and identity
gradients, not formal adaptation or tracking gain.

Checkpoint:/data2/user/LocateMOT_GroundJEV/outputs/groundjev/ground_adaptation/
native_tracking_grad3_v6_native_runtime_flags/projector_delta.pt. Frozen
nonpersistent rotary buffers remainFP32. Native official CUDA-graph flag0
and task-local donated_buffer=False accommodate checkpointed backward and
independent gradient probes; math/masks/loss weights/seed42unchanged.
Failed meta-buffer/image-schema/list-output/CUDAgraph/donation units retained.

Exported complete DLM variant models/GroundAnything-DLM-TrackProj3 replaces
only5native projector tensors, retaining all569keys and every other native
weight/tokenizer/code/config unchanged.9,687,419,704bytes, SHA256
a626aad29c5bbc17e3a465a519e93681aaedacd81cbb21805cce40b7122777e9.
Original model/server untouched. A distinct GPU7replica/8102 and launch bundle
are used for48same-profile native frames:4dense+8single-ref+12multi-ref,
original/updated, fixed Simple association. Requesteddecoderseed42, same
unchanged official parsers and evaluator. These functional retention probes
do not stand in for the full original GroundAnything benchmark suite.
The one blocking owner stops only its owned adapted replica on completion.


### Verified native projector retention controls (2026-10-14)

All six fresh inference and official evaluation units passed scope and encoded
video frame checks:48frames total, no runtime GT or future access, every
rejected/empty frame retained. Original versus projector3 HOTA/IDF1 fractions:

| Scope | Original HOTA | Projector3 HOTA | Original IDF1 | Projector3 IDF1 | IDSW before/after |
|---|---:|---:|---:|---:|---:|
| MOT17 4frames | .745083 | .755029 | .824427 | .847328 | 3/1 |
| Single RMOT 8frames | .299355 | .303853 | .375000 | .400000 | 1/1 |
| Multi RMOT 12frames | .229517 | .227592 | .222222 | .212766 | 0/1 |

Both single variants reject frame152; both multi variants reject frame34.
These frames remain in official denominators. Decoder tracking_dense,
requested seed42, official_scoring parser, unchanged geometry Simple and
legal validation scopes are identical. Short dense and single controls
improve slightly, multi-referring declines. This is not formal adaptation,
full original grounding/referring benchmark retention or a broad gain claim.
Actual detector-inclusive FPS are in aggregate_results.json; concurrent
workloads and original GPU6 versus replica GPU7 preclude an isolated speed
comparison. The first replica failed KV allocation at mem-fraction.25;
its failure is retained. The completed replica used.4 and stopped its own
process group. Original server/model and foreign UDMT are unchanged.

Authoritative receipt:
outputs/groundjev/ground_adaptation/projector3_retention_v2_memory_budget/aggregate_results.json.
OneJev formal500step owner remains live, latest returned report100/500;
full1203frame DanceTrack owner remains live, latest returned frame1183.
No completion claim for either. Current persistent feature service on5
uses about4GiB; no GroundJEV process remains on7 after replica cleanup.
Next research hypothesis: exact object geometry grouping improves localization,
but copying a native query label leaves ClsA0. A real typed OneJev category
head using every supplied vocabulary class, separate from identity and absent
detection confidence, is required. It must preserve all candidates and GT.
