# Experiment registry

| Experiment | Status | Checkpoint / scope |
|---|---|---|
| B0 old U2 | perception results preserved; no trained tracker | epoch-1 1d9077ae… |
| B1 GroundAnything + Simple | complete1203frame controls validated | V1HOTA8.530%; freshV3HOTA8.257%, each its own exact Full candidates |
| B2 GroundAnything + U2 Core | actual300steps and all1203frames complete: HOTA1.243%,IDF1.625%,FN4191; no promotion | original256D core/learned2560to256ROIbridge; actualoverflow-only geometry repair,198prefix exactlyreproduced; new training, notoldcheckpoint |
| B3 GroundAnything + pretrained OneJev Full | complete1203frame nativeV3 validated; HOTA2.389%, actual E2EFPS.118560 | real originalOneJev, singlevideo/seed, concurrent workloads |
| B4 tracking-SFT OneJev | actual500complete/all16shortcontrols; shared600 paired arms resumed from450/250 with exact fullstates; same-policy controls queued | MOT500 is notmultitask; new600 adds shared nativeRMOT/Basecategory supervision, no600metric claim |
| B5 distilled Compact | three matched300step controls/full1203video complete; semantic joint1200 and matched broad-fit interventions complete | matched SetGT/MLP/soft-distilled; short RMOT transfer remains poor, TAOBase-only |

Every run records date/commit, devices, split, checkpoints, steps/loss,
official metrics, full FPS, peak memory, failures and next action. Empty
metrics are pending work, never inferred performance.

Evidence controls V1/V2/V3 and their negative results are recorded in
outputs/groundjev/metrics/mot17_first4_sft_evidence_controls.json. No zero
IDSW result from a collapsed predictor counts as good association.


## Native adaptation / legal task scope registrations

- native_ground_features:real DLM visual/projector reference match and ROI
  coordinates; not model tracking performance.
- native_temporal200:actual frozen native-backbone adapter training,seed0,
 693632changed parameters, exact reload. Calibration131/134 matches native.
- rmot_single_empty_pretrained_v1 and rmot_single_empty_simple_fixed:legal
 v2validation0007,8frames, same native candidates; corrected target coverage4/5.
- pretrained_v1_registered_v1/rmot_multi_empty:12real frames, Full/Simple
 completed; evaluator first failed on absent empty-frame YOLO files. Reused
 predictions and fixed target-only lookup; no GT/value/scope changes.
- pretrained_v1_registered_v2_evaluation_repair:reuses completed units and
 runs TAO4frame full1203vocabulary scope plus exact-candidate control/TETA.
- mot17_first4_pretrained_v4_native_temporal200:failed56426token history
 request at303 after1complete frame; all artifacts retained.
- mot17_first4_pretrained_v5_native_temporal200:registered CSV milli-units
 numeric evidence, real native features, unchanged4frame native proposals;
 actual result pending. No source crop-only SFT recipe or32768limit changed.

Full40TAO annotation-frame and full800-frame query scopes are registered for
subsequent runs. Short scope scores never become full-dataset claims. Physical
TAO FPS missing is an explicit limitation. Pilot and final goal remain active.


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
are active; first native label remains an explicitly unlearned class baseline.
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
Both failed units preserved, new real-gradient check active onGPU7. Four
physical GPUs4/5/6/7max; foreign UDMT jobs untouched. Dense/referring retention,
exported bundle inference and formal adaptation remain required.


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


### Native class decisions, actual detector rollout, and serialization repair

Opt-in native_hierarchical category inference now shares the actual OneJev
engine/parameters with Full identity decisions. All1203input categories are
represented in128class root groups; selected-group leaf and explicit UNKNOWN.
This is approximate hierarchical routing, not a flat global MAP classifier.
Native path probability remains separate from identity probability and absent
native detector confidence. Every306exact native geometry is retained.

Completed Simple/Full TAO controls:33.042/23.320TETA, bothClsA0, unchanged
localization/association compared to native-label controls.299/306objects
choose rootUNKNOWN; seven receive vehicle labels. Full and Simple category
decisions are exactly equal; all identity probability differences are0.
Actual four-frame videos, six-decimal probability rounding, all candidates,
GT/empty/rejected frames and unmodified TETA verified. Native category
inference is implemented but does not solve classification; it remains
opt-in. No OVMOT gain or Novel result claim. Next step is legal category
supervision and targeted visual/group-routing diagnosis, not GT relabeling.
Receipt:outputs/groundjev/task_controls/native_category_v1/aggregate_results.json.

Actual GroundAnything proposals now feed a finite48frame legal own-history
training exporter on physical7, original DLM service6. Actor is the valid
OneJev100step checkpoint, V3 crops/motion. No GT proposal substitution,
synthetic fragments, topK/NMS or corrected actor state. Independent GT
matching supplies training targets only; lifecycle labels use targets matched
to actual current proposals. Record only accepted context groups and retain
all candidates including empty/rejected native frames. This closes a genuine
training-data gap; corpus owner remains active, not completed training.
Plan:configs/groundjev/native_candidate_rollout_20261014.json.

Full native DanceTrack0004 predictions completed1203frames,27rejected frames
(2.244%) retained. Original20frame resume means end_to_end_fps remains null.
Official full-video metrics remain pending the exact-candidate Simple control.
Simple recompressed all historical PNG crops every frame:actual155track
profile3.283s,3.256s in465PNG saves. Exact per-field cached encoding retains
at most3encodings per live track and releases them with the track. Actual
223track snapshot:4.487s cold/.003874s unchanged warm call;20,884,556cached
encoded bytes.32frame recorded causal replay, cold/warm payload and fresh
reload are exact. These are serializer timings, not whole-video speed gains.

Only verified owned Simple child4127910 was interrupted after its valid
200frame prefix; original failed owner and interruption receipt preserved.
Fresh owner reuses all complete native Full outputs and restores the entire
unchanged Simple prefix. First reuse preflight wrongly expected the newer
multitask specification field in the older ordinary-MOT receipt; retained
failure and fixed using actual registered paths/frame numbers/native prompts,
not synthesized metadata. v5 owner is active with cached PNG export, no new
native detector calls or changed experimental protocol.
Full/Compact/matched trained controls, mixed task supervision, full original
GroundAnything retention and formal500step checkpoint remain required.


### Full DanceTrack0004 baseline is now official and validated

Both1203frame video artifacts and exact candidate equality passed. Native
Full V1 HOTA.0214546/IDF1.00482694/IDSW3792/FP9300/FN241; same-candidate
Simple HOTA.0852976/IDF1.0395616/IDSW746/FP9334/FN256.27native rejected
frames (2.244%) stay in every official denominator. GT4586observations,
native13664candidates, one-to-one IoU.5 coverage4341/4586=.946577.
Full correct surviving identity associations234/4330=.054042; correct
option coverage3668/4330=.847113.9323unmatched candidate diagnostics
include official ignore regions and are not themselves official FP labels.
These negative results show both excessive candidate output and weak
pretrained identity decisions; detection recall alone is insufficient.

The complete200frame Simple prefix matches its preserved source row for row.
Original Full20frame resumption and Simple200frame resumption preclude a
continuous E2E claim. Simple32.62198replayFPS excludes native detection and
full persistence/rendering costs. Scalar diagnostics never modify output.
Receipt:outputs/groundjev/pilot_runs/dancetrack0004_pretrained_dense_v5_cached_memory_control/aggregate_results.json.

A fresh complete1203frame V3 native run is now active on physical6. Same
pretrained OneJev/original DLM, decoder and official evaluator, real current
candidate/history crops and motion evidence, explicitNO_MATCH, no resumed
prefix or Full candidate replay. Native context splitting retains all boxes
and up-to32identity options. Exact cached PNG export preserves checkpoint
payloads. Completion will measure actual complete pipeline E2E speed; no
performance or full-goal completion claim while active. Formal500step
training remains the original live owner; no restart or shortened protocol.


### Actual native detector own-history corpus completed and validated

48declared legal training frames,1184actual native proposals,5661supervised
rows:1184identity,1184update,1184reliability,69withheldNO_MATCH,2040lifecycle.
All typed targets/options/media and training split membership validated,
no GT IDs in model state, no GT state correction or synthetic fragments.
Native GT matches:474/526MOT17,110/110DanceTrack,496/806MOT20. Uncovered
GT remains explicitly counted; never invented as native proposals. This is
decision supervision, not localization supervision for missing boxes.
Actor outputs only74across48frames (31/12/31first-frame births, subsequent
collapse), exposing the valid100step actor's negative transfer rather than
hiding it. All raw own histories are retained.31,045,329bytes of crop evidence,
no full source-dataset feature cache. Corpus completion is not weight update
or formal training completion. Receipt:
outputs/groundjev/training_data/native_ground_own_history_v1/corpus_validation.json.


### Actual trained Compact and matched baselines (2026-10-14)

All48legal native detector/OneJev100own-history train frames are packed as
80base groups plus69declared option-withholding augmentations,149examples.
Native DLM ROI2560and uncorrected anchor/recent/long feature history; noGT
state correction. The removed teacher identity mass is projected toNO_MATCH
only in the declared training augmentation. Native pack131,123,485bytes,
not a whole-source-dataset feature cache.

Three actual300step/seed0 models completed with finite nonzero gradients,
changed weights, exact checkpoint reloads. Compact distilled and GT-only
set control each930,829parameters; feedforward distilled control929,805,
ratio1.001101. Two set blocks provide interactions; feedforward control has
no set attention but shares mean context, evidence and structured heads.
All support dynamic identities/NEW/IGNORE/NO_MATCH, update, reliability and
lifecycle. They are trained structured decision heads, not direct pretrained
OneJev decoding. GT supervision plus.2native recorded Full teacher loss;
identical set GT-only control omits only that loss. Auxiliary visual backbone
remains original frozen DLM. Parameters for all models changed, reload0,
candidate/history permutation error<=7.63e-6. Allocation peaks about40.5MB
are core-only, excluding CUDA runtime and shared native visual/DLM services.
Initial Compact training is MOT-only: task-index conditioning exists, but
joint expression/class training and full unified-task/long-memory proof remain
required. Never present natural set capabilities as JEV-exclusive novelty.

Same real128native proposals/ROIs,4MOT17development frames and unchanged
TrackEval, actual4frame videos validated:

| Core | HOTA fraction | IDF1 | IDSW | FP | FN |
|---|---:|---:|---:|---:|---:|
| Compact distilled set | .669111 | .738197 | 1 | 9 | 46 |
| Same set, GT-only | .576415 | .638889 | 8 | 3 | 57 |
| Matched distilled feedforward | .634360 | .703196 | 3 | 3 | 54 |
| Existing plain Simple reference | .748813 | .836502 | 1 | 17 | 24 |

Distillation helps the same architecture in this tiny single-seed control;
Compact still trails Simple and has more missed targets. ReplayFPS4.358/
3.959/4.512exclude detector/native feature extraction and full export costs;
no E2E or full-video accuracy claim. Receipts in
outputs/groundjev/compact/native300_controls_v1/aggregate_results.json;
trained checkpoints and SHA receipts in outputs/groundjev/compact/native300_v1.

Registered full1203frame DanceTrack0004 Compact and two controls reuse every
original native detector candidate, unchanged official GT and scorer. Stream
ROI once through physical5native feature service; controls reuse identical
ROI values on physical7. Source frames are read progressively rather than
loading the complete replay into memory; actual4/1203row equivalence passed.
These are candidate/feature replay controls, not detector-inclusive E2E.
First run failed at actual empty-proposal frame21: empty ROI pool returned
BF16 whereas nonempty interpolation and optional temporal layer useFP32.
Preserved20complete frames and failure; fixed only empty output dtype.
Only verified owned native feature service4043767 was restarted on5. A
real frame21 empty RPC/lifecycle probe passes and real frame20nonempty ROI
max difference after restart is0. A probe initially used the wrong timestamp
constant; it was corrected to the registered20Hz clock, with no data/clock
protocol changes. Fresh v2 owner restores all20native prediction/state rows
and resumes frame21, without dropping it. Original OneJev500/Ground/freshV3
and foreign UDMT remain untouched. Full metrics are pending its terminal
owner. First model frame latency includes ROI RPC; later controls read fixed
ROIs, so do not claim a direct comparable speed ranking across those latencies.

## Formal native Ground adaptations and current strict sampling controls

Native200complete checkpoint chain and120additional real-absence updates are
validated (200+120native320). Complete569tensor models preserve non-projector
weights; all failures remain. Shared-service short scores are descriptive:
original200: MOT75.250->84.553%; RMOT26.866->25.277/24.780->23.959%.
Fresh200/320: MOT83.737/83.402%; RMOT18.406/20.046 and23.496/24.817%.
Absence outputs unchanged7/3; no successful absence remedy established.

The original native RNG adds monotonic request_index to sampling_seed42;
shared full9services consume unrelated calls and do not pair effective seeds.
Strictisolated96plan resets native services percheckpoint with identical24call
order: original/200/200repeat/320. Require identical repeated200raw outputs,
candidates and predictions. Firstoriginal24unit is complete; transient port
reuse failure preserved, v2waits TCPport release and reuses only that complete
unit. No source/metric/seed/GT edits; full9baseline/adapted200 owners untouched.

One500 completion-controls plan will verify actual nativeMMreload and fourway
Simple/pretrained/init100/500 with exact native proposals acrossMOT4/RMO8/
RMO12/TAOBase4. It reusesGPU4 after existingtrainerexit without restarting it.
Complete500video comparison, actual-native-proposal SFT, Novel-diverse/fulltask
scopes, long-memory evidence/update ablations and fair complete speed/resource
comparisons remain required; no final research completion is asserted.

B2 registration: configs/groundjev/u2_core300_baseline_v1.json. Same48existingnativeMOTfitframes/300updates, but different originalU2 losses, parameter count, optimizer andclip exposure versusCompact; notFLOP/parammatched. NooldU2queue/perception rerun. OriginalSharedTrackCore/assignment/state/losses unchanged; CPUFP32 bridge/core training with sameGTignore/distractorreader. ExactsameoriginalV1DTcandidate/ROIfile fornewbaseline. Scope MOTonly. Owner: baselines/u2_core300_cpu_v1.

TAOBaseNovel63 Simple full40 annotationframes now complete, all35228nativeboxes/160GT preserved, officialTETA18.640 (Base23.3725/Novel9.1737). NovelClsA0; thisonevideo isnotfullTAO. SharedCompactsame-source control stillpending. Summary: metrics/tao_base_novel63_full40_simple_summary.json.


TAO63 full40 raw-candidate controls now bothcomplete: Simple TETA18.640/
AssocA28.618/ClsA20.290 versus sharedCompact14.517/5.1478/30.314. BaseTETA
23.3725/18.93545 andNovel9.1737/5.679; bothNovelClsA0. No modelpromotion.
Raw35228candidates remain. GTfreeaudit shows7343exactgeometrygroups/27885aliases;
a separate CPU exactalias proposal-policy control is registered andrunning,
notanassociation-only comparison. FullRMOT two800frame freshCompact/native
andmatchedpretrained/500/teacher600/native600 controls wait verifiedGPU5handoff;
configs/groundjev/full_rmot_study_v1.json. Fullgoalremains active.


ExactaliasTAO40bothcomplete: SimpleTETA22.556 vsCompact21.773, comparedto
raw18.640/14.517. BothNovelClsA0; no promotion. All35228aliases/7343groups,
40frames preserved, rawandgroupedproposalIoU.5coverage147/160unchanged.
Separateproposalpolicy, notisolatedJEVgain; source402723754bytes/1GiBcap.

Fourtask-subset1200CPUtrainingscomplete, sameactualsharedMOT300+seed0heads,
1146511trainablecoreparams, same536pack. Common400taskdrawprefixes exact,
all3taskreloads0. Fullvideo controlsrunning; onlyMOTonly1203reportedsofar:
HOTA19.6386/IDF120.7863,IDSW348,FN1765. Parametercount excludesfrozenOne
texttable andnativeGround. RMOsourcepointercontractfixedusingexistingexact
ROIlogs; replaceonlyCPUmanager,preservecurrentMOTworker/checkpoints, no retrain.
Currentowner: compact/task_subsets1200_cpu_v2_feature_sources/owner_state.json.
TeacherFull600trainingalso complete; native600/official600controls pending.
RestoredFP32stateexact butoverlaplossdelta.001735 disprovesbitwisereplayclaim;
curatedresumeelapsed/throughput scope excludesfalsecumulativerate.
