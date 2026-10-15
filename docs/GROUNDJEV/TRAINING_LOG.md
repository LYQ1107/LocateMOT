# Current training scope — 2026-10-15

Cross-task shared-model fine-tuning has run; large-scale, long-duration,
end-to-end joint optimization of GroundAnything and OneJev has not. Ground
adaptation and OneJev SFT are separate optimizer runs, subsequently composed
for tracking. There is no joint gradient through both foundations and the
online association/memory pipeline. Elapsed evaluation/cache time is not
training time, and corpus rows containing repetitions are not unique frames.

* Full teacher-history and native-own-history branches each retained600
  additional updates/2400sampled examples from the completed MOT500 model.
  Their saved FP32/Adam/scheduler/RNG state and actual native-MM reload are
  validated. Sources: `training/onejev_unified600_{teacher,native}_v2/`
  `resumed_completion_validation.json`, relative to `outputs/groundjev/`.
  The3595.081s teacher and8767.184s native timers cover only resumed tails
  (150/350updates), not complete600-stage or lifetime training duration.
* Shared Compact has a1200-step multi-task stage followed by1200steps on
  native own-history data; this remains a small fixed-corpus adaptation.
  Configurations: `configs/groundjev/compact_joint_semantics_20261014.json`
  and `configs/groundjev/memory_full1203_20261015.json`.
* Ground tracking200 trains27,267,584native projector parameters on112rows/
  56pairs/72images. Ground320 adds120updates with40fit absence rows; the
  localization-only200 control preserves the original200pair schedule.
  These are small controlled adaptations, not joint foundation training.

Specification-conditioned persistent memory, uncertainty-aware identity
updates and Full-to-Compact transfer are research hypotheses. Existing typed
OneJev heads, memory and distillation alone do not establish novelty. Completed
controls show mixed/negative transfer, and ordinary-model-level performance
on all three tasks remains unachieved; see `CURRENT_STATUS.md` and
`FINAL_GOAL.md`. No unique JEV advantage is established.

Next declared intervention remains the bounded current-native60048-frame fit
rollout, serialized after the original Full1203owner. It reuses all1184actual
native candidates without correcting actor history from GT. Review real
existing-ID and known-conflict support before launching further training;
assigned-ID-conditioned update supervision requires an explicit matching
training/runtime question, not relabeling the old observation-only question.
No additional training or model invocation is launched by this log correction.

Latest execution update: the original Full1203owner has now exited0 with all
three arms complete; registered48-frame current-actor refresh automatically
started onGPU4. Whole-video MOT500/native600/teacher600 HOTA percentages are
7.848/8.703/7.577 and IDF1 5.056/5.826/5.015. All original V1 proposals and
parser-rejected frames are retained. The teacher branch does not improve this
full video; neither branch establishes stable multi-task gains. No new SFT
is implied by the active refresh. Full cohort official/output validation:
`outputs/groundjev/training/onejev_full1203_controls_v2_wait_fix/`
`completed_units_validated_summary.json`.

## Historical snapshot — 2026-10-14 (superseded)

100-step official full language-model check completed400examples/1509.262s,
peak15.3GiB. All474weights saved; three BF16-aligned MLP probes changed,
two real multimodal reloads agree exactly. Original fork check unchanged.
Same-candidate V1 HOTA worsens.42163 to.32834, IDSW60to75. Preserve the
negative checkpoint and do not equate lower training loss with tracking gains.

Teacher context was limited to18options and had no wrong detections/model
states, unlike production35options. V2 crop/motion evidence collapsed due
to ambiguous NO_MATCH; V3 semantic repair retains substantial missing outputs.
All controls/results remain in the registry.

Two actual V3 actor rollouts completed96legal observations and13,501unique
supervised rows. They include526 contaminated history observations and
explicit synthetic false fragments; GT IDs never correct the actor. GT
proposals remain a declared curriculum source, not GroundAnything predictions.
Mix both branches with4x identity repeats:20,881rows. Official plan max22817
of32768tokens,0dropped.The first500-step run was stopped for203 verified lifecycle target errors:
GT-visible identities absent from actual current proposals need LOST. All
5951 lifecycle rows passed the corrected-proposal regression, preserving
actual states/images/IDs/other supervision. An underscore parsing error
in the first repair was fixed, and that failed preflight was retained.
A fresh corrected500-step finite full-language SFT owner is running onGPU4
from checkpoint100, fresh optimizer as an explicit curriculum stage, unchanged
seed0/lr5e-6. First2steps finite loss/gradient, peak18.1GiB, about76s/step.
Save every125steps, then validate/reload and run the exact before/after protocol.

Actual DLM-proposal rollout and the other task/adaptation studies remain
required; this finite stage does not by itself finish the pilot or full goal.

## Earlier setup history

# Training log

2026-10-13: no GroundJEV training run yet. Old U2 selected foundation and
completed evaluation receipts preserved. GroundJEV dependency setup started;
system Python lacks ensurepip, so bootstrap uses pip targeting the new
environment without modifying LocateMOT's environment.

Setup outcomes: Python 3.12.15 installed; isolated uv setup cache fixed a
300-second lock timeout. OneJev and custom SGLang dependency installs remain
active. Model download transport failures are recorded in CURRENT_STATUS;
completed byte prefixes remain. No training optimizer or GPU inference has
been launched yet.

OneJev install completed in a separate Python3.10 environment using copied,
ABI-matched local Torch2.10 CUDA dependencies and freshly installed
Transformers5.5.4/qev/peft/accelerate. All69 packages compatible; CUDA check on
visible physical GPU6 succeeds. Slow unused Python3.12 OneJev install stopped
after that verification. GroundAnything retains its own Python3.12/Torch2.9.1
recipe and official cuDNN override; remaining package stages use verified
local PyPI wheel transport. Weight-dependent real-frame check is queued;
training steps remain zero.

2026-10-14: actual OneJev weight loading (474 entries) initially failed the
official fork self-check under Transformers5.5.4 (.0857 probability error vs
.05 tolerance). Checkpoint metadata declares5.17.0; upgraded only OneJev's
isolated runtime and retained the check/tolerance. Same checkpoint then passes
auto validation and answers real MOT-frame Choice/Noul/Score questions, native
probabilities retained. Request .286737s, allocator peak3,119,928,832 bytes;
not a video benchmark. GroundAnything's separate official serve setup finished
and passed its cuDNN compatibility check. DLM weights were pending at that check; optimizer steps were0 then.


## Frozen native DLM temporal adapter — actual200steps

Official428,482,048parameter visual/projector frozen;693,632parameter
LayerNorm2560→Linear256→GELU→Linear128 auxiliary adapter trained on legal
MOT17/DanceTrack/MOT20 identity pairs,seed0/AdamWlr1e-3. Loss4.20862→.09104,
finite gradients,693,632elements changed, checkpoint reload maxdiff0. Native
backbone gradients absent; DLM decoder unchanged. Disjoint train calibration
retrieval131/134 for both original native features and trained adapter; this
is GT-region retrieval, not a tracking benchmark. Small transient region
tensors, no full-dataset feature cache. Actual checkpoint metadata is in
outputs/groundjev/ground_adaptation/native_temporal200/result.json.

Region adapter gradients are real; detection/decoder SFT gradients, actual
tracking benefit and dense/referring retention benchmarks remain required.


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

## 2026-10-14 native Ground200 plus explicit absence120

Original27,267,584projector parameters FP32; frozen real native vision/language.
OfficialMDM.5/causal.5 plus .1GTidentity/.1actualdecoded-proposal identity losses,
seed42/lr2e-6/200steps;112rows/56pairs/all72images. Exact200checkpoint chain,
changedFP32/BF16 and reload0 validated; repeatedCUDAillegalaccess attempts
preserved, rootcause not established. Separate nativeTorch2.10cu128/TF5.7/
Triton3.6 runtime, concrete shapes and synchronized native operator documented.

Additional120stage from exactFP32 projector200, explicitAdamWreset/lr1e-6,
retains112rows and adds40realfit absences (152rows/76pairs); all76pairs and40
negatives receive actual training. First-layer native attention forward/QKV
reference checks for dense/positive/absent pass unchangedcriteria after
isolating detached diagnostic checkpoint hooks.120finite updates, exactreload0,
27,266,797FP32/8,256,449BF16changes relative200; peak18,414,616,064bytes.
Native320bundle569tensors/fiveprojector replacements; original models retained.

Shared-service48frame before/after gains/regressions do not pair effective
native seeds becauserequest_index enters originalsampling. Real metrics remain;
no causalimprovement assertion. Strictorderedfresh-service original/200/repeat/
320 control is running, with incompleteportunit failure retained and valid
original24frames reused. Fulloriginal9retention is still pending. Real absence
correction and successful unified tracking are not established by trainingloss.

OriginalOne500statefulreplay continues unchanged; latestpublished observation
300/500, fullFP32/Adam/scheduler/allRNG/cursor state available. A finite verified
completion owner waits the actual trainer, checks500nativeMMreload then runs
matchedthree-task controls. It does not restart or alter the training recipe.
