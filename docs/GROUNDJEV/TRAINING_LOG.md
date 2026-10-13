# Latest training status — 2026-10-14

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
