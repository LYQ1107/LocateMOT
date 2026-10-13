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

GroundJEV uses physical4and6 within allowlist4/5/6/7; other projects untouched.
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

GroundAnything tracking-aware gradient adaptation and native hidden-feature
export remain pending; current historical visual evidence is honestly labeled
real crops. Full-video results, formal500-step before/after, true detector
rollout, legal RMOT/OVMOT, Compact/matched controls, long-memory ablations and
joint-task research remain incomplete. Pilot completion will not complete the
full unified online tracking goal. Old U2 perception recall is not MOT HOTA.
