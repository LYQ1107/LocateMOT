# L88 Preregistered Plan — RMOT-Aware Grounding Adaptation

## Identity and status

- Project root: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`
- L88 worktree: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L88`
- Luna thread: `01a02014-fce8-7f51-8414-e7ed6ab44745`
- Branch: `codex/l88-rmot-aware-grounding-lora-20260904`
- Starting code commit: `0f5d8e9cf5b7d31966104cf06302630011580601`
- L86 base commit: `97bff208929474d4c4b0d659c80e7eba2f3f5d0a`
- Fixed manifest SHA256: `06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`
- L87 common evaluation policy SHA256:
  `46a11807d28007e193390ab85899f184cfbd1ac8acfd0fb98c95b83c66276943`

L88 is one RMOT-only representation adaptation experiment. It will not change
ordinary MOT, OVMOT/TAO, UIDM, the L69 bank, tracker code, frozen checkpoints,
or the fixed manifest. It will stop after the registered internal evidence and
wait for supervisor review; it is not permission to read screening or official
test labels.

## Evidence motivating the experiment

L87-A corrected the temporal negative target-bag contract and completed 40
epochs, but fixed validation recall was `0.4516129`, multi-positive recall
`0.6000`, and inactive false acceptance `1.0`; its internal HOTA was `28.5752`
(V1) and `22.1300` (V2). L87-B corrected deployment/reselection without
training, but fixed validation recall was `0.2903226`, multi-positive recall
`0.5000`, and internal HOTA was `27.8320`/`20.1233`. L87-A improved V2
association but did not improve detection sufficiently. The L86
GT-privileged oracle references are HOTA `66.8724` (V1) and `53.9669` (V2),
so the evidence leaves an upstream grounding/correspondence gap. These are
the accepted reasons to test representation adaptation; L87 remains frozen.

## Single hypothesis and change boundary

**Hypothesis:** frozen Z1 contains query-sensitive structure, but the
pretrained GroundingDINO cross-modal representation is not adapted enough for
fine-grained, multi-target RMOT discrimination. Zero-initialized low-rank
adaptation of the final two bidirectional vision-language fusion blocks and
decoder layer 1 should improve target-bag separation and full-video DetA/DetPr
without sacrificing L87-A's temporal path.

The only new scientific variable is trainable GroundingDINO LoRA. The visual
backbone, BERT body/text backbone, bbox heads, reference-point head, decoder
layers 2–6, L69 candidates, L86/L87 sidecar architecture, losses, sampler,
seed, and deployment protocol remain fixed. No external training data,
synthetic counterfactual text, rank sweep, threshold repair, top-k/NMS, or
tracker change is allowed.

## Frozen inputs and output locations

Read-only inputs:

- L69 budget-40 complete candidate bank and its native frame pointers;
- L86/L87 causal observation/history fields and existing L85 compact Z1/text
  cache for parity and sidecar context;
- L49 V1/V2 fit labels, L82 video-disjoint fit/dev split, and fixed internal
  validation labels only under the registered selection/evaluation order;
- local GroundingDINO config/weight/BERT runtime and the local MMDetection
  v3.3.0 reference tree;
- fixed manifest above.

New code is limited to `locatemot/rmot/l88_*.py` and `tools/l88_*.py`.
Reports and compact metadata are under `reports/l88/` or `reports/l88_*`.
Training/cache outputs are under `outputs/l88/`. Because `/data1` has about
32 GB free and is at 100% allocation while `/data2` has about 1.2 TB free,
the query-independent encoder-input cache and L88 outputs will target
`/data2/usr_for_deadline/locatemot_l88/project_outputs` through a recorded
`outputs/l88` symlink if the preflight confirms the target. No raw image,
dense attention, or full-model checkpoint is written.

## LoRA contract

The target manifest will be frozen before training. It must discover six
encoder fusion layers and target `[fusion_count-2, fusion_count-1]` (expected
layers 4 and 5), and decoder `layers[0]` only. Rank is `16`, alpha `32`,
scale `2.0`, no adapter bias, and B is exactly zero at initialization. A is
standard Kaiming/uniform initialized. LoRA uses a local weight
parametrization actually consumed by the live forward, including PyTorch MHA
`in_proj_weight`/`out_proj.weight` where applicable; no naive child
`out_proj` wrapper is accepted.

Authorized weights:

- fusion layers 4 and 5: `attn.{v_proj,l_proj,values_v_proj,values_l_proj,out_v_proj,out_l_proj}.weight`;
- decoder layer 0 self/text attention MHA projection matrices;
- decoder layer 0 deformable attention projection weights when present;
- decoder layer 0 FFN linear weights.

The exact live names/shapes and parameter counts are recorded in
`outputs/l88/preregister/lora_target_manifest.json`. All base parameters stay
frozen; every trainable GroundingDINO parameter must be an L88 A/B factor.
The base model remains `eval()` while autograd remains enabled for LoRA.

## Representation/runtime contract

The query-independent cache stores only
`pre_transformer(extract_feat(...))`: `feat`, `feat_mask`, `feat_pos`,
`spatial_shapes`, `level_start_index`, and `valid_ratios`, in float32 unless
the measured disk budget requires a documented compact dtype. It contains no
labels, query strings, semantic scores, or query IDs as features. Frames are
deduplicated by dataset/video/frame across fit, video-disjoint dev, fixed
semantic units, and internal V1/V2 validation.

For each query tile, canonical GroundingDINO caption construction and frozen
BERT language encoding feed all encoder layers. The complete L69 candidate set
is retained. Candidate boxes are normalized under the audited L82/L84 rule,
pooled from adapted encoder memory, combined with the existing reference
position and original-content seed, and sent through decoder layer 0 only.
Decoder layers 1–6, reference refinement, bbox branches, and native top-k
proposals are not run for L88 Z1.

L88 sidecar is the L86/L87 factorized model architecture initialized fresh with
seed `20260829`. Inputs are adapted Z1, frozen text/frame globals, current L69
observation, and causal L69 history. Its loss is exactly the L87-A/L86
objective:

```text
1.00 faithful semantic target-bag loss on r_total
0.30 faithful semantic target-bag loss on r_static
1.00 target-bag membership loss
0.50 presence loss
0.50 candidate-vs-NULL competing loss
0.10 corrected temporal identity loss
0.01 temporal delta regularization
```

Temporal negatives remain
`(previous_available | current_available) - referred_targets`, with real
candidate-GT target bags only. No synthetic objectness negatives are added.
Token/span-to-region and static/motion alignment remain `UNALIGNED`.

## Mandatory pre-training gates

1. Compile all L88 modules/scripts.
2. Run one real differentiable fit-group smoke with B=0. It must verify
   LoRA/base/sidecar gradient separation, finite values, target manifest,
   candidate rows, and no future history.
3. Run one zero-init L88 Z1 parity group against the existing L84/L85 Z1
   representation: same order/shape/finite and max absolute difference at
   most `1e-3`. A parity failure blocks training and receives one minimal code
   correction only.

After both pass, testing stops and the registered cache/training flow begins.

## Training preregistration

- V1+V2 L49 fit supervision only; no external data;
- seed `20260829`;
- S epochs 1–8 (temporal off), T epochs 9–20 (4-frame causal clips), J epochs
  21–40 (full joint objective);
- target effective frame-group batch `8`; world 4 if four free GPUs exist,
  otherwise use available 1–3 and record deterministic accumulation;
- query tile `4`, reduce only to `2` if OOM and record the deviation;
- BF16 autocast, FP32 master LoRA/sidecar/optimizer, gradient clip `1.0`;
- AdamW, sidecar lr `2e-4`, LoRA lr `1e-4`, weight decay `1e-2`/`0`, betas
  `(0.9,0.999)`, 5% warmup, cosine decay;
- exactly 40 epochs, even-epoch checkpoints 02 through 40;
- no early stop for weak metrics; stop only for technical invalidation.

The intended four-GPU command is one blocking `torchrun` command with one rank
per free physical GPU. It must record actual mapping, world size, accumulation,
wall time, throughput, peak memory, code SHA and cache target.

## Dev selection and fixed evaluation

Build legal video-disjoint fit-dev full-video sequences from
`outputs/l82/protocol/fit_video_train_dev_split.json`, using only fit labels
after each candidate strategy is frozen. Score all 20 even checkpoints on the
existing 138-group dev set as a shortlist. The shortlist is the deduplicated
set `{epoch08, epoch20, epoch40, best target_bag_f1, best distinct_target_recall}`
subject to target-bag precision `>=0.08` (maximum five; if no recall candidate
meets the floor, use the highest-recall checkpoint). For each shortlist
checkpoint, fit Rule B/R/P from dev target-level metrics and run full-video
dev TrackEval for the three rules. Select by higher dev HOTA, then DetA, AssA,
distinct target recall, lower inactive false acceptance, earlier epoch. This
selection is frozen before fixed validation labels are read.

Candidate grid:
`[-1,-.75,-.5,-.25,0,.25,.5,.75,1]`; presence uses the same grid; NULL
margin is `[0,.25,.5,.75]`. Rule B maximizes target-bag F1 with inactive
emissions counted as false; R maximizes distinct recall among precision `>=.08`;
P maximizes precision among distinct recall `>=.60`. Ties follow the exact
ordering in the L88 prompt. Target-level and legacy row metrics are separate.

Then evaluate the fixed 16 calibration + 24 validation units exactly once.
Only legacy metrics use the historical numeric guardrails; target-level metrics
are primary diagnostics. The fixed semantic result never authorizes bypassing
full-video HOTA, but no screening/official-test labels are read in L88.

## Pre-registered result descriptors and stopping

Material full-video improvement is recorded only if both V1 HOTA `>=31.5752`
and V2 HOTA `>=25.1300`; this is a descriptor, not an early stopping rule.
Interpretation will distinguish DetA/DetPr gains, AssA-only gains, dev
overfitting, and no material gain. The final stage always stops after V1/V2
internal TrackEval and reporting, with no automatic rank increase, extra
epochs, STORM/COAL data, phrase labels, tracker change, screening, or official
test.

## Required flags and artifacts

Every L88 machine artifact will carry:

```text
screening_gt_used=false
official_test_labels_read=false
ordinary_mot_ovmot_touched=false
candidate_deletion=false
candidate_truncation=false
z1_layer_selection_changed=false
groundingdino_lora_used=true
groundingdino_backbone_trainable=false
bert_body_trainable=false
bbox_head_trainable=false
decoder_layers_2_to_6_trainable=false
token_span_region_alignment=UNALIGNED
static_motion_alignment=UNALIGNED
```

Required preregistration artifacts are `reports/l88/L88_PREREGISTERED_PLAN.md`,
`outputs/l88/preregister/config.json`,
`outputs/l88/preregister/lora_target_manifest.json`, and
`outputs/l88/preregister/source_of_truth.json`. Required final artifacts are
`reports/l88/STAGE_L88_FINAL_REPORT.md`, reports for training/selection/semantic
evidence, complete machine-readable provenance/status, checkpoint hashes, and
failure decomposition. Code will be committed and pushed before the final
handoff.

## Literature boundary

The implementation audit will record primary URLs/tags for FlexHook, STORM,
COAL, DKGTrack, PropVG, ReferDINO, VMRMOT, and the MMDetection v3.3.0
GroundingDINO reference. These sources are structural context only unless
code is explicitly read and reused. L88 does not import their checkpoints or
data and does not claim token/span or motion-language supervision.
