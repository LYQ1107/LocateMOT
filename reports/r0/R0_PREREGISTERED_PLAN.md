# R0 Track-Centric Grounding Hook — preregistered plan

## Identity and base

- Luna thread: `01a02014-fce8-7f51-8414-e7ed6ab44745`
- Project asset root: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`
- Isolated worktree: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_R0`
- Branch: `codex/r0-track-centric-grounding-hook-20260909`
- Base: `7fa201e4967f71a42ba669e90f330a6e3979e1cd`
- Fixed manifest: `outputs/l19/protocol/kitti_fast_eval_manifest.json`
- Required manifest SHA256: `06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`

This is a new RMOT-only sidecar. The original worktree and all existing L11–L89E
assets remain read-only. No ordinary MOT, OVMOT/TAO, UIDM, tracker, L69 bank,
GroundingDINO weights, old model, or production entrypoint will be changed.

## Scientific question

L89E left V1/V2 internal TrackEval at 28.7628/21.8385 HOTA with fixed semantic
recall/multi-positive recall 0.4516/0.3611. R0 tests whether the principal
ceiling is the compressed Z1 observation rather than the frozen L69
candidate/tracker ceiling. The registered hypothesis is that query-independent,
multi-scale GroundingDINO visual features pooled into inner/context tokens,
combined with causal track geometry and pure language, can provide a richer
candidate observation for active pairwise correspondence and a complete
candidate-set reasoner. R0 does not use Z1, L89/QSC-D, or an additional tracker.

The final labels are preregistered:

- `R0_BREAKS_Z1_LOCAL_OPTIMUM` iff V1 HOTA >= 34 and V2 HOTA >= 28.
- `R0_PARTIAL_BREAKTHROUGH` iff at least one benchmark improves >= 3 HOTA
  points over L89E but the joint minimum above is not reached.
- `R0_VISUAL_OBSERVATION_STILL_INSUFFICIENT` if neither condition is met and
  the contracts remain valid.
- `R0_DATA_CONTRACT_INVALID` for a data/key/label/provenance violation.
- `R0_ENVIRONMENT_BLOCKED` for an external runtime, storage, or dependency
  blocker that cannot be repaired within the registered contract.

These are evidence classifications, not a claim of ordinary-RMOT completion;
the ultimate target remains V1/V2/Dance 45/40/42.

## Frozen inputs and legal scope

Read-only inputs are the immutable L69 budget-40 feature/dual banks, L49
train/calibration/validation unit metadata, the already verified pure language
cache (reuse missing entries only through the registered frozen text path), the
L82 GroundingDINO runtime/config/checkpoint/BERT, L69 frame universe, and the
fixed manifest. R0 training is benchmark-specific:

- R0-V1 fit/train videos only: `refer_kitti_v1` train scope;
- R0-V2 fit/train videos only: `refer_kitti_v2` train scope;
- no unified V1+V2 head and no dataset ID input.

Legal dev/internal scopes are the existing L82/L89E train/dev and internal
validation video scopes. Screening and official-test labels are never read.
The R0 V1/V2 train-video union is the previously audited L69 set; exact video
scope and every native frame ID will be recorded in the dense-index reports.

## Registered architecture

1. One frozen query-independent GroundingDINO `extract_feat` pass per native
   frame produces exactly four `[1,256,H,W]` maps. `r0_visual_tokens.py`
   samples every L69 candidate at 3x3 inner and fixed 1.75x 3x3 context grids
   with `grid_sample(mode=bilinear, padding_mode=border, align_corners=False)`.
   All 72 tokens per row are retained; no top-k/NMS/deletion.
2. `R0CandidateVisualEncoder` uses the fixed level/region/grid embeddings, a
   zero-initialized learnable observation token, and exactly two 256/8-head,
   FFN-1024 transformer layers to produce a per-candidate summary.
3. `R0TrackGeometryEncoder` consumes only normalized current/causal previous
   boxes as five 10-D steps and a unidirectional GRU. Track IDs select history
   rows only and are never model inputs.
4. Two fixed pairwise correspondence blocks perform text-to-pair and
   candidate-visual cross-attention, followed by an FFN. Two fixed candidate
   set transformer layers operate over all current-frame rows.
5. The head outputs one membership logit per query/candidate and one coverage
   presence logit per query. Deployment is fixed at membership >= 0 and
   presence >= 0; no threshold fitting, NULL margin, top-k, NMS, or tracker
   change is permitted.

The exact config is `dim=256`, `heads=8`, visual/pair/set layers `2`, FFN
`1024`, dropout `.10`, visual levels `4`, grid `3`, region types `2`, geometry
`5x10`. The only trainable parameters are the R0 sidecar modules. GroundingDINO
and the language cache are frozen; no L89 temporal delta/gate or QSC-D branch
is loaded.

## Data and supervision contract

Visual cache rows are built before labels and contain no labels/targets/GT.
Native L69 `frame_ids/frame_ptr` construct every candidate row. Every row keeps
`dataset|video|frame|row_offset`, candidate index, track ID, pool ID and raw rank
as provenance only. Track geometry is causal (`history_frame <= current_frame`)
and zero-padded to four prior steps. The dense train index then attaches the
existing expression-level L49/L69 labels. `present_uncovered` membership loss
is masked; it is not converted to inactive. Inactive rows receive explicit
negative membership and coverage target 0. Multi-positive target bags all
remain positive.

The only registered loss is
`membership_set_loss + 0.50 * coverage_presence_loss`: covered positive and
negative bag classification, smooth weakest-positive loss with `tau=0.20`, a
hard-negative set margin, and balanced query-level coverage presence BCE. No
external labels, teacher scores, cross-expression negatives, extra temporal or
contrastive term may enter.

## Execution gates and fixed choices

1. Implement only the listed R0 files and boundary guard. Compile affected
   files, run the guard, then run exactly one real train-frame prompt-invariance
   plus forward/backward contract smoke. That smoke must show four finite
   visual levels, exact prompt invariance, 36/36 inner/context tokens, unchanged
   L69 rows/order, label attachment only after visual construction, finite R0
   forward/loss, nonzero visual/pair/set/membership gradients, and no detector
   or tracker parameter in the optimizer. This is the minimum necessary smoke.
2. Audit pure language-cache coverage for legal train/dev/internal queries.
   Build only missing label-free entries if any; never rebuild existing items.
3. Build the query-independent visual cache under the prescribed `/data2`
   root, sharded by legal `(dataset,video)` only. Cache contains compact FP16
   72-token rows and provenance, never raw feature maps, pixels, labels, or GT.
4. Build V1 and V2 dense native-frame train indexes separately. Reject any
   non-fit or official/screening video.
5. Train each benchmark-specific head for exactly 12 epochs x 4000
   frame/query-tile microsteps, seed `20260909` as fixed by the R0 CLI, AdamW
   lr `1e-4`, wd `1e-2`, clip `1.0`, warmup `5%`, cosine, BF16 if safe, with
   checkpoints at epochs 2/4/6/8/10/12. Use all native train frames with
   nonzero probability and deterministic 3 multi/2 single/2 inactive/1
   present-uncovered query tile when available. Use up to four GPUs only if
   free; use prescribed gradient accumulation for fewer GPUs.
6. Score all legal dense dev frames/queries at the fixed zero-logit rule. Build
   at most a three-checkpoint shortlist per benchmark using the preregistered
   dense-dev criteria, then run true-full-video TrackEval only for that
   shortlist. Do not use fixed/internal labels for checkpoint selection.
7. Freeze one selected checkpoint per benchmark, run per-domain fixed semantic
   diagnostics post-selection, then run only the prescribed internal V1/V2
   full-video TrackEval. No screening or official test is part of R0.

## Resource and stop rules

Use `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, at most four GPUs and one process
per GPU. Check `/data1` and `/data2` before cache creation. If predicted cache
or checkpoint growth exceeds available storage, stop with `R0_ENVIRONMENT_BLOCKED`
and preserve completed shards. No full-model checkpoint or raw/dense debug
cache is allowed. Each long run is one blocking shell command; after it returns,
inspect outputs once, fix only the first actionable failure, and run one
targeted regression before resuming.

## Boundary and reporting

`tools/r0_boundary_guard.py` is required to reject any path outside the exact
R0 allow-list. Final reports separate source/interface audit, visual cache,
dense data, fit training, dev selection, fixed semantic diagnostics and
internal TrackEval. Every machine-readable output carries:

`screening_gt_used=false`, `official_test_labels_read=false`,
`ordinary_mot_ovmot_touched=false`, `tracker_source_changed=false`,
`uidm_source_changed=false`, `l69_source_changed=false`,
`production_entrypoint_changed=false`.

After internal TrackEval and the final boundary guard, stop with
`STOPPED_PENDING_SUPERVISOR_REVIEW`; do not launch any automatic follow-up.
