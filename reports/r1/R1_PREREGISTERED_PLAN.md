# R1 Aligned Track Conditioning Hook — preregistered plan

## Identity and frozen base

- Luna thread: `01a02014-fce8-7f51-8414-e7ed6ab44745`
- canonical asset root: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`
- isolated worktree: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_R1`
- branch: `codex/r1-aligned-track-conditioning-hook-20260910`
- exact base commit: `0bb8e0d3c8bef3218899748b3c2a204e9a2d97c6`
- fixed manifest SHA256: `06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`

The main LocateMOT checkout, ordinary MOT/OVMOT/TAO entrypoints, UIDM,
tracker lifecycle, L69 bank/source, GroundingDINO production path, PBD
paths, TrackEval source, and every L0–R0/L89 asset remain read-only. R1 code
is an isolated RMOT sidecar. No R1 artifact will be merged into production.

## Frozen anchor and decision rule

The frozen L89E Stage-S anchor is:

`/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89/outputs/l89/train/joint40/checkpoint_l89_epoch004.pt`

with SHA256
`5ab3cb344b73b320b34de7b4bb41622ce665ecb17c4d90c1999640a318e69aa8`.
The authoritative selection artifact is
`/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89E/outputs/l89e/dev/selection_attempt1/checkpoint_selection.json`.
The selected Rule-B object has been read from that file: `candidate_threshold`
is `1.0`, `presence_threshold` is `0.5`, and `null_margin` is `0.0`. R1 code
will load and hash this object through one helper; these values are recorded
here only as the resolved audit result, not as a second source of truth.

At zero residual, R1 must reproduce L89E candidate energy, presence, null,
and the exact Rule-B emission mask to `1e-5` with duplicate suppression off.
Failure is `R1_ANCHOR_REPRODUCTION_FAIL` and stops the branch before training.

Before training, legal development evidence will compare `RAW` and
`EXACT_INDEX_DEDUP` under the frozen Rule-B thresholds. The only allowed
emission choice is `EXACT_INDEX_DEDUP` when HOTA is strictly higher, DetPr is
strictly higher, and DetRe loss is at most one percentage point; otherwise
`RAW` is frozen. This is decided before R1 fitting and cannot be switched
after training.

## Scientific hypothesis

R0B showed that pre-fusion raw ROI detail can be useful but replacing the
pretrained aligned L89E semantics reduced internal HOTA to V1/V2
`26.4582/15.6097`, while L89E was `28.7628/21.8385`. R1 tests only whether a
zero-initialized residual, conditioned first on aligned Z0/Z1/Z4 states and
then on language-directed R0 local detail, can learn representative target
bags without destroying the frozen anchor. It does not extend R0B, retune
L89E, alter the bank, or add a tracker.

The registered labels are `R1_STRONG` iff V1 HOTA >=38 and V2 HOTA >=33,
`R1_BREAKTHROUGH` iff V1 >=34 and V2 >=28, `R1_PARTIAL` iff at least one
domain improves >=3 HOTA points over L89E without the joint minimum, and
`R1_NO_BREAKTHROUGH` iff a valid clean run remains around V1<31/V2<24.
These are research classifications, not ordinary-RMOT completion.

## Inputs and cache protocol

Read-only inputs are the L69 budget-40 feature/dual bank, legal R0 dense
V1/V2 train and dev indexes, the R0 query-independent 72-token visual cache,
the pure L89 language cache plus its legal missing-entry supplement, the
L80 current observation implementation, and the frozen L89E checkpoint.
The existing R0 visual entity cache is used through its finalized manifest;
its underlying data remains at
`/data2/usr_for_deadline/locatemot_r0a_visual_tokens_retry1` and is not
copied. Existing dense indexes and safe target artifacts are read-only.
No screening or official-test labels are read.

R1 builds a compact aligned cache only for an exact prebuilt request manifest:
the V1/V2 requested fit query-frame pairs (six epochs, 3000 frame tiles per
epoch, deterministic quota 3 multi-positive/2 positive/2 inactive/1
present-uncovered, maximum Q=8) and all legal V1/V2 dev query-frame pairs.
The cache stores only query-conditioned FP16 `Z0`, `Z1`, `Z4`, text/frame
summary vectors, and row provenance. It contains no labels, target IDs,
candidate scores, GT boxes, or pixels. To stay within the observed `/data1`
space budget, tensor shards may live under a separately documented
`/data2/usr_for_deadline/locatemot_r1_*` path while `outputs/r1/` contains
compact manifests and audits.

Every native row remains in original bank order with the key
`(dataset,video,frame_id,query_id,bank_path,row_offset)`. Duplicate
candidate indices remain legal and are never deleted. Track IDs are used only
to select causal history; source/pool/group/query/state IDs are not model
features. `history_frame_ids` must be <= the current frame. Token/span→region
and static/motion alignment remain `UNALIGNED`.

## R1 model contract

The frozen `FrozenL89EAnchor` instantiates `L89FullRMOT(L89Config(**package[
"model_config"]))`, loads the checkpoint strictly, uses Stage-S temporal
off, zero history, and returns base candidate energy/presence/null plus
`set_static_state` and candidate prior. The anchor never enters the optimizer.

The trainable sidecar uses only the following preregistered modules and
dimensions:

1. `R1StageMixer`: stack `[Q,N,3,256]` from Z0/Z1/Z4; query is normalized
   Z1 plus text_global; one 8-head attention over the three stage tokens and
   an FFN residual.
2. `R1QueryConditionedDetailHook`: two repeated language-cross-attention →
   72-token visual-cross-attention → 256→1024→256 FFN blocks. Text stays a
   masked token sequence; it is not replaced with only a sentence mean.
3. `R1GeometryEncoder`: existing `[N,5,10]` causal geometry projected
   `10→128→256` and one unidirectional GRU; no ID embedding.
4. normalized box projection `4→128→256`.
5. `R1ResidualSetReasoner`: two 8-head TransformerEncoder layers over all
   current candidates, FFN 1024, norm-first.
6. zero-initialized candidate `delta_energy` and optional zero-initialized
   `delta_presence` heads. The final null is exactly the frozen base null;
   R1 has no new NULL head.

Final outputs are `base_candidate_energy + delta_energy`,
`base_presence + delta_presence`, and `base_null`. No old score, threshold,
teacher, source/pool/group/query ID, or tracker state is a neural input.

## Target-bag loss

All loss is calculated in the frozen Rule-B decision space. Each referred GT
ID defines one positive bag over its candidate rows; each non-referred GT bag
is negative and a background row is a singleton negative bag. Covered active
queries use positive-bag softplus, negative-bag softplus, a weakest-positive
vs hardest-negative margin of `0.50`, and `0.25` duplicate suppression loss
for non-winning rows inside a positive bag. The old row-positive term that
pushes every same-GT duplicate above the emission boundary is explicitly
removed. Every distinct referred target bag still receives a positive loss.

Inactive and present-uncovered units treat current candidates as wrong for
the candidate emission because no valid current candidate can emit the
target; both use all-negative candidate loss. Present-uncovered is retained
as its own category and is never relabelled inactive. A presence residual is
trainable only if its frozen-anchor legal-dev presence-only miss fraction is
at least `0.05`; this decision is written before fitting and never changed.
Residual regularization is `0.01*(mean(delta_energy^2) +
0.25*mean(delta_presence^2))`. No new null loss, temporal loss, or
cross-expression comparison is introduced.

## Training and selection schedule

After compile, boundary, cache, anchor, duplicate, forward/backward and
one <=4-GPU DDP wiring smoke, testing stops and formal training starts. V1
and V2 use separate heads with identical architecture and hyperparameters;
there is no dataset embedding or unified head. Each uses the exact request
manifest for six epochs × 3000 global native-frame tiles/epoch, Q<=8. The
registered optimizer is AdamW `lr=5e-5`, weight decay `1e-2`, betas
`(0.9,0.999)`, gradient clip `1.0`, 5% warmup then cosine; BF16 is allowed
only after the FP32 contract is finite. DDP is at most four processes with
the R0B accumulation map (1→16, 2→8, 3→5, 4→4). Before formal training,
the process table and GPU memory are checked once; only genuinely idle GPUs
are selected, respecting the user instruction to prefer cards with no
processes.

Checkpoints are adapter-only and store config, optimizer/scheduler, input
hashes and anchor/rule provenance without copying frozen L89E weights.
Epochs 1 preview, 2, 4 and 6 are saved; epochs 2/4/6 are eligible for
selection. Preview at epoch2 is diagnostic only; each run continues to epoch6
unless a technical contract fails.

Selection uses only legal dev true-full-video TrackEval under frozen Rule-B
and the preregistered tuple `(HOTA, DetA, AssA, distinct_target_recall,
-inactive_false_acceptance, -epoch)`. It occurs separately for V1 and V2
before any fixed/internal labels. After selection, the selected checkpoint is
frozen, internal aligned features are built label-free, and fixed semantic
diagnostics and internal TrackEval are run. No screening or official-test
labels are in R1.

## Required gates and stop conditions

Anchor reproduction, key/row preservation, finite tensors, and no forbidden
source changes are hard pre-training gates. A formal run requires finite
loss/gradients, strict reload, complete candidate sets, and frozen anchor.
The final fixed semantic gate is diagnostic and requires complete rows,
candidate/target-bag recall and precision, hard-negative and multi-target
metrics, and no candidate deletion/truncation. It is not a HOTA gate.

R1 is `R1_STRONG`, `R1_BREAKTHROUGH`, `R1_PARTIAL`, `R1_NO_BREAKTHROUGH`,
`R1_PROTOCOL_INVALID`, or `R1_ENVIRONMENT_BLOCKED` exactly as defined by the
attachment and the final evidence. A clean result below V1 31/V2 24 ends the
compact anchor-sidecar family; it does not authorize threshold/layer/lr
sweeps. Any contract failure preserves a new attempt, repairs only the
first actionable technical cause, and runs one targeted regression.

## Resource and boundary flags

`/data1` was observed with only about 8.3 GB free at planning time. No new
multi-GB data cache or model copy is permitted. Large score/TrackEval outputs
must use the prior legal `/data2/usr_for_deadline/locatemot_*` convention and
be referenced by compact manifests. No `torch.cuda.empty_cache()` or
per-parameter diagnostics are used inside long loops.

Every R1 machine-readable output carries:

```text
ordinary_mot_ovmot_touched=false
tracker_source_changed=false
uidm_source_changed=false
l69_source_changed=false
screening_gt_used=false
official_test_labels_read=false
```

The final status is `STOPPED_PENDING_SUPERVISOR_REVIEW`; R1 completion is not
the project’s ordinary-RMOT target and no automatic follow-up is launched.
