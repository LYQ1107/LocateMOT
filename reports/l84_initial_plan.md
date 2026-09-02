# L84 initial plan — paired mid-decoder correspondence verification

Date: 2026-09-02
Project root: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`
Luna thread: `01a02014-fce8-7f51-8414-e7ed6ab44745`
Branch: `codex/l84-paired-middecoder-20260902`
Starting HEAD: `5fd92b87927d60c831e4ac75774929f07f371d7e`

## Purpose and frozen evidence

L83 completed the corrected duplicate-aware target-bag implementation and a
finite/reloadable fit probe, but its faithful training gate failed.  The
authoritative decoder-sharpness audit is
`outputs/l83/audit/decoder_sharpness_attempt9/decoder_sharpness.json` with 138
video-disjoint development groups.  It found a diagnostic, not a deployable
semantic result: Z4 has aggregate bag-hard `0.717948718`, hit@1
`0.381410256`, V2 bag-hard `0.706806283`, and V2 hit@1 `0.376963351`; the
corrected L59/Z0 comparison is aggregate hard `0.794871795`, hit@1
`0.320512821`, V2 hard `0.774869110`, and V2 hit@1 `0.319371728`.
The L83 faithful gate was
`faithful_target_bag_training_gate_fail`; no screening, official-test labels,
TrackEval, HOTA, ordinary MOT, or OVMOT result exists.

L84 tests only whether that apparent mid-decoder sharpening survives a paired
initialization, paired RNG trajectory, and three fixed seeds under the exact
L83 loss/schedule.  It does not re-litigate or relabel the L83 failure.

## Single hypothesis and representations

If L70/L71-style failure was caused mainly by deep set/emission calibration and
not by a complete absence of frozen correspondence information, a paired
probe should reproduce a stable benefit for an intermediate decoder state.
The only stages are:

* `Z0`: L59 fused-ROI visual seed;
* `Z1`, `Z4`, `Z6`: fixed-reference decoder layers 1, 4, and 6;
* `R1`, `R4`, `R6`: native iterative-refinement decoder states 1, 4, and 6.

R states are semantic diagnostics only.  L69 candidate rows, boxes, row
identity, and counts remain unchanged; refined references never become output
boxes.  No source/pool/group/query/track ID is a model feature, and
`token_span_region_alignment=UNALIGNED`,
`static_motion_alignment=UNALIGNED`.

## Fixed inputs and source-of-truth checks

Before any probe, hash and record in
`outputs/l84/preregister/source_of_truth.json` and
`outputs/l84/preregister/frozen_assets.json`:

* fixed manifest
  `outputs/l19/protocol/kitti_fast_eval_manifest.json`, expected SHA256
  `06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`;
* all 17 L69 budget-40 feature banks and their manifests;
* L48 text cache;
* L81 step-100 and L82/L59/L81 candidate controls;
* GroundingDINO config, detector weight, local BERT snapshot, and UIDM
  step-11000;
* immutable L83 metric/loss code used by the corrected probe.

Any mismatch is `source_of_truth_mismatch` and stops L84.  No frozen file is
rewritten.  The runtime is the already audited local MMDetection/GroundingDINO
path; its local checkout is recorded as unverified if it has no verifiable
HEAD.  All L84 representations are rebuilt or captured in process and no
dense/raw feature cache is persisted.

## Paired protocol

The probe is exactly the L83 faithful probe architecture: LayerNorm(256),
Linear(256,256), GELU, Dropout(0.05), Linear(256,1), 66,561 parameters.  Use
the same 524 train groups, 138 video-disjoint dev groups, 10 epochs, AdamW
`lr=2e-4`, weight decay `1e-4`, 5% warm-up, cosine schedule, gradient clip
`1.0`, and corrected L83 target-bag loss/metrics.  For each of seeds
`20260829`, `20260830`, and `20260831`, construct one CPU canonical initial
state and load that exact state into every representation stage.  Pre-materialize
one deterministic schedule per seed and use paired CPU/CUDA/NumPy RNG seeds
`seed + rank*100003` before each stage.  Initial parameter maximum difference
must be zero.

The only new structural test is permitted after representation selection:
for the selected decoder stage, run the same three seeds with reference
positional encoding removed from the content seed (`content=visual_seed`) but
still used through reference points/query-pos/deformable attention.  No other
depth, loss, schedule, or representation change is allowed.

## Pre-registered stable-mid-layer gate

For a candidate M, all of the following are required:

* A: mean over seeds `Z0_hard - M_hard >= 0.03` and M hard is lower for all
  three seeds;
* B: mean `M_hit1 - Z0_hit1 >= 0.04` and at least two seeds improve hit@1;
* C: mean V2 hard improvement `>=0.03`, no seed worsens V2 hard by more than
  `0.01`, and mean V2 hit@1 improvement `>=0.03`;
* D: mean query-swap accuracy is no more than `0.03` below Z0;
* E: mean multi-target exact is no more than `0.03` below Z0;
* F: the 10,000-resample paired bootstrap 95% CI lower bound for aggregate
  bag-hard improvement is greater than zero.

If multiple stages pass, select lexicographically by lower mean hard, higher
mean hit@1, lower V2 hard, higher V2 hit@1, higher multi exact, then earliest
/ simpler stage.  If none pass, select the stable fallback `Z0_fallback` and
continue to L85 as required by the overarching plan.  A selected decoder
stage receives the single no-refPE test; it is selected only if mean hard
improves by at least `.02`, mean hit@1 by at least `.02`, and V2 hard does not
worsen.  Legal L84 end states are `middecoder_verified`,
`middecoder_not_verified_use_z0_fallback`, `no_refpe_selected`, and
`original_content_seed_selected`; every state has `NEXT_STAGE=L85_FULL_RMOT`.

## L85 authorization boundary

After L84, pure representation probing stops.  L85 must use the selected
representation or Z0 fallback in a new factorized full-RMOT model with
query-independent candidate identity, causal temporal history, target-bag
semantic rank, candidate prior `A_i`, frame/query presence `B_q`, and
`S=A+B+R`. It must audit/materialize a query-independent full-video bank if
needed, run the candidate oracle, audit TrackEval protocol, train Refer-KITTI
and Refer-KITTI-V2 separately for the fixed 40-epoch curriculum, select only
from internal video-disjoint dev, and run full-video TrackEval/HOTA on each
legal validation/evaluation split. It must report HOTA, DetA, AssA, LocA,
DetRe/DetPr, AssRe/AssPr and supported MOTA/IDF1/IDs/FP/FN.

L85 may proceed after valid source/data/checkpoint/full-video/TrackEval
contracts even if L84 surrogate metrics are weak; it may not add a new pure
probe. Hidden official-test labels remain forbidden unless a later authorized
protocol explicitly provides them. Ordinary MOT/OVMOT/TAO, PBD, UIDM,
L11–L83 assets, and their production entrypoints remain untouched.

## Required outputs and flags

L84 outputs are under `outputs/l84/`, with paired protocol, audit, training,
selection, and final-integrity directories. Reports are under `reports/l84/`.
Every machine-readable output records command, inputs/outputs, hashes, status,
candidate retention, finite checks, and the flags:

`screening_gt_used=false`, `official_test_labels_read=false`,
`hota_trackeval_run=false`, `ordinary_mot_ovmot_touched=false`,
`candidate_deletion=false`, `candidate_truncation=false`.

After L84 and L85 evidence is written, append concise entries to
`research_log.md`. Only L84/L85 files and the corresponding log/report entries
will be staged for the GitHub commit; pre-existing unrelated worktree changes
will not be included.
