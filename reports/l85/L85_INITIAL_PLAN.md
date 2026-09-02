# L85 initial plan — full RMOT training and full-video HOTA

## Identity and frozen boundary

- Project root: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`
- Luna thread: `01a02014-fce8-7f51-8414-e7ed6ab44745`
- Branch: `codex/l85-full-rmot-hota-20260902`
- Starting commit: `c65af026c02fbe7fd24e72a315963d89373dcd4c`
- Fixed manifest: `outputs/l19/protocol/kitti_fast_eval_manifest.json`, expected
  SHA256 `06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`.

L11/L16/L17/L18/L19/L26/L29/L42/L59/L62/L64–L84 assets, the shared UIDM
checkpoint, TrackEval sources, labels and ordinary MOT/OVMOT entrypoints are
read-only. This stage adds only an RMOT sidecar and evaluation artifacts under
`locatemot/models/l85_*`, `locatemot/rmot/l85_*`, `tools/l85_*`, `outputs/l85/`
and `reports/l85/`. The sidecar must be disabled without changing ordinary
MOT/OVMOT output.

## Evidence carried forward

L84 passed the paired mid-decoder diagnostic stability gate for Z1/R1. The
registered tie-break selected the earlier fixed-reference Z1 state; the
corrected no-refPE test did not replace it. Z1 is therefore the frozen semantic
representation for this stage (`[Q,N,256]`). L84 was not a deployable semantic
gate and produced no HOTA/TrackEval result. No L85 training result is assumed
before the contracts below pass.

## Inputs and split policy

The query-independent candidate source is the L69 budget-40 feature/dual-bank
view at `outputs/l69/attempt9/`; its V2 files resolve to the audited L69 attempt4
files and are not copied. Native `frame_ptr`/`frame_ids` construct every current
candidate set. The observation contract is
`clip[512]+history_clip[512]+uidm_h[384]+geometry[7]+motion[8]+lifecycle[8]+objectness[1]`
(`1432` dimensions), with causal history length 8. L48 text is used only as a
word-token input to the compact adapter. L84 Z1 states are generated from the
verified GroundingDINO runtime in a process-local or compact, measured semantic
state view; raw images and detector weights are never copied.

Training uses only the 5,314 L49 V1/V2 `split=fit` units and expression-level
membership labels attached after complete feature construction. Refer-KITTI and
Refer-KITTI-V2 are trained independently. Candidate coverage and target-bag
metrics are kept separate from HOTA. Public/local internal validation videos
are `refer_kitti_v1: 0004,0018` and `refer_kitti_v2: 0016,0017,0020`; hidden or
official-eval videos (`0005,0011,0013,0019`) are not read in this stage.

## Fixed model and curriculum

The factorized model is `S=A_i+B_q+R_total(i,q)` with Z1 static semantic
interaction, causal observation history, query-independent candidate prior
`A`, query/frame presence `B`, semantic rank losses on `R_total` and
`R_static`, membership loss on `S`, shared-energy `null=-B+bias`, and a small
temporal auxiliary. No IDs, old scores, top-k, NMS, or candidate deletion are
inputs or operations. The pre-registered curriculum is S=8 epochs single
frame, T=12 epochs clip length 4, J=20 epochs clip length 4, total 40 epochs.
Z1 permits LoRA only on the last two fusion encoder blocks plus decoder layer 1;
base weights stay frozen and adapter checkpoints never contain full detector
weights.

The four-GPU training contract is `CUDA_VISIBLE_DEVICES=0,1,2,3`, world size 4,
BF16 after finite checks, activation checkpointing/gradient accumulation as
needed, and complete candidate sets. Query tiles are chosen once by a label-free
memory audit from `{8,16,24,32}` using the largest legal candidate set and a
four-frame clip. Checkpoints are atomic and include epoch, RNG, sampler,
protocol/source hashes and the isolated adapter/LoRA state.

## Evaluation and stopping gates

Before any label-based selection: audit bank completeness, candidate oracle
coverage and TrackEval format; run a label-free memory contract. Checkpoints
and thresholds are selected only on video-disjoint internal dev data using the
fixed target-bag tuple and registered global threshold grid. Final legal
validation/evaluation runs are frozen before reading their labels. Every full
video result is explicitly labelled `full-video validation HOTA` unless a
verified public benchmark split is used. Required metrics include HOTA, DetA,
AssA, LocA, DetRe/DetPr, AssRe/AssPr and supported MOTA/IDF1/IDs/FP/FN.

If the TrackEval contract is valid, L85 must run at least one genuine full-video
HOTA evaluation even if surrogate metrics are weak. If data, checkpoint,
runtime, or TrackEval contracts fail, preserve the first traceback and mark the
stage `*_contract_fail`/`trackeval_protocol_blocked`; do not fabricate HOTA.
Regardless of score, stop after the registered 40-epoch per-dataset training
and legal full-video evaluation, with no automatic extra epochs or threshold
repair. No screening/hidden official-test labels are read.

## Required flags

`candidate_bank_gt_conditioned=false`, `candidate_bank_query_conditioned=false`,
`screening_gt_used=false`, `official_test_labels_read=false`,
`ordinary_mot_ovmot_touched=false`. Token/span-to-region and static/motion
alignment remain `UNALIGNED`. Oracle, fit, calibration, dev selection,
validation HOTA and TrackEval evidence are reported in separate files.
