# R0A frame-specific dense target contract — execution plan

## Identity and authorized scope

- Luna thread: `01a02014-fce8-7f51-8414-e7ed6ab44745`
- Asset root: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`
- Worktree: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_R0A`
- Branch: `codex/r0a-frame-specific-target-contract-repair-20260909`
- Base: `19fcb5fd3dcf1d26adb60a5fcceb3121c6f58c5b`
- R0 historical stop: `R0_DATA_CONTRACT_INVALID`; it remains unchanged.

R0A corrects only the invalid assumption that target IDs are constant over a
video/query. The authoritative semantics are a stable sentence for
`(dataset, video, query_id)` and a frame-specific target map for
`(dataset, video, query_id, frame_id)`. No target union, nearest-frame lookup,
forward/backward fill, pseudo-label, or candidate-derived identity is allowed.

## Source and leakage checks

The local source audit must record the actual read-only
`locatemot/rmot/l49_data.py` path, SHA256, `load_l49_queries` signature,
annotation paths, and schema before code uses it. The L49 helper loads only its
train-pool fit/calibration/validation videos; R0A still filters every optimizer
record through `outputs/l82/protocol/fit_video_train_dev_split.json`. Only L82
train `(dataset, video)` pairs may enter V1/V2 dense training indexes. Dev,
internal, calibration, validation, screening, and official-test labels cannot
enter optimizer data.

## Repair and gates

1. Replace target-bearing `R0TrainQuery` with sentence-only query identity.
2. Add authoritative frame target lookup and explicit frame supervision,
   including distinct covered-target and partial-coverage accounting.
3. Audit all 5,314 sparse L49 fit rows against the authoritative frame map.
   Required result is zero sentence mismatch, target mismatch, missing query,
   and invalid frame. Any nonzero result stops R0A.
4. Build separate V1/V2 native-frame dense indexes only after the audit passes;
   verify train/dev disjointness, native L69 frame pointers, sidecars, and
   candidate rows.
5. Apply the one registered visual-region embedding correction and pure
   language masked-mean helper. Complete only the R0 tools allowed by the
   boundary list.
6. Compile, run the boundary guard, perform one real prompt-invariance and
   forward/backward contract smoke, then freeze code before formal cache and
   training.
7. If all contracts pass, build legal query-independent visual cache, train
   separate R0-V1/R0-V2 heads for the fixed 12-epoch schedule, select only on
   legal dev TrackEval, then run post-selection internal diagnostics and
   internal full-video TrackEval. No screening or official test is part of R0.

## Fixed scientific and resource boundaries

The R0 architecture, loss, seed `20260909`, native L69 candidate rows,
GroundingDINO runtime, pure language cache, zero-logit deployment rule, and
V1/V2 benchmark-specific heads remain unchanged. Only the target contract is
repaired. Shared MOT/OVMOT/TAO, UIDM, tracker, L69 bank, TrackEval source and
production entrypoints remain read-only. `/data2` cache is query-independent,
FP16 token-only, and created only after all R0A gates pass. No raw/dense debug
cache or full-model checkpoint is allowed.

The final R0A classification is not assigned until the repaired contract,
formal training, dev selection, and internal TrackEval complete. If any R0A
contract invariant fails, the stage stops with
`R0A_DATA_CONTRACT_STILL_INVALID` and preserves the first failure.

## First execution finding

The required source audit found that V2 is not safely isolated by the local
`load_l49_queries` implementation: it parses the complete monolithic
`outputs/l11/data/rmot_kitti/expressions.json` before filtering video keys, and
that file contains official-eval videos `0005`, `0011`, and `0013`. The R0A
contract explicitly requires stopping when a loader necessarily reads a
monolithic source containing forbidden labels. This attempt is therefore
`R0A_BLOCKED_L49_QUERY_SOURCE`; no 5,314-row source-consistency audit, dense
index, visual cache, training, dev selection, or TrackEval is valid under this
attempt. A separate supervisor-approved train-scope source/isolation repair is
required before resuming.

## R0A safe-source continuation (current authorized attempt)

The supervisor-approved continuation keeps the historical stop above intact.
The current worktree remains the isolated `LocateMOT_R0A` checkout at commit
`54541f1d99d4f1d8836354f26d554577ffd84a3d`; no raw V2 loader call is allowed.
The new `r0_safe_target_source.py` scans the monolithic V2 files lexically and
deserializes only allowlisted video values. A synthetic CPU fixture must pass
before any real annotation payload is accessed. The valid retry must retain
`forbidden_payload_deserialized=false` and `official_test_labels_read=false`.

After that source contract, the fixed order is: build label-bearing safe
fit/train/dev/internal artifacts; audit exactly 5,314 fit rows against their
frame-specific authoritative records with four zero-mismatch counters; build
separate dense native-frame V1/V2 indexes using only L82 train pairs; fix the
R0-only `(width,height)` versus `(height,width)` geometry property and cache
metadata contract; compile/boundary/real forward-backward smoke; then, and
only then, proceed to the registered visual-cache, benchmark-specific R0
training, legal dev TrackEval selection, and internal full-video TrackEval.
Any source or data-contract mismatch stops the attempt before model work.

## R0A real smoke result

The R0-only implementation gate passed in
`outputs/r0/audit/r0_contract_smoke_retry2/contract.json`: one legal V1 fit
frame completed native visual-token construction, frame-specific supervision,
forward/backward, and strict reload.  It retained 57 rows including 18
duplicate candidate indices; visual tokens were `[57,36,256]`, pure language
tokens were `[1,256,256]`, geometry was `[57,5,10]`, loss was finite, and all
108 trainable gradients were finite and nonzero.  The optimizer contained no
GroundingDINO or tracker parameters, strict reload max difference was zero,
and the temporary visual item was removed.  This is an implementation gate,
not semantic/RMOT/HOTA evidence.  The next action is to commit/freeze this
code and build the legal query-independent cache under `/data2`.
