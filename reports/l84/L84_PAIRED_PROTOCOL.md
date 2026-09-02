# L84 paired mid-decoder protocol

Project root: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`
Luna thread: `01a02014-fce8-7f51-8414-e7ed6ab44745`
Branch: `codex/l84-paired-middecoder-20260902`

L84 was the final representation-only verification before L85.  It used the
immutable L69 rows and the local audited GroundingDINO runtime.  The only
registered states were Z0 (fused ROI seed), Z1/Z4/Z6 (fixed-reference decoder
layers), and R1/R4/R6 (native iterative-refinement diagnostics).  R states did
not replace boxes or candidate identities.

Every state used the same 66,561-parameter probe, the corrected L83
target-bag loss and metrics, 524 fit groups, 138 video-disjoint dev groups, 10
epochs, AdamW `2e-4`, weight decay `1e-4`, 5% warm-up, cosine schedule and
gradient clipping at 1.0.  Seeds were `20260829`, `20260830`, and `20260831`.
Each stage loaded the same CPU canonical initialization for its seed.  The
global stage schedule was pre-materialized and paired; no validation or
screening label was used.

The source snapshot is
`outputs/l84/preregister/source_of_truth.json` and the full training output is
`outputs/l84/train/paired_middecoder/`.  A four-group GPU0 contract audit is
at `outputs/l84/audit/forward_contract_attempt2/`; attempt1 is retained as a
script-only status aggregation failure.  The corrected selection audit and
the only corrected no-refPE run are at
`outputs/l84/train/selection_correction_attempt2/`.

All candidates remained in native row order.  `candidate_deletion=false`,
`candidate_truncation=false`, `token_span_region_alignment=UNALIGNED`, and
`static_motion_alignment=UNALIGNED`.
