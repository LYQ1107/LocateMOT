# GroundJEV-MOT execution redirect audit — 2026-10-19

This audit is rooted at `/data2/user/LocateMOT`. Runtime artifacts live in the
registered GroundJEV and OneJev checkouts; the paths below are the authoritative
receipts, not inferred process names.

## Verified terminal work

- The protected MOT20-05 G0 export is complete at
  `/data2/user/LocateMOT_OneJev_CVPR/outputs/groundjev/full_data_v2/candidate_export_v2/mot20_MOT20-05_resumed_v2/`.
  Its receipt is `VIDEO_INFERENCE_COMPLETE_WITH_REJECTED_PERCEPTION`, covers all
  1,658 legal TRAIN frames, resumes an exact 1,560-frame prefix, retains 224
  rejected/strict-invalid perception frames as empty prediction frames, and
  records `gt_read_by_runtime=false` and `future_frames_read_by_runtime=false`.
  The native candidate archive is 2,314,870,593 bytes. The evaluation-only
  TrackEval receipt reports HOTA 0.433441, DetA 0.638348, AssA 0.294655, and
  IDF1 0.441085; this is a G0 TRAIN-scope reference, not a held-out claim.
- The expression-conditioned RMOT pilot is terminal with all 24 legal query
  manifests complete and `failed=null`. It uses `native_G0_expression_conditioned`
  proposals and excludes held-out/evaluation videos.
- The prior G1-200 projector adaptation and its G0/G1 controls remain preserved.
  The measured short controls are mixed/mostly negative, so that result is kept
  as an ablation and is not promoted as a detector improvement.

## Active work and safe gates

- No process matching the registered MOT-only launch names is alive. The U2
  state is `STOPPED_BY_USER_RESEARCH_PIVOT`; its code/checkpoint artifacts are
  retained and no optimizer restart is allowed.
- The real OneJev J1 course reads 43,533 rows: identity 21,168, RMOT 16,677,
  and OVMOT 5,688. Its distributed multimodal run is alive on GPU4/GPU6 and
  has a durable FP32/Adam/RNG checkpoint at step 700 (`adapter-step700`). No
  final receipt or held-out score exists yet.
- Clean G2 is still running on GPU3/GPU5 from the exact clean manifest. The
  latest observed event is optimizer step 131 of 18,227; its terminal receipt
  is absent. The G1 GPU5 queue is alive but explicitly `WAITING_G2_CLEAN_SUCCESS`,
  and the bundle exporter waits for both G2 and G1 terminal receipts. No new
  G1 process is started while G2 owns GPU5.
- The J1 held-out gate remains `WAITING_J1_FINAL`. It will only consume the
  exact step-1200 checkpoint and registered held-out manifests after J1 writes
  its terminal receipt.

## Recovery order

1. Let clean G2 reach a valid terminal receipt without changing its manifest,
   seed, world size, or checkpoint protocol.
2. Allow the existing GPU5 G1 queue to launch its registered fresh adaptation
   only after that receipt; retain the prior G1-200 result as a separate control.
3. Let J1 reach step 1200, then run the fixed-scope held-out gate and preserve
   rejected-frame and no-runtime-GT receipts.
4. Export the clean G2/G1 bundle and publish the independent MOT/RMOT/OVMOT
   held-out results. Until these receipts exist, no unified quality improvement
   is claimed.

