# Stage L10 — TAO Train Candidate Generation Audit

Date: 2026-08-17

## 1. Why train-side candidates are generated with DLA (Detic)

The official TAO OVMOT evaluation protocol used by MASA / OVTrack / TETA
feeds the tracker with **public Detic-SwinB detections** on TAO val
(`masa/results/public_dets/tao_val_dets/teta_50_internms/detic_tao_val_det`).
This is confirmed in:

- MASA `docs/model_zoo.md`: "For TAO TETA, we use public detections from
  Detic-SwinB".
- MASA `docs/benchmark_test.md`: public dets are downloaded from
  `huggingface.co/dereksiyuanli/masa/.../public_dets_masa.zip`.
- `tools/build_l7_tao.py` reads exactly that directory and keeps the
  top-50 / score>=0.05 protocol.

For training candidates to be in the same observation distribution as
the evaluation candidates, the L10 train stream therefore uses the same
detector family: **Detic Centernet2 Swin-B FPN 4x LVIS** with the
MASA checkpoint `saved_models/masa_models/detic_masa.pth`
(HF `dereksiyuanli/masa`), invoked through the MASA mmdetection
integration (`projects/Detic_new/configs/
detic_centernet2_swin-b_fpn_4x_lvis-base_in21k-lvis.py`).

Protocol chosen for the L10 train stream (identical to the L7/L9 val
builder's filtering):

- image resize `(480, 288)` keep-ratio
- RCNN `max_per_img` (model test cfg = 100), then keep score >= 0.05
- keep at most top-50 by score
- store `det_bboxes [N,5]` (xyxy + score) and `det_labels [N]`
- write-through per frame as `*.pth`, resume-safe, order preserved

Caveat recorded honestly: the val public dets and the train DLA dets come
from the same Detic-SwinB detector family, but we did not verify
bit-identical parameters between the published val det files and our
`detic_masa.pth` generation.  Both are Detic-SwinB trained on LVIS; the
remaining checkpoint-level difference is small and documented as a
protocol caveat, not silently assumed to be zero.

## 2. torchvision roi_align OOM: reproduced root cause

Environment: `masaenv` conda env, torch `2.1.2.post304`, torchvision
`0.16.2`.

The OOM is **not** caused by the image resolution or by Detic itself.  It
is caused by torchvision 0.16's pure-Python `roi_align` implementation
for `sampling_ratio=0` (adaptive sampling): it materialises the full
6-D interpolation grid `[K, C, PH, PW, H, W]` for every ROI at once.
Reproduced example:

- feature map `H x W = 100 x 136`, `C = 256`, 216 ROIs,
  output `PH x PW = 7 x 7`
- requested tensor = 216 x 256 x 7 x 7 x 100 x 136 x 4 bytes
  ≈ **274 GiB** → instant OOM on a 40 GB GPU.

The Detic pipeline hits this path with its RCNN proposal count because
torchvision 0.16 routes `sampling_ratio=0` to the Python fallback.

## 3. Fix: standard ROIAlign kernel math, drop-in patch

`tools/patch_adaptive_roi_align.py` implements the same adaptive-sampling
math as the standard ROIAlign CUDA kernel (per-ROI sample grid, bilinear
interpolation at `(i+0.5)*bin_h/grid` positions), with per-ROI loops that
keep memory at `O(K*C*PH*PW*grid_h*grid_w)` instead of
`O(K*C*PH*PW*H*W)`.

Verification already completed (Stage L10 handoff, reproducible):

- small-tensor equivalence vs the original torchvision path:
  max abs diff ≈ 1e-6, for both `aligned=True` and `aligned=False`;
- real Detic single-image inference: the previously OOM-ing image now
  completes with 300 dets, peak VRAM ≈ 1.8 GB;
- candidate semantics unchanged: same detector, same scores/boxes, only
  the ROIAlign implementation is replaced.

The patch is imported at the top of the DLA worker
(`tools/generate_l9_tao_train_dets_subset.py`), so it is active in the
L10 full-train generation run.

## 4. DLA generation status

- Command:
  `tools/generate_l9_tao_train_dets_subset.py --gpus 0,1,2,3
  --video-names outputs/l10/data/tao_train_all_videos.json
  --out outputs/l10/cache/tao_train_candidates`
- Scope: all 500 TAO train videos, 18,274 frames (TAO train.json images).
- Output layout: `train/<dataset>/<video>/frameNNNN.pth`
  (`det_bboxes [N,5]`, `det_labels [N]`, N<=50, score>=0.05).
- **COMPLETE** (2026-08-17): 18,274 / 18,274 frames written; workers
  reported `done 4568/4569/4568/4569 frames`, no failures.  Total cache
  ~44 MB (per-frame pkls are small; ~4 KB per frame at 50 dets).

## 5. C-TAO: continuous annotations for the same 500 videos

The COVTrack official repository
(`LocateMOT_reference_repos/covtrack`, commit
`9b0ced5779ee36f5dd73dbe39b5ae5d57abb4b3b`, Apache-2.0) publishes
C-TAO, a continuous annotation of the TAO train videos.  The local
annotation file (already present on this server, from the COVTrack
HuggingFace release):

- path: `/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/data/
  external_annotations/covtrack/ctao_base.json`
- size: 546,599,277 bytes; md5 prefix `c6ff067bf38f606f`;
  sha256 prefix `4269b1a5f026350c`
- content: 500 videos, 490,210 frames, 1,489,637 boxes, 2,588 tracks,
  1,203 LVIS categories (`ctao_base.json`, base categories)
- annotation density ≈ 26x the sparse TAO train GT, matching the paper
  claim ("first continuous annotated training dataset for OVMOT",
  ICCV 2025).

L10 uses C-TAO as the **identity-supervision GT** for the DLA candidates
(C-TAO track ids are continuous and consistent per video), while the
evaluation remains on the official TAO val protocol (unchanged).  For
17,131 of the 18,274 TAO train frames C-TAO provides the same boxes as
TAO train.json (verified on samples); the remaining 1,143 frames use the
original TAO train GT as fallback.

## 6. Why DLA + C-TAO instead of the L9 LocateAnything stream

The L9 OVMOT training stream used LocateAnything-generated boxes on 105
videos (4,200 frames, 7,522 candidates).  Its candidate distribution is
very different from the Detic public dets used at evaluation
(~1.8 candidates/frame vs ~44/frame), which the L9 report identified as
the main reason full-PBD adaptation only recovered AssocA 29.34.

L10 replaces the candidate source with Detic-SwinB dets on **all 500
videos / 18,274 frames**, and aligns candidates to C-TAO continuous
identities.  This makes the training observation stream much closer to
the TAO-val full-PBD protocol while keeping the same cond_gated UIDM
architecture.

## 7. Training-stream hard-negative cap (compute-driven)

The full DLA candidate stream is 905,400 candidates (49.5/frame).
LocateAnything-3B crop-PBD costs ~0.6-0.8 s/crop on this 4-GPU server
(~40k crops/h aggregate), so caching all candidates would take ~23 h.
Stage L10 therefore caps the **training** stream at:

- all GT-matched candidates (31,277, 99.8% of positives), plus
- top-16 unmatched candidates by detection score per frame;
- final stream: 322,843 candidates, 17.7/frame (43x L9's 7,522).

The cap is applied only to training candidates; TAO-val evaluation keeps
the official full candidate protocol.  This is a compute/IO decision
recorded transparently, not a protocol change for the benchmark.
