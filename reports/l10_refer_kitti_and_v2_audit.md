# Stage L10 — Refer-KITTI / Refer-KITTI-V2 Audit

Date: 2026-08-17

## 1. Refer-KITTI (v1)

- Source task/paper: "Referring Multi-Object Tracking" (RMOT),
  Dongming Wu et al., CVPR 2023; official repo
  https://github.com/wudongming97/RMOT.
- Data: built on the KITTI tracking benchmark training sequences
  (21 sequences, `training/image_02/0000..0020`), with referring
  expressions and per-frame target track ids.
- Official evaluation: TrackEval with KITTI-format RMOT evaluator
  (HOTA / DetA / AssA / MOTA / IDF1), using the official sequence map.
- Protocol note: RMOT/Refer-KITTI trains and evaluates on the same
  21 KITTI sequences; there is no separate held-out test set in the
  official protocol.

## 2. Refer-KITTI-V2

- Source: "Bootstrapping Referring Multi-Object Tracking" (TempRMOT),
  arXiv:2406.05039; official GitHub https://github.com/zyn213/TempRMOT
  (cloned at `LocateMOT_reference_repos/temp_rmot`, commit
  `6a65640d849fdee4a32bb055945ee34c3b0edeb1`; no LICENSE detected, no
  code copied).
- Relation to v1: V2 is an **expansion of Refer-KITTI on the same 21
  KITTI tracking sequences**.  It starts with 2,719 manual annotations
  (addressing class imbalance, adding keywords) and expands them with
  GPT-3.5 to **9,758 annotations / 617 distinct words**.  The `label`
  field in each expression JSON maps frame ids to KITTI track ids; the
  `raw_sentence` field records the original manual annotation and
  `sentence` the expanded version.
- Official split (verified from the release files): `refer-kitti-v2.train`
  contains **17 training sequences** (5,171 frames: 0000,0001,0002,0003,
  0004,0006,0007,0008,0009,0010,0012,0014,0015,0016,0017,0018,0020) and
  `datasets/data_path/seqmap.txt` evaluates **4 held-out sequences**
  (0005, 0011, 0013, 0019; 861/862 expression entries).  There is
  therefore **no train/eval video overlap** in the official protocol;
  L10 follows it exactly (cross-domain evidence additionally comes from
  training RMOT only on Refer-Dance and evaluating on Refer-KITTI-V2).

## 3. Local data status

Present (verified, read-only):

- Expressions: `/data1/LWR/vranlee/MFT2025/REFER-MFT25/
  refer-kitti-v2/expression/<seq>/<expr>.json` for sequences 0000-0020
  (21 dirs; per-expression `label`, `ignore`, `video_name: KITTI_<n>`,
  `sentence`, `raw_sentence`).
- Labels: `/data1/LWR/vranlee/MFT2025/REFER-MFT25/refer-kitti-v2/
  labels_with_ids/image_02/<seq>/<frame>.txt`
  (6-column `class_id track_id x1 y1 w h`, KITTI-normalized, per the
  TempRMOT loader).
- Official train list and evaluator seqmap: `temp_rmot/datasets/data_path/
  refer-kitti-v2.train`, `.../seqmap.txt`, and `temp_rmot/TrackEval`.

Missing (was blocking):

- KITTI tracking images `KITTI/training/image_02/<seq>/<frame>.png`.
  The only local `KITTI-train_test-Image/training/image_2/` directory is
  the KITTI **object-detection** image set (14,999 flat files), not the
  tracking sequence layout required by Refer-KITTI/V2.

## 4. Image acquisition

- Official source: KITTI tracking benchmark
  (https://www.cvlibs.net/datasets/kitti/eval_tracking.php).
- Direct download URL (official S3 mirror):
  `https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_2.zip`
  (15,813,146,295 bytes, HTTP 200, no login required).
- Started 2026-08-17 ~15:08 local time; destination
  `/data1/LWR/vranlee/SERVER_ONLY/avis/KITTI_tracking/
  data_tracking_image_2.zip` (shared dataset location; LocateMOT will
  symlink, never copy the images into the repo).
- After download: `unzip` will extract `training/image_02/*` (the 21
  training sequences; testing images are not needed for Refer-KITTI-V2
  evaluation) and the directory will be recorded with checksums in the
  final report.

## 5. Leakage rules for L10

1. Refer-KITTI-V2 is not an independent dataset from Refer-KITTI: same
   videos, expanded queries.  L10 will therefore **not** present
   Refer-KITTI-v1 and V2 as two independent generalization tests; V2 is
   the larger/updated version of the same benchmark.
2. The stronger cross-domain evidence in L10 remains:
   train RMOT only on Refer-Dance (or Refer-Dance + V2) and evaluate on
   Refer-KITTI-V2 with the official evaluator; this isolates language/
   domain generalization through one shared checkpoint.

## 6. Decision

- Follow the official TempRMOT protocol for Refer-KITTI-V2 (train list +
  861-expression TrackEval seqmap).
- Use the official KITTI tracking images via symlink from the shared
  dataset location.
- Integrate Refer-KITTI-V2 as the second RMOT benchmark; keep
  Refer-Dance as the first.  If the KITTI image download/extract or the
  official evaluator cannot be completed in time, this is recorded as
  `BLOCKED_BY_*` and the TAO main line continues.
