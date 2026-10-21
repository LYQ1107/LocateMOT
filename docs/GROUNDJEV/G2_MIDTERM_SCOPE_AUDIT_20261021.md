# G2 Clean midterm input scope audit — 2026-10-21

The historical step-3000 directory remains labelled `G2-PL` and was not rewritten. The audit indexed the `583,251`-row Clean TRAIN manifest `f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71` and recorded every one of the 24 midterm input paths with a SHA-256.

- Exact image-path overlap: **4 / 24** frames.
- Independent of the Clean image-path index: **20 / 24** frames.
- Independent generalization claim: **not allowed**; the clips are short functional diagnostics.

## MOT17-02-FRCNN

- Dataset/sequence: `mot17` / `MOT17-02-FRCNN`
- Frames: 4; exact TRAIN overlap: 4
- Same video/sequence appears in G2 Clean TRAIN: `True`
- Overlap paths:
  - `/data2/user/LocateMOT/data/MOT17/MOT17/train/MOT17-02-FRCNN/img1/000302.jpg`
  - `/data2/user/LocateMOT/data/MOT17/MOT17/train/MOT17-02-FRCNN/img1/000303.jpg`
  - `/data2/user/LocateMOT/data/MOT17/MOT17/train/MOT17-02-FRCNN/img1/000304.jpg`
  - `/data2/user/LocateMOT/data/MOT17/MOT17/train/MOT17-02-FRCNN/img1/000305.jpg`

## Refer-KITTI-0007-single

- Dataset/sequence: `refer_kitti_v2` / `0007`
- Frames: 8; exact TRAIN overlap: 0
- Same video/sequence appears in G2 Clean TRAIN: `False`

## Refer-KITTI-0007-multi

- Dataset/sequence: `refer_kitti_v2` / `0007`
- Frames: 12; exact TRAIN overlap: 0
- Same video/sequence appears in G2 Clean TRAIN: `False`

The JSON file is the authoritative per-frame record, including normalized path, existence, SHA-256, and matching G2 source rows.
