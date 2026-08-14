# Stage L8 — Dataset Inventory

Date: 2026-08-14

## 1. Refer-Dance (primary RMOT dataset)

- Official source: contributed by iKUN (CVPR 2024), downloaded via the
  iKUN Baidu disk link; zip archived locally at:
  `/data3/testdata/vranlee/.MOTSynth.partial/Refer-Dance/1665795909_薛定谔的小旺财/Refer-Dance.zip`
- Project-local read-only mirror:
  `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/data/refer_dance/`
  - `expression/` (2051 files)
  - `gt_template/` (425 query dirs, 40 non-empty GT)
  - `labels.json`, `seqmap.txt`
  - `DanceTrack/training/image_02/` → symlinks to
    `/data1/LWR/vranlee/DATASETS/JDE/dancetrack/{train,val}/<seq>`
    (65 sequences; no new image copy, no modification of the original)
- Structure verified against the iKUN paper and the official RMOT repo.
- The zip is not modified; the local mirror is a fresh extraction.

## 2. Refer-KITTI / Refer-KITTI-V2 (blocked, not blocking)

- Refer-KITTI-V2 annotations exist at
  `/data1/LWR/vranlee/MFT2025/REFER-MFT25/refer-kitti-v2`
  (`expression/` + `labels_with_ids/`), but the corresponding KITTI
  tracking images are not present locally in a matching form (MFT25
  sequences do not match Refer-KITTI frame numbering).
- No legal auto-download available without account/interactive steps.
- Decision: proceed with Refer-Dance; Refer-KITTI is recorded as a blocker
  only for multi-dataset RMOT, not for the L8 main line.

## 3. Existing LocateMOT caches reused by L8

| Cache | Path | Use |
|---|---|---|
| DanceTrack val candidate manifest (LocateAnything-3B, person) | `outputs/l1_c/fixed_candidate_manifest/dancetrack_val.jsonl` | RMOT eval candidates; PBD features via `tools/eval_l3.py build_candidates` |
| DanceTrack train candidate manifest | `outputs/l1_c/fixed_candidate_manifest/dancetrack_train.jsonl` | RMOT train alignment (1280×720) |
| L6 PBD train cache | `outputs/l6/data/dancetrack_train/*.pkl` | PBD tokens (2048-d) for Refer-Dance train |
| L7 CLIP crop cache (train) | `outputs/l7/data/clip_closed/dancetrack_train/*.pkl` | CLIP tokens (512-d) for Refer-Dance train |
| L7 CLIP crop cache (val) | `outputs/l7/data/clip_eval/dancetrack_val/*.pkl` | CLIP tokens (512-d) for RMOT eval |
| L7 closed-set CLIP caches (ordinary MOT) | `outputs/l7/data/clip_closed/{bdd100k_train,dancetrack_calibration,dancetrack_train,mot17_train,mot20_train}` | joint unified training |
| L6 PBD caches (ordinary MOT) | `outputs/l6/data/{bdd100k_train,dancetrack_calibration,dancetrack_train,mot17_train,mot20_train}` | identity-only ablation |
| TAO val OVMOT cache | `outputs/l7/data/tao_val` | OVMOT official evaluation |

## 4. Disk / space notes

- `/data1` free ≈ 158G; `/data3` free ≈ 122G.
- Refer-Dance mirror after removing the bundled mp4 videos ≈ 177M
  (metadata + symlinks); no large duplicate copy is created.
- New L8 caches (RMOT merged features, checkpoints) are small (feature
  arrays are float16/float32, 62 train expressions) and stay in
  `outputs/l8/`.

