# LocateMOT-U data provenance — U0

The frozen machine-readable protocol is
`outputs/unified/protocol/data_manifest.json`; the legal split is
`outputs/unified/protocol/legal_video_split.json`.

## Current public inputs

- KITTI tracking training left images: official S3 archive, CRC-validated,
  8,008 extracted PNGs.
- Refer-KITTI V1: author v1.0 expression and `labels_with_ids` releases;
  fit/calibration/validation expression counts 556/13/91.
- Refer-KITTI V2: author Drive expression/corrected-label release; 9,758
  expression JSONs and 7,162 label files in 16,920 unique files.
- GroundingDINO Swin-T and local BERT assets are available for smoke only.
  MM-GroundingDINO-B remains a U1 acquisition requirement.

## Legal boundary

All development loaders must call `locatemot.unified.data.legal_scope` before
opening a video. V1 `0005,0011,0013` and V2 `0005,0011,0013,0019` are reserved
and unopened. The current manifest records `official_test_labels_read=false`.

V2 contains 157 legal expression-frame references across 11 physical
video/frame/target-ID pairs whose expression target ID is absent from the
corrected label file. These rows carry `SOURCE_TARGET_ID_MISSING`. They are
not positive supervision, false-negative supervision, silently dropped
records, or guessed labels; they remain available as unlabeled image/text
records.

## Missing and deferred assets

Historical L69, UIDM, L89E/R1 checkpoints and dense/aligned caches are absent.
Their absence is recorded rather than replaced by a fake reproduction. BDD,
MOT17/20, DanceTrack, TAO/C-TAO, Refer-Dance, STORM-Bench and prompt datasets
are U0/U1 acquisition items and are not silently treated as present.
