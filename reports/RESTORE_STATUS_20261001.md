# LocateMOT restoration — 2026-10-01

## Scope

This restoration follows the latest R1 branch at commit
`99090c8b009ec21278e979099360ebc774f6293e` and the migrated project root
`/data2/user/LocateMOT`. Historical checkpoints and frozen feature banks were
not available on the new server, so no old R1 score or HOTA result is being
claimed as reproduced. The next run must be a new reconstruction from the
assets recorded below.

## Verified assets

- KITTI tracking left images are restored from the official S3 archive. The
  archive is `15,813,146,295` bytes, CRC validation passed, and the training
  extraction contains `8,008` PNGs (`6,522,485,115` uncompressed bytes).
- Refer-KITTI V1 expressions and `labels_with_ids` are restored from the
  author's v1.0 release. Both ZIPs passed CRC validation. The legal fit,
  calibration and validation pools contain 556, 13 and 91 expression files;
  all audited V1 frame and target references are complete.
- Refer-KITTI V2 expressions and corrected labels are restored from the
  author's Drive folder. The download has 16,920 unique local files from
  16,940 manifest records: 9,758 expression JSONs and 7,162 label files. The
  20 duplicate expression paths were byte-identical and are recorded in
  `outputs/restore_20261001/v2_duplicate_audit.json`.
- V2 has a source annotation anomaly, not a download failure: 157 legal
  expression-frame target references (11 physical video/frame/ID pairs,
  repeated across expressions) point to target IDs absent from the
  corresponding corrected label files. Missing frames are zero. The complete
  anomaly list is in `outputs/restore_20261001/latest_public_manifest.json`;
  the compact asset audit is in `outputs/restore_20261001/refer_kitti_audit.json`.
  Any new adapter must surface these references explicitly rather than
  filtering them silently.
- Official evaluation videos remain unopened: V1 `0005,0011,0013`; V2
  `0005,0011,0013,0019`.

## Runtime

- Project-local `.venv` uses system PyTorch `2.10.0+cu128` with CUDA enabled,
  `transformers==4.46.3`, `mmengine==0.10.7`, `mmcv==2.1.0`, `mmdet==3.3.0`,
  and `fairscale==0.4.13`.
- Official MMDetection GroundingDINO Swin-T weights are present and SHA-256
  verified as
  `b448804bb1af6fa688887f0f2454625edbeeae4e868bc95620e3e6413581051a`.
  The local `bert-base-uncased` snapshot is present under `weights/`.
- The model builds with 172,977,693 frozen parameters and no missing
  checkpoint keys. The only unexpected key is the standard
  `position_ids` buffer. A real GPU inference on KITTI video `0008`, frame
  `000000`, sentence `left cars in red` returned 300 finite candidate boxes
  and finite scores; evidence is in
  `outputs/restore_20261001/grounding_inference_smoke.json`.
- Direct dependencies of the latest R1 source were recovered from the
  published source snapshot and migrated to `locatemot.paths`. Import and
  compile checks pass for the R1 modules and their L49/L79/L80 data/runtime
  dependencies, including the R0 safe-source loader for the restored
  per-expression V2 layout. The MMEngine/PyTorch 2.10 checkpoint load uses an explicit
  `weights_only=False` load only for the SHA-verified official checkpoint.

## Still unavailable

The old R1/L89E checkpoints, L69 budget-40 bank, dense train/dev indexes,
language/visual/aligned caches, and the old fixed protocol manifest are not
present. Their paths now fail explicitly under the migrated root instead of
silently resolving to `/data1`. Rebuilding those products would be a new
experiment and cannot be called historical reproduction.

## Next reconstruction step

Keep the latest R1 code isolated and build a new public-data manifest/index
from the restored legal fit/calibration/validation pools. Preserve all native
candidate rows, retain the V2 anomaly as an explicit status, and reserve the
official-evaluation videos. Only after that index and a newly generated
GroundingDINO candidate bank pass finite/coverage checks should a new
correspondence model be trained. The old R1 sidecar will not be extended by
threshold, layer or learning-rate sweeps because its evidence identifies
semantic correspondence plus score/volume collapse as the failure mode.

The public-data manifest is now materialized. A deterministic 40-row frozen
GroundingDINO coverage probe is also complete. On its midpoint-frame replay,
123 target boxes had mean best IoU `0.3188`, IoU@0.25 `0.6585`, IoU@0.50
`0.1463`, and mean best score `0.0107`; the first frame-policy attempt is
preserved separately. This is candidate-input evidence, not an RMOT score.
It supports rebuilding and auditing a new candidate bank before any R1-style
training; it does not justify threshold tuning.

## Provenance files

- `outputs/restore_20261001/kitti_archive_validation.json`
- `outputs/restore_20261001/v1_download_validation.json`
- `outputs/restore_20261001/v2_download_validation.json`
- `outputs/restore_20261001/refer_kitti_audit.json`
- `outputs/restore_20261001/grounding_runtime_build.json`
- `outputs/restore_20261001/grounding_inference_smoke.json`
- `outputs/restore_20261001/environment.json`
- `outputs/restore_20261001/latest_public_manifest.json`
- `outputs/restore_20261001/latest_public_queries.jsonl`
- `outputs/restore_20261001/grounding_coverage_probe_attempt1.json`
- `outputs/restore_20261001/grounding_coverage_probe_attempt2.json`
- `outputs/restore_20261001/recovered_source_provenance.json`
