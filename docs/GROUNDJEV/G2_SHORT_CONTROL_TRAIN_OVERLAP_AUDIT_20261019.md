# G2 short-control / TRAIN overlap audit (2026-10-19)

G2 rows observed: **583251** (manifest declares 583,251); malformed rows: **0**.

Each of the 24 control images was hashed with SHA256. Training rows were read read-only and matched by normalized absolute image path; video/frame indicators were also recorded. Any exact overlap is classified as TRAIN-scope and excluded from an independent-generalization claim.

| Control scope | Frames | Exact TRAIN path/frame matches | Classification |
|---|---:|---:|---|
| MOT (MOT17-02-FRCNN) | 4 | 4 (302,303,304,305) | TRAIN-scope/non-independent diagnostic |
| RMOT-single (Refer-KITTI-0007-single) | 8 | 0 (none) | no exact TRAIN path overlap found |
| RMOT-multi (Refer-KITTI-0007-multi) | 12 | 0 (none) | no exact TRAIN path overlap found |

## Interpretation

The MOT control has same-video TRAIN coverage (602 TRAIN rows for MOT17-02-FRCNN) and exact path matches for all four evaluated frames. The RMOT controls have no exact Refer-KITTI 0007 path matches in the manifest. The audit is an overlap audit, not a performance result. Short controls with exact TRAIN path/frame matches are retained only as diagnostics. Scopes with no exact path overlap still require the larger independent validation requested in P1.

Receipt: `/data2/user/LocateMOT/docs/GROUNDJEV/G2_SHORT_CONTROL_TRAIN_OVERLAP_AUDIT_20261019.json`
