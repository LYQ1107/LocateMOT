# TempRMOT legal-development baseline

TempRMOT was executed with its published Refer-KITTI checkpoint on two videos from the frozen LocateMOT-U V1 validation split. The run used the isolated source copy at `/data2/user/reference_envs/temprmot/source`, the adapter `tools/unified/baselines/run_temprmot_legal.py`, and no LocateMOT-U training.

| Item | Value |
|---|---|
| Source commit | `6a65640d849fdee4a32bb055945ee34c3b0edeb1` |
| Checkpoint | `checkpoint_rk.pth` |
| Checkpoint SHA-256 | `55e9b2cecbc1590aface4c9d8639008e694e8c4556d1d3c27bf90549f408c7e8` |
| Dataset | Refer-KITTI V1 legal validation subset |
| Videos | `0012`, `0014` |
| Expressions | `cars-in-front-of-the-camera`, `black-cars-in-the-left` |
| Predictions / GT rows | `94 / 93` |
| Official-test labels | not read |
| Screening ground truth | not used |

The upstream model and its TrackEval fork were kept in an isolated environment. Small compatibility changes were confined to that copy: the local torchtext shim, the current `torch.load` argument, disabling an unrelated pretrained-backbone download, the CUDA extension scalar-type API, and NumPy aliases in TrackEval. The checkpoint and raw predictions remain outside this repository. These changes are recorded in the machine-readable summary at `outputs/unified/external_baselines/temprmot_legal_v1.json`.

TrackEval was run on the two generated sequences with HOTA, CLEAR, and Identity. Combined values were HOTA `94.679`, MOTA `98.925`, and IDF1 `99.465`; the full table and command are preserved in `/data2/user/reference_envs/temprmot/trackeval.log`. This bounded result establishes a real external prediction/evaluation path, but it is not the formal LocateMOT-U generic or expression gate and cannot satisfy the two-method external-baseline requirement by itself.
