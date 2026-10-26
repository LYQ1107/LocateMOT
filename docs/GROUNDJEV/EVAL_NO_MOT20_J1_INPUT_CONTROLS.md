# J1 current-crop and memory-input controls

Status: registered before new GPU inference; no tracking improvement established.

The canonical project is `/data2/user/LocateMOT_GroundJEV`; this code is in its
authorized independent evaluation worktree. The main G2 native queue, frozen
scopes, official parser, thresholds, seed and checkpoints remain unchanged.

The actual J1 TRAIN row `MOT17-04-FRCNN_000001_c7_identity_birth` contains two
images: the current frame and a current-candidate crop sheet. The legacy native
adapter sends only the frame. Rendering its 44 xyxy candidate crops with the
TRAIN builder's layout reproduces the saved 640x2112 TRAIN PNG pixel for pixel.
This establishes an input-contract omission, not a measured performance cause.
No historical crop image is added: that would be another intervention.

The compact memory request also renumbers selected candidates locally while
keeping original indices in `assigned_identity_per_candidate`; the question
and assignment can therefore describe different candidates. Its competition
matrix still treats xyxy as xywh. A regression with assignments at original
indices 1 and 3 checks local question/state keys 0 and 1, separate physical
objects, and remapping of returned probabilities to original candidates.

Two controls use the already frozen complete SportsMOT video
`sportsmot__v_9MHDmAMxO5I_c003__d743f9a1134bb802` (226 frames):

| Arm | Current candidate image | Memory transport |
|---|---|---|
| `j1_competition_xyxy_v2` | full frame | legacy |
| `j1_current_crops_v3` | full frame + TRAIN-layout crops | legacy |
| `j1_current_crops_memory_v4` | full frame + TRAIN-layout crops | local indices + converted geometry |

The baseline and Simple candidate SHA is
`46fce75613893b8642a0b0be118b0462e2e89996e38fba2feec7b711159e34b6`.
The merged J1 step1200 weight SHA is
`5cc64d1bcae70db4a4d8596c655bc38ce0e8c1d63cb3bd2bf14d30e9848c07a5`.
No forced NEW, probability adjustment, VAL-selected threshold, detector rerun
or new training is permitted in these controls. Legacy and v2 files are retained.

Run both complete-video controls sequentially on a registered physical GPU:

```bash
/data2/user/LocateMOT_GroundJEV/.venv-onejev-py310/bin/python \
  tools/groundjev/eval_no_mot20/run_j1_input_controls.py \
  --job /data3/user/LocateMOT-runs/GroundJEV_Eval_NoMOT20_4GPU/request_local_seed_v2/jobs/sportsmot__v_9MHDmAMxO5I_c003__d743f9a1134bb802 \
  --gpu 3
```

The owner validates the GPU UUID/free memory, restricts its task-local CUDA
visibility, writes the registration before inference, waits for each actual
process, and records official metrics and failures. It may share an existing
GPU slot; it never adds a fifth physical GPU or an SGLang service. Sharing makes
its wall time unsuitable for a controlled speed comparison. Receipts are under
`request_local_seed_v2/engineering_checks/j1_input_contract/`.

The committed main arm remains v2 until these controls are validated. Merely
passing the request test or matching TRAIN crop pixels does not demonstrate
better tracking, and the complete evaluation goal remains unfinished.
