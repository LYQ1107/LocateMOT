# J1 current-crop and memory-input controls

Status: both226-frame real GPU controls are complete; no effective tracking gain established.

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

After complete real GPU inference and official scoring, the main J1 arm now
uses v4 to repair the verified input-contract errors. This is an engineering
correction, not adoption of a performance improvement. Legacy/v2/v3 remain
separate historical arms. The complete evaluation goal remains unfinished.

The actual completed J1 recipe points to `three_task_sft_v2b/rows.jsonl` and
uses multimodal TRAIN with 43,533 rows. All 43,533 declare image2 and have the
corresponding second media entry. Of 3,471 birth rows, all omit a candidate
index despite multiple candidates. The actual upstream `render_example`
confirms two input groups with conflicting labels (39 rows total):
`MOT17-09-FRCNN` frame16 includes both NEW and REJECT; frame15 includes REJECT
and NO_MATCH under identical state/question/media. Row IDs and supervision
metadata are not part of the rendered prompt. TRAIN option permutation does
not restore a candidate index. This is a concrete data bug, not a quantified
explanation of the checkpoint's tracking deficit; the original rows and
checkpoint remain unchanged. See `EVAL_NO_MOT20_J1_TRAIN_INPUT_AUDIT.json`.

All controls use the same complete226-frame video and2,660 candidate decisions.
Percentages below are not dataset-wide results.

| Arm | HOTA % | IDF1 % | IDSW | FN | Output boxes | Output IDs |
|---|---:|---:|---:|---:|---:|---:|
| Simple |64.7157|63.5562|52|67|2660|128|
| Legacy J1 |13.6083|7.7990|0|2230|226|1|
| Competition v2 |13.5220|7.6490|0|2230|226|1|
| Current crops v3 |13.6705|7.8107|0|2235|222|1|
| Current crops + memory v4 |13.6705|7.8107|0|2235|222|1|

The verified input fixes do not resolve identity collapse. See the paired
metric, raw-probability and output-count receipt in
`EVAL_NO_MOT20_J1_INPUT_CONTROL_RESULTS.json`.

Per-video SportsMOT receipt scope labels have a separate reporting erratum in
`EVAL_NO_MOT20_SCOPE_LABEL_ERRATUM.json`. The full226-frame video is not the
entire45-video validation set; original frozen metrics and predictions remain
unchanged. Main aggregate tables already retain this distinction.

Final control result: v4 completes226 frames with prediction SHA identical to
v3 (`79a5ccae3911d18ffbff8d66119a71c300fe3232691fbc0947ca091c145591bb`).
HOTA13.6705%, IDF17.8107%, IDSW0, FN2235; both produce222 boxes/one ID.
The one-ID full-video trace does not exercise simultaneous multi-identity
memory assignment. The noncontiguous two-ID request regression establishes
transport correctness, not a GPU tracking benefit on such cases.

The full TRAIN audit also finds39,985 rows with nonempty historical tracks,
but zero rows with more than two media entries. The only declared images are
the current frame and current candidate crops. This J1 recipe does not supply
past visual identity crops. Restoring current crops therefore does not restore
a trained past-appearance memory pathway. The performance contribution of
this training-design limit is not separately identified here. No retraining
or unregistered history-image intervention was performed.

A bounded owner now backfills the current v4 protocol on already Simple-scored
complete videos, sharing existing physical GPU3 and reading the immutable
candidate caches. Subsequent main-queue J1 invocations also use v4. Earlier
variants remain separate in the reports, and net gains always pair exact
completed scopes. No new detector service or fifth physical GPU is allocated.
