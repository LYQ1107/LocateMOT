# GroundJEV final-G2 coupled pilot and video diagnostic (2026-10-22)

G2 Clean is terminal at step 18227. This record is the first bounded real-GPU
coupling run after that terminal checkpoint; it does not restart or extend the
G2 formal training.

## Coupled readout pilot

- **Backbone:** G2 Clean, frozen, adapter step 18227
- **Backbone SHA:** `3cf09355cde8cf4877cdc76e0a72301d572c1d6263b2831f8a03368285bb65e2`
- **Adapter SHA:** `cf1b5a6f626810ead94eb4e16adaf3feb6c6da3f99ca5023b6b5c234726f5742`
- **Joint manifest SHA:** `cc632115aa2b87161dc7b8413d08c3c471e1225179559a76509e7366190764c1`
- **Checkpoint:** `/data3/user/LocateMOT-checkpoints/groundjev_native_three_task_v1/groundjev-native-three-task-step000240.pt`
- **Checkpoint SHA:** `94479584761abe287e5e13f5eb11151246c033897e8bbb5c7b61a77bf44378aa`
- **Tasks:** MOT, RMOT membership/Yes-No, and OVMOT group/leaf category readouts
- **Rows:** 24 per task; 72 total; 240 requested optimizer steps and 72 effective updates per task
- **Architecture contract:** one shared visual input/forward per unique input, zero independent VLM instances, frozen G2 backbone
- **Resource:** GPU3; peak allocated memory about 9.35 GiB; 74.16 s wall time

The same sampled rows changed readout accuracy from MOT `0.4444` to `1.0000`,
RMOT `0.6667` to `0.8750`, and OVMOT `0.0417` to `0.7083`. These are small
training-sample readout fits, not HOTA/TETA or generalization evidence. Twenty-four
identity targets were legally masked because causal history slots were absent.
The native DLM prompt was bounded at 16 candidate boxes; 19 sampled rows were
truncated with the target retained, while original source rows were preserved.

The authoritative receipt is at
`/data3/user/LocateMOT-checkpoints/groundjev_native_three_task_v1/receipt.json`.

## Fixed-candidate MOT video diagnostic

The coupled readout was replayed on the immutable 128-frame `MOT20-03` candidate
archive (frames 538--665, 14,784 candidates) using the same cached native region
features. This cache predates the terminal G2 model, so the result is a
train-scope association/runtime diagnostic rather than an end-to-end G2 benchmark.

| HOTA | DetA | AssA | IDF1 | IDSW | replay FPS |
|---:|---:|---:|---:|---:|---:|
| 0.01518 | 0.02023 | 0.01142 | 0.01601 | 182 | 13.27 |

There were 319 predicted boxes on 107 frames and 7 empty-decision frames. The
TrackEval receipt is
`/data2/user/LocateMOT_GroundJEV/outputs/groundjev/coupled/fixed_candidate_mot20_03_final_g2_readout_v1/official_metrics.json`.

## Independent evaluation queue

The original GPU7 held-out owner remains preserved and waiting because unrelated
processes occupy GPU7. A second owner was started on free GPU3 at
`/data2/user/LocateMOT_GroundJEV/outputs/groundjev/heldout/g2_clean_independent_gpu3_v1`.
It uses the same parser, decoder, tracker, scorer, and overlap audit for G0 and
G2 across the registered MOT/RMOT/OVMOT scopes. Its metrics are not claimed here
until the owner reaches a terminal receipt.

## Scope statement

No independent OneJev or YesTrack model was trained in this run. JEV identity,
memory, lifecycle, and OVMOT category heads are coupled readouts on the single
G2 visual backbone. Historical G2 results and checkpoint directories were not
overwritten.
