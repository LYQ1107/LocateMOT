# J1 causal visual-history evidence control

Status: complete full-video diagnostic; the main evaluation remains v4.

V5 fixes initial births but produces844 ID switches on226 frames. The model
has no past identity images in any of its226 requests; the frozen TRAIN
recipe likewise contains only current images/current crops.

Register `j1_causal_history_v6` against the same complete v5 video and G2
candidate SHA46fce75613893b8642a0b0be118b0462e2e89996e38fba2feec7b711159e34b6.
Only add one sheet of the shortlisted IDs’ actual causal anchor/long/recent
images. Those images are the existing tracker’s predicted-ID memory before
the current frame update. Full current image/current crops, birth question,
model step1200, parser, options, thresholds and Hungarian allocation remain
unchanged. No numeric motion prior or probability reweighting is added.

This is a new inference-evidence intervention. Past visual crops were absent
from J1 TRAIN, so it must not be described as restoration of a trained memory
pathway. A regression verifies that the history image is from the prior frame
and the PIL images never enter JSON state. That is engineering evidence only.

```bash
CUDA_VISIBLE_DEVICES=4 /data2/user/LocateMOT_GroundJEV/.venv-onejev-py310/bin/python \
  tools/groundjev/eval_no_mot20/run_j1_input_controls.py --history-only --gpu 4 \
  --job /data3/user/LocateMOT-runs/GroundJEV_Eval_NoMOT20_4GPU/request_local_seed_v2/jobs/sportsmot__v_9MHDmAMxO5I_c003__d743f9a1134bb802
```

The owner checks actual registered-GPU UUID/free memory and leaves detector
restart margin. No new SGLang replica, fifth physical GPU, training row or
checkpoint is created. Full-video receipts and failures remain under
`engineering_checks/j1_causal_history_contract/`.

## Complete-video result

All226 frames completed with exactly the registered G2 candidates and no G2
regeneration. V5 and V6 first-frame candidates and outputs match; the first
request has two images and the remaining225 identity requests have three.

| Same226-frame video | HOTA (%) | DetA (%) | AssA (%) | IDF1 (%) | IDSW | FN | Output boxes | IDs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Simple | 64.7157 | 76.9959 | 54.7984 | 63.5562 | 52 | 67 | 2660 | 128 |
| J1 main v4 | 13.6705 | 7.4397 | 25.1336 | 7.8107 | 0 | 2235 | 222 | 1 |
| V5 TRAIN birth question | 18.8133 | 44.4246 | 7.9778 | 16.8554 | 844 | 1177 | 1356 | 7 |
| V6 causal past images | 17.5892 | 38.6148 | 8.0255 | 14.2299 | 729 | 1374 | 1143 | 7 |

V6 association replay is0.3843 FPS; complete control wall time600.95 seconds.
GPU sharing makes this a measured execution cost, not a controlled speed gain.
The v6 prediction SHA is
`fae8cef521552ead5d8a7ff32325b8053cc022fefdeb2e88ca795c3da5f1f6d6`.

Post-hoc GT matching finds1045 adjacent observations where the preceding
output ID is still in the current options. Raw best existing identity
preserves that ID in291 cases; the actual allocator preserves it in171,
changes it in639, and emits no output in235. These are conditional diagnostic
counts, not the official IDSW or a causal decomposition.

Decision: retain this negative diagnostic and do not replace main v4. Neither
the birth-question intervention nor added untrained visual history produces
an advantage over Simple. V6 reduces retained detections versus v5; its lower
IDSW alone does not establish improved association. The main frozen datasets
continue independently of this local failure. Detailed metrics, probabilities,
media counts and identical-input receipts are in
`EVAL_NO_MOT20_J1_CAUSAL_HISTORY_RESULTS.json`.
