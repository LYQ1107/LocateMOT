# J1 causal visual-history evidence control

Status: preregistered; the main evaluation remains v4.

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
