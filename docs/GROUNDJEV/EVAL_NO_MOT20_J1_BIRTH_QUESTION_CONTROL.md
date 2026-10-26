# Frozen J1 empty-history question control

Status: preregistered; no measured tracking improvement claimed.

The completed J1 TRAIN recipe contains3,471 birth rows with a dedicated
empty-history question. The runtime used the ordinary identity question even
when no persistent identity existed. This control uses the exact TRAIN birth
question only in that true empty-history state, and retains the normal indexed
question after a live history exists. The known missing-index ambiguity of the
TRAIN birth question is preserved and disclosed.

Arm: `j1_train_birth_prompt_v5`; baseline: `j1_current_crops_memory_v4`.
Both use the same226-frame SportsMOT video and native candidate SHA
`46fce75613893b8642a0b0be118b0462e2e89996e38fba2feec7b711159e34b6`,
J1 step1200, current crops, aligned memory transport and xyxy competition.
NEW, NO_MATCH and REJECT all remain native model options; no probabilities,
thresholds, GT inputs, seeds or model weights change. This is a complete-video
question-protocol experiment, not a claim of dataset-wide generalization.

```bash
CUDA_VISIBLE_DEVICES=4 /data2/user/LocateMOT_GroundJEV/.venv-onejev-py310/bin/python \
  tools/groundjev/eval_no_mot20/run_j1_input_controls.py --birth-only --gpu 4 \
  --job /data3/user/LocateMOT-runs/GroundJEV_Eval_NoMOT20_4GPU/request_local_seed_v2/jobs/sportsmot__v_9MHDmAMxO5I_c003__d743f9a1134bb802
```

The main v4 evaluation and four physical GPU limit remain in force. No extra
SGLang service is started. Receipts are under
`engineering_checks/j1_birth_question_contract/`.
