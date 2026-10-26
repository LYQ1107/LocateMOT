# Frozen J1 empty-history question control

Status: complete226-frame control; association quality degrades despite higher detection retention.

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

The frozen v5 result has HOTA18.8133%, DetA44.4246%, AssA7.9778%,
IDF116.8554%, IDSW844, FN1177 and FP92. It outputs1356 boxes/7 IDs, compared
with v4’s222 boxes/1 ID. This fixes the initial multi-ID birth failure while
exposing poor identity association. It is not an effective improvement over
Simple (HOTA64.7157%, AssA54.7984%, IDF163.5562%, IDSW52 on the same scope),
and the main v4 evaluation remains unchanged.

A post-hoc adjacent-frame diagnostic finds1237 observations with a previous
GT-matched output and a current GT-matched candidate. All previous IDs are
in the model options; the raw best existing-ID choice preserves that ID in
344 cases. Actual assignment preserves219, switches748 and omits270. Mean
model probability of the previous ID is0.2313 versus0.5406 for the strongest
other ID. Every identity request has only the current frame/current crops.
These diagnostics are computed after inference and do not replace official
IDSW or establish a causal fraction of the error. See the separate JSON
receipts. No runtime GT input, probability change or new training occurred.
