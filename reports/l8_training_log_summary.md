# Stage L8 — Training Log Summary

All runs use seed 20260806, UIDM-large, 4 GPUs (0/1/2/4), DDP batch 4/GPU.

## L8-A1 — frozen-core joint (uidm_l8_joint)

- init: L6 `uidm_full` core frozen + fresh adapter;
- 2,387 steps (387 + 2,000), p_rmot=0.5, relevance weight 0.2;
- log: `outputs/l8/uidm_l8_joint_train.log`, `..._train2.log`;
- result: RMOT HOTA 34.12 (invalid later; PBD eval-key bug), ordinary
  Macro AssA 0.28.

## L8-B1 — core fine-tune, sem-in-core (uidm_l8_final)

- init: uidm_l8_joint; unfrozen core, core LR 5e-5, adapter LR 1.2e-4;
- 3,000 steps; log: `outputs/l8/uidm_l8_final_train.log`;
- result: RMOT HOTA 33.71 (invalid later), ordinary Macro AssA 0.26;
  conclusion: semantic residue in the identity token stream is a negative
  result.

## L8-B2 — v2 identity-pure (uidm_l8_v2)

- init: hybrid = L6 core + L8-B1 adapter; `sem_in_core=False`;
- 2,500 steps, core LR 4e-5, adapter LR 1e-4, p_rmot=0.35,
  PBD-dropout=0.15;
- log: `outputs/l8/uidm_l8_v2_train3.log`;
- final checkpoint: `outputs/l8/checkpoints/uidm_l8_v2/latest.pt`.

## Eval logs

- RMOT: `outputs/l8/trackeval/rmot_v2_fix/` (40 GT queries, official RMOT
  TrackEval, calibrated threshold -0.1);
- ordinary: `outputs/l8/trackeval/uidm_l8_v2_fix/` (four-domain TrackEval);
- OVMOT: `outputs/l8/trackeval/ovmot_v2e/` (TAO val, official TETA);
- calibration: `outputs/l8/calib/threshold_v2.json` (train-set F1 0.9175).

