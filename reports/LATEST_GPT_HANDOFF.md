# LocateMOT — Stage L8 GPT Handoff

Date: 2026-08-14
Stage: L8 Specification-Conditioned Unified MOT
Status: **SUPPORTED**

## In one sentence

We connected RMOT (Refer-Dance) to the same UIDM identity-dynamics core
used for ordinary MOT and OVMOT, and obtained positive results on all
three formulations with a single shared checkpoint.

## Where the project lives

- Root: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`
- Conda: `/home/lwr/anaconda3/envs/locatemot`
- Final report: `reports/STAGE_L8_UNIFIED_MOT_FINAL_REPORT.md`
- This handoff: `reports/LATEST_GPT_HANDOFF.md`

## Headline numbers (same core class, one checkpoint per variant)

| Formulation | Dataset | Metric | L8 result | Reference |
|---|---|---|---|---|
| Ordinary MOT | Dance/BDD/MOT17/MOT20 | Macro AssA | 0.5045 (v2) / 0.5087 (B1) | L6 0.4922 |
| OVMOT | TAO val official TETA | TETA / AssocA | 34.33 / 30.44 (v2) | L7 33.94 / 29.51 |
| RMOT | Refer-Dance 40 queries | HOTA / AssA | 35.20 / 28.63 (v2); 37.88 / 31.02 (B1) | iKUN 29.06 / 33.35 |

Protocol caveat: RMOT baselines use a different person detector
(ByteTrack/DLA vs LocateAnything-3B); DetA is not directly comparable.

## Method

- Unified Specification: frozen CLIP ViT-B/32 text embeddings (category /
  "all objects" / referring expression).
- Unified Observation Adapter: gated CLIP+spec semantic residue; either
  added to UIDM candidate tokens (L8-B1 `sem_in_core=True`) or used only
  by a relevance head (L8-B2 identity-pure).
- Shared UIDM core: L6 `uidm_full` (large) with PBD box-end identity
  tokens; PBD-dropout 0.15 so the same core works without PBD (TAO).
- Training: 4 GPUs, DDP, seed 20260806, MOT+RMOT balanced sampler;
  tracking loss + relevance BCE.

## Critical bug fixed during the stage

The L8 evaluation accidentally used the PBD coord-mean token while the
core was trained on the box-end token. This produced a false "semantics
destroy identity" negative result. After fixing the feature key, both L8
variants preserve (and slightly improve) ordinary MOT.

## Key files

- Checkpoints: `outputs/l8/checkpoints/uidm_l8_v2/latest.pt`,
  `outputs/l8/checkpoints/uidm_l8_final/latest.pt`
- Eval results: `outputs/l8/trackeval/{rmot_v2_fix,uidm_l8_v2_fix,
  ovmot_v2e,rmot_semcore_fix,uidm_l8_semcore_fix,ovmot_semcore}/`
- Calibration: `outputs/l8/calib/threshold_v2.json` (threshold -0.1,
  train F1 0.9175)
- CSV: `results/l8/results_summary.csv`
- Data: `data/refer_dance/` (symlinked, read-only)
- Reference repos (outside git): `LocateMOT_reference_repos/{iKUN,
  rmot_official,temp_rmot,motip}`

## Honest limitations

- RMOT AssA (28.6-31.0) below iKUN's 33.35; crowded dance scenes remain
  hard for language-driven identity.
- OVMOT uses the semantic-only (PBD-zero) regime because TAO val has no
  cached PBD; full PBD+CLIP OVMOT is future work.
- RMOT eval has only 40 queries; treat numbers as indicative.
- The two variants differ by a few tenths of a point; no multi-seed error
  bars.

## Next steps

1. Compute TAO val PBD cache (LocateAnything-3B) for a full-observation
   OVMOT run.
2. Add Refer-KITTI RMOT once KITTI images are available.
3. Longer joint training / more RMOT data.
4. Write the paper around the core claim in `STAGE_L8_UNIFIED_MOT_FINAL_REPORT.md`
   section 10.

