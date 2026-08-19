# LocateMOT — Stage L10 GPT Handoff (final)

Date: 2026-08-18

## In one sentence

Stage L10 scaled full-PBD OVMOT supervision to all 500 TAO train videos
(DLA detections + C-TAO continuous GT, 322.8k candidates), and the
expanded stream **failed**: full-PBD TAO AssocA collapsed to ~7-8
(vs L9-adapted 29.34), while ordinary MOT and RMOT stayed stable.  The
paper's main shared checkpoint remains the L9-ovmot checkpoint; L10
provides a rigorous supervision-coverage negative result plus the
Refer-KITTI-V2 second RMOT benchmark.

## Final state

- Main shared checkpoint: `outputs/l9/checkpoints/uidm_l9_main_ovmot/
  latest.pt` (ordinary Macro AssA 0.5056; RMOT Refer-Dance HOTA 36.79 /
  AssA 29.86; full-PBD TAO TETA 33.79 / AssocA 29.34).
- L10 v1 checkpoint: `outputs/l10/checkpoints/uidm_l10_main/
  v1_final_step15000.pt` (ordinary 0.5041; RMOT 36.32/28.79; OVMOT
  26.39/7.26).
- L10 v2 checkpoint (target fix): `outputs/l10/checkpoints/
  uidm_l10_fix/latest.pt` (ordinary 0.4982; RMOT 36.10/29.18; OVMOT
  26.24/7.86).
- TAO-train full-PBD cache: COMPLETE (`outputs/l10/cache/tao_train_pbd`,
  5.5 GB, 18,274 frames, merged 100%).
- KITTI-V2 data + DLA dets + PBD cache + RMOT eval: COMPLETE
  (`outputs/l10/cache/kitti_pbd`; official 862-query TrackEval with the
  L9-ovmot shared checkpoint -> HOTA 3.74 / DetA 0.93 / AssA 16.72 /
  MOTA -4153 / IDF1 0.97; Detic-SwinB candidates, threshold -0.3).
  The low score reflects 50 candidates/frame (DetPr ~1%) and Refer-Dance
  only language training; reported as a cross-domain data point, not a
  fair comparison with TempRMOT.

## Key scientific findings

1. DLA (Detic-SwinB) candidates on TAO train match C-TAO base GT at only
   ~3.5% (val ~5%).  With the L9 target scheme (every unmatched candidate
   = positive relevance + NEW birth), the model learns to birth a new id
   for nearly every detection (verified: 1,612 unique ids over 1,650
   rows in a sample video vs 374 in L9) -> AssocA collapses.
2. The target-correction retrain (relevance negatives + score-gated NEW,
   thr 0.4) does not rescue it (7.86); eval NEW margins 0-2 only reach
   subset AssocA ~6.4.  Dense continuous GT covering detector detections
   (C-TAO base_and_novel is still insufficient, +51 tracks) or a
   temporal pseudo-track self-supervision would be required.
3. Ordinary MOT and RMOT are robust across L10 variants, so the failure
   is OVMOT-training-supervision-specific, not global model damage.
4. Training speed: batch 4 -> 8 (+27% clips/s), 15k steps instead of 30k
   for the same sample budget, LR scaled 2x.

## Where things live

- Final report: `reports/STAGE_L10_ICLR_CLOSURE_FINAL_REPORT.md`
- Literature/code audit: `reports/l10_literature_and_code_audit.md`
- Candidate-generation audit: `reports/l10_tao_candidate_generation_audit.md`
- Refer-KITTI/V2 audit: `reports/l10_refer_kitti_and_v2_audit.md`
- Full-PBD cache: `reports/l10_tao_train_full_pbd.md`
- Speed audit: `reports/l10_training_speed_audit.md`
- Scaling ablation + failure analysis: `reports/l10_supervision_scaling_ablation.md`,
  `reports/l10_failure_analysis.md`
- Results: `reports/l10_{mot,ovmot,rmot}_results.md`
- Live status: `outputs/l10/STATUS.md`

## Next steps

1. Optionally implement temporal pseudo-track self-supervision for
   unmatched OVMOT detections (the evidence-based direction for
   recovering full-PBD OVMOT).
2. Git commit the L10 stage.
