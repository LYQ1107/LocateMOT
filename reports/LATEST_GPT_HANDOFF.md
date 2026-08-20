# Latest GPT Handoff — LocateMOT Stage L11–L12

Date: 2026-08-20

## State

- Project: LocateMOT, root
  `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`.
- Git: see `git log -1` (reports/tools committed incrementally).
- Running: L11 low-LR continuation
  (`outputs/l11/checkpoints/uidm_l11_main/`, resume step10000 ->
  max-steps 13000, p_rmot=0.3/p_ovmot=0.3, log
  `outputs/l11/logs/train_l11_main.log`).
- A fresh-optimizer balance run (p_rmot=0.45/p_ovmot=0.20) was tried
  and REJECTED at step 1000: RMOT-Dance 32.06 / ordinary Macro 0.4663,
  both worse than the low-LR continuation (33.26 / 0.4913 at s11k).

## Key results (locked)

- OVMOT repair SUCCESS: `uidm_l11_main/step10000.pt` full TAO val
  TETA 36.63 / AssocA 37.10 (L9-ovmot: 33.79 / 29.34).
- KITTI repair (same L9-ovmot ckpt): HOTA 3.74 -> 6.28, MOTA -4153 ->
  -766, DetA 0.93 -> 3.09 with whitelist+NMS+CLIP top-12 candidates.
- Pseudo-track generator: 99.26% same-ID precision, 17,779 tracklets /
  102,093 pseudo candidates over 500 TAO train videos.
- RMOT-Dance (calibrated thr -0.3): s10k 32.49 / s11k 33.26 HOTA
  (L9-ovmot 36.79) — balance run is targeting this.
- Ordinary Macro AssA: s10k 0.4855 / s11k 0.4913 (L9-ovmot 0.5056).
- L12 frozen prompt-seeded (DAVIS 2017 val, 10 multi-object videos):
  persistence mask 0.465 / box 0.410 / point 0.418 at thr -1; mask and
  point more robust than box.  Joint fine-tune NOT launched (mixed
  signal; L11 balance prioritized).

## Next steps

1. Wait for `uidm_l11_bal` checkpoints; evaluate at step 1000/2000:
   RMOT-Dance (`eval_l8_rmot.py --threshold-file
   outputs/l9/calib/threshold_l9.json`), ordinary
   (`eval_l8_ordinary.py`), then full TAO TETA
   (`eval_l9_ovmot_full.py`).
2. Fill final numbers into
   `reports/STAGE_L11_L12_UNIFIED_TRACKING_FINAL_REPORT.md`.
3. Optionally re-run KITTI eval with the final balanced checkpoint
   (`tools/eval_l11_rmot_kitti.py --clip-topk 12`).
4. Decide ICLR readiness (currently NEAR_READY; final numbers pending).
5. Final git commit (do not commit datasets/caches/checkpoints).

## Paths

- Checkpoints: `outputs/l11/checkpoints/uidm_l11_main/step10000.pt`
  (best OVMOT), `uidm_l11_bal/` (in progress).
- Pseudo sidecars: `outputs/l11/data/pseudo_tracks/`.
- KITTI repaired data: `outputs/l11/data/rmot_kitti/`;
  calibration `results/l11/kitti_calibration.json`.
- DAVIS: `outputs/l12/data/davis/`, PBD `outputs/l12/cache/davis_pbd/`,
  seeds `outputs/l12/data/davis_seed_pbd.json`; results
  `results/l12/davis_{mask,box,point}.json`.
- Logs: `outputs/l11/logs/`.
