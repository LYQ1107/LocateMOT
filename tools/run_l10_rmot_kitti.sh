#!/bin/bash
# Parallel Refer-KITTI-V2 RMOT prediction (4 shards) + single TETA eval.
set -e
cd /data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT
PY=/home/lwr/anaconda3/envs/locatemot/bin/python
CK=outputs/l9/checkpoints/uidm_l9_main_ovmot/latest.pt
OUT=outputs/l10/trackeval/rmot_kitti
THR=outputs/l9/calib/threshold_l9.json

for s in 0 1 2 3 4 5 6 7; do
  gpu=$((s / 2 + 1))
  (
    $PY tools/eval_l10_rmot_kitti.py --ckpt $CK --out $OUT \
      --gpu $gpu --shard $s --num-shards 8 --predict-only \
      --threshold-file $THR \
      > outputs/l10/logs/rmot_kitti_shard$s.log 2>&1
  ) &
done
wait
echo "[rmot_kitti] predictions done"

$PY tools/eval_l10_rmot_kitti.py --ckpt $CK --out $OUT \
  --gpu 1 --eval-only --threshold-file $THR \
  > outputs/l10/logs/rmot_kitti_eval.log 2>&1
echo "[rmot_kitti] eval done"
