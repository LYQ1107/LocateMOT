#!/bin/bash
# Evaluate the L10-fix checkpoint on one TAO-val shard with several
# NEW-margin values, then merge each variant (TETA on that shard).
set -e
cd /data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT
PY=/home/lwr/anaconda3/envs/locatemot/bin/python
CK=outputs/l10/checkpoints/uidm_l10_fix/latest.pt
PBD=outputs/l9/cache/tao_val_pbd

idx=0
for m in 0.0 0.5 1.0 2.0; do
  gpu=$((idx + 1)); idx=$((idx + 1))
  (
    $PY tools/eval_l8_ovmot.py --ckpt $CK \
      --out outputs/l10/trackeval/ovmot_margin_$m \
      --gpu $gpu --shard 2 --num-shards 4 --pbd-cache $PBD \
      --new-margin $m \
      > outputs/l10/logs/ovmot_margin_$m.log 2>&1
    $PY tools/eval_l8_ovmot.py --out outputs/l10/trackeval/ovmot_margin_$m \
      --merge-only --pbd-cache $PBD \
      >> outputs/l10/logs/ovmot_margin_$m.log 2>&1
  ) &
done
wait
echo "[margin sweep] all done"
