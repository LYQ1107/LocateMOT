#!/bin/bash
# Stage L10-fix: retrain from L9 final with corrected OVMOT targets.
#
# Evidence-based fix after the expanded-stream negative result:
# 1) OVMOT unmatched candidates get relevance target 0 (hard negatives),
#    not 1 (previously every candidate was a positive target);
# 2) NEW births are gated by detection score (--new-score-thr 0.4),
#    so low-score noise is NO_MATCH instead of a new track.
set -e
cd /data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT

OUT=outputs/l10/checkpoints/uidm_l10_fix
LOG=outputs/l10/logs/uidm_l10_fix_train.log
mkdir -p outputs/l10/logs

CUDA_VISIBLE_DEVICES=1,2,3,4 \
/home/lwr/anaconda3/envs/locatemot/bin/python -m torch.distributed.run \
  --nproc_per_node=4 --master_port=29521 \
  tools/train_l9_uidm.py \
  --out "$OUT" \
  --mode unified --model large --epochs 200 --batch 8 \
  --lr 2.4e-4 --lr-core 1e-4 \
  --gpu 1,2,3,4 --ddp --cond-gated \
  --max-steps 15000 \
  --resume outputs/l9/checkpoints/uidm_l9_main_ovmot/latest.pt \
  --ovmot-dir outputs/l10/data/tao_train \
  --p-rmot 0.3 --p-ovmot 0.3 --pbd-dropout 0.15 \
  --new-score-thr 0.4 \
  --save-every 1000 --workers 4 --prefetch 4 \
  --profile-steps 120 --profile-out outputs/l10/data/speed_profile_fix.json \
  2>&1 | tee "$LOG"
