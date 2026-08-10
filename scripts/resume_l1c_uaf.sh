#!/usr/bin/env bash
# Resume Stage L1-C UAF training from the latest checkpoint.
# Usage: bash scripts/resume_l1c_uaf.sh [GPU]
set -euo pipefail
ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT
GPU="${1:-1}"
cd "$ROOT"
CKPT=$(ls -1 outputs/l1_c/checkpoints/uaf/step*.pt 2>/dev/null | sort -V | tail -1 || true)
ARGS=(--gpu "$GPU" --steps 50000 --batch 8 --lr 1e-4 --warmup 500 --save-every 5000 --out outputs/l1_c/checkpoints/uaf)
if [[ -n "$CKPT" ]]; then
  ARGS+=(--resume "$CKPT")
  echo "[resume] from $CKPT" >&2
fi
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT" \
  /home/lwr/anaconda3/envs/locatemot/bin/python -u tools/train_l1c_uaf.py "${ARGS[@]}"
