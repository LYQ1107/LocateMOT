#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L87A
ASSET_ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT
PY=/home/lwr/anaconda3/envs/masaenv_debug/bin/python
CACHE="$ASSET_ROOT/outputs/l85/features/fit_dev_eval_full_attempt2"
export LOCATEMOT_ASSET_ROOT="$ASSET_ROOT"
export PYTHONPATH="$WORK_ROOT:$ASSET_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

"$PY" -m compileall -q \
  locatemot/rmot/l87_eval_policy.py locatemot/rmot/l87a_losses.py \
  tools/l87a_contract_smoke.py tools/l87a_train_full_rmot.py tools/l87a_score_dev.py \
  tools/l87a_select_checkpoint.py tools/l87a_eval_fixed_semantic.py \
  tools/l87a_infer_fullvideo.py tools/l87a_run_trackeval.py

"$PY" tools/l87a_contract_smoke.py --cache "$CACHE" \
  --out outputs/l87a/audit/contract_smoke --device cuda:0

"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=3 --master_port="${MASTER_PORT:-29687}" \
  tools/l87a_train_full_rmot.py --epochs 40 --seed 20260829 --cache "$CACHE" \
  --out outputs/l87a/train/joint40 --effective-clip-batch 9 --bf16

"$PY" tools/l87a_score_dev.py --cache "$CACHE" --checkpoint-dir outputs/l87a/train/joint40 \
  --out outputs/l87a/eval/dev_cheap --device cuda:0
"$PY" tools/l87a_select_checkpoint.py --scores outputs/l87a/eval/dev_cheap \
  --out outputs/l87a/eval/dev_selection
"$PY" tools/l87a_eval_fixed_semantic.py --cache "$CACHE" \
  --selection outputs/l87a/eval/dev_selection/checkpoint_selection.json \
  --out outputs/l87a/eval/fixed_semantic --device cuda:0
"$PY" tools/l87a_infer_fullvideo.py --cache "$CACHE" \
  --selection outputs/l87a/eval/dev_selection/checkpoint_selection.json \
  --out outputs/l87a/trackeval/fullvideo_validation --device cuda:0 --query-batch-size 8
"$PY" tools/l87a_run_trackeval.py \
  --inference-root outputs/l87a/trackeval/fullvideo_validation \
  --out outputs/l87a/trackeval/fullvideo_eval --tracker-name l87a
