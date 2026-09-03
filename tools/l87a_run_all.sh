#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L87A
ASSET_ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT
PY=/home/lwr/anaconda3/envs/masaenv_debug/bin/python
CACHE="$ASSET_ROOT/outputs/l85/features/fit_dev_eval_full_attempt2"
RUN_SUFFIX="${L87A_RUN_SUFFIX:-}"
AUDIT_OUT="outputs/l87a/audit/contract_smoke${RUN_SUFFIX}"
TRAIN_OUT="outputs/l87a/train/joint40${RUN_SUFFIX}"
DEV_OUT="outputs/l87a/eval/dev_cheap${RUN_SUFFIX}"
SELECTION_OUT="outputs/l87a/eval/dev_selection${RUN_SUFFIX}"
SEMANTIC_OUT="outputs/l87a/eval/fixed_semantic${RUN_SUFFIX}"
FULLVIDEO_OUT="outputs/l87a/trackeval/fullvideo_validation${RUN_SUFFIX}"
TRACKEVAL_OUT="outputs/l87a/trackeval/fullvideo_eval${RUN_SUFFIX}"
export LOCATEMOT_ASSET_ROOT="$ASSET_ROOT"
export PYTHONPATH="$WORK_ROOT:$ASSET_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

"$PY" -m compileall -q \
  locatemot/rmot/l87_eval_policy.py locatemot/rmot/l87a_losses.py \
  tools/l87a_contract_smoke.py tools/l87a_train_full_rmot.py tools/l87a_score_dev.py \
  tools/l87a_select_checkpoint.py tools/l87a_eval_fixed_semantic.py \
  tools/l87a_infer_fullvideo.py tools/l87a_run_trackeval.py tools/l87a_runtime_regression.py

"$PY" tools/l87a_contract_smoke.py --cache "$CACHE" \
  --out "$AUDIT_OUT" --device cuda:0

"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=3 --master_port="${MASTER_PORT:-29687}" \
  tools/l87a_train_full_rmot.py --epochs 40 --seed 20260829 --cache "$CACHE" \
  --out "$TRAIN_OUT" --effective-clip-batch 9 --bf16

"$PY" tools/l87a_score_dev.py --cache "$CACHE" --checkpoint-dir "$TRAIN_OUT" \
  --out "$DEV_OUT" --device cuda:0
"$PY" tools/l87a_select_checkpoint.py --scores "$DEV_OUT" \
  --out "$SELECTION_OUT"
"$PY" tools/l87a_eval_fixed_semantic.py --cache "$CACHE" \
  --selection "$SELECTION_OUT/checkpoint_selection.json" \
  --out "$SEMANTIC_OUT" --device cuda:0
"$PY" tools/l87a_infer_fullvideo.py --cache "$CACHE" \
  --selection "$SELECTION_OUT/checkpoint_selection.json" \
  --out "$FULLVIDEO_OUT" --device cuda:0 --query-batch-size 8
"$PY" tools/l87a_run_trackeval.py \
  --inference-root "$FULLVIDEO_OUT" \
  --out "$TRACKEVAL_OUT" --tracker-name l87a
