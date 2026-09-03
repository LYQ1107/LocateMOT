#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L87B
ASSET_ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT
PY=/home/lwr/anaconda3/envs/masaenv_debug/bin/python
CACHE="$ASSET_ROOT/outputs/l85/features/fit_dev_eval_full_attempt2"
export LOCATEMOT_ASSET_ROOT="$ASSET_ROOT"
export PYTHONPATH="$WORK_ROOT:$ASSET_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

"$PY" -m compileall -q \
  locatemot/rmot/l87_eval_policy.py tools/l87b_reselect_existing_l86.py \
  tools/l87b_eval_fixed_semantic.py tools/l87b_infer_fullvideo.py \
  tools/l87b_run_trackeval.py

"$PY" tools/l87b_reselect_existing_l86.py \
  --out outputs/l87b/selection
"$PY" tools/l87b_eval_fixed_semantic.py --cache "$CACHE" \
  --selection outputs/l87b/selection/corrected_l86_selection.json \
  --out outputs/l87b/eval/fixed_semantic --device cuda:0
"$PY" tools/l87b_infer_fullvideo.py --cache "$CACHE" \
  --selection outputs/l87b/selection/corrected_l86_selection.json \
  --out outputs/l87b/trackeval/fullvideo_validation --device cuda:0 --query-batch-size 8
"$PY" tools/l87b_run_trackeval.py \
  --inference-root outputs/l87b/trackeval/fullvideo_validation \
  --out outputs/l87b/trackeval/fullvideo_eval --tracker-name l87b
