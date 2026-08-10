#!/usr/bin/env bash
# Resume interrupted L1-B road cache (BDD done; TAO partial).
# One blocking command; children are nohup'd and survive agent teardown.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
PY=/home/lwr/anaconda3/envs/locatemot/bin/python
GPUS=(0 1 2 4 6 7 8 9)
PID_FILE=runs/l1b_road_cache_pids.txt
: > "$PID_FILE"
for s in 0 1 2 3 4 5 6 7; do
  nohup "$PY" -u tools/cache_l1b_locateanything.py \
    --split-config configs/l1_b/road_v2_videos.json \
    --gpu "${GPUS[$s]}" --shard "$s" --num-shards 8 \
    > "runs/l1b_road_cache_shard$s.log" 2>&1 &
  echo "$! ${GPUS[$s]} $s" >> "$PID_FILE"
done
cat "$PID_FILE"
wait
echo "L1B_ROAD_CACHE_RESUME_DONE"
