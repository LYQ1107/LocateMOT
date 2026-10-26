# Measured four-GPU production window

Status: measured observation window; full registered evaluation remains INCOMPLETE.

Source: `419a2ce360ed4e6b71618b98a9a4c45ec95cd7bb`. Model: final G2 step18227 `7c4a81349e333a304e32f42cf1999db289802d8c31eba415b85827038d14801f`.

In197.76 seconds the existing queue persisted64 new query-frames and83 new native requests. Aggregate production throughput was0.3236 query-frames/s and0.4197 native requests/s. Partial TAO chunks can cross observation boundaries, so requests need not equal the chunk count of newly completed frames.

| Dataset in this window | Newly persisted frames | Rejected frames |
|---|---:|---:|
| animaltrack | 62 | 1 |
| tao_amodal | 2 | 2 |

Sampled task-process residency in this window totals0.213189 GPU-hours across at most4 registered physical GPUs. Telemetry covers197.76 seconds; largest adjacent sample gap7.70 seconds.

| Physical GPU | Sampled residency GPU-hours | Sampled task-process peak MiB |
|---|---:|---:|
| 1 | 0.048392 | 18194 |
| 3 | 0.054932 | 22236 |
| 4 | 0.054932 | 18194 |
| 6 | 0.054932 | 22236 |

This is one production window under shared load, including association, service rotation, recovery and rejected frames. Its task mix contains AnimalTrack and sparse TAO frames; it does not represent every dataset. Residency includes idle loaded models and is not kernel busy time. Foreign utilization is not attributed to GroundJEV. Memory samples are lower bounds on instantaneous peaks. These are not full-run GPU-hours, physical video FPS or evidence of fourfold acceleration.

The observer started/stopped no inference process, made no additional detector request, read no GT and changed no protocol. Its registration and original receipt are represented in `EVAL_NO_MOT20_RESOURCE_MEASUREMENT.json`.

```bash
CUDA_VISIBLE_DEVICES='' python3 tools/groundjev/eval_no_mot20/measure_running_throughput.py \
  --minimum-new-frames 64 --max-seconds 600 \
  --out /data3/user/LocateMOT-runs/GroundJEV_Eval_NoMOT20_4GPU/request_local_seed_v2/engineering_checks/production_throughput_v1.json
```
