# Measured runtime and remaining detection cost

Status: INCOMPLETE. Estimates use actual input-specific native request times, including all TAO vocabulary blocks. They are provisional and exclude future JEV cost.

| Dataset | Cached frames / registered | Requests saved | Measured inputs | FPS | Remaining GPU-hours |
|---|---:|---:|---:|---:|---:|
| animaltrack | 2550/13167 | 2550 | 32 | 0.225 | 13.297 |
| refer_kitti_v2 | 1672/854073 | 1672 | 32 | 0.950 | 454.100 |
| sportsmot | 2050/26970 | 2050 | 32 | 0.363 | 24.767 |
| tao_amodal | 137/36562 | 1793 | 32 | 0.002 | 3117.220 |

animaltrack: registered 32-input preflight; mean 4.442 s, p95 5.361 s.
refer_kitti_v2: registered 32-input preflight; mean 1.052 s, p95 1.151 s.
sportsmot: registered 32-input preflight; mean 2.757 s, p95 2.990 s.
tao_amodal: registered 32-input preflight; mean 463.046 s, p95 510.957 s.

Remaining-work forecasts prefer all completed native video/query scopes over the original short preflight. The early completed videos are a small, size-prioritized sample, so these are provisional cost extrapolations, not completion-date guarantees.
animaltrack forecast: all completed native video/query caches, 2308 measured frames.
refer_kitti_v2 forecast: all completed native video/query caches, 1672 measured frames.
sportsmot forecast: all completed native video/query caches, 2050 measured frames.
tao_amodal forecast: all completed native video/query caches, 124 measured frames.

Workers: four shared A800 GPUs, physical 3/4/6/1, UUIDs and memory limits in the committed configuration. A service runs one native request at a time. Allocation time and sampled utilization are recorded separately. No fourfold speedup is assumed.

Association replay FPS is separate from detector throughput. A sum of measured stage costs is an estimate, never a directly measured end-to-end FPS.
