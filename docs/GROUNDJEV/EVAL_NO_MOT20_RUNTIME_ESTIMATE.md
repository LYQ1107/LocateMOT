# Measured runtime and remaining detection cost

Status: INCOMPLETE. Estimates use actual input-specific native request times, including all TAO vocabulary blocks. They are provisional and exclude future JEV cost.

| Dataset | Cached frames / registered | Requests saved | Measured inputs | FPS | Remaining GPU-hours |
|---|---:|---:|---:|---:|---:|
| sportsmot | 1509/26970 | 1509 | 32 | 0.363 | 24.156 |
| animaltrack | 1746/13167 | 1746 | 32 | 0.225 | 13.307 |
| refer_kitti_v2 | 1254/854073 | 1254 | 32 | 0.950 | 446.756 |
| tao_amodal | 104/36562 | 1361 | 32 | 0.002 | 3320.189 |

sportsmot: registered 32-input preflight; mean 2.757 s, p95 2.990 s.
animaltrack: registered 32-input preflight; mean 4.442 s, p95 5.361 s.
refer_kitti_v2: registered 32-input preflight; mean 1.052 s, p95 1.151 s.
tao_amodal: registered 32-input preflight; mean 463.046 s, p95 510.957 s.

Remaining-work forecasts prefer all completed native video/query scopes over the original short preflight. The early completed videos are a small, size-prioritized sample, so these are provisional cost extrapolations, not completion-date guarantees.
sportsmot forecast: all completed native video/query caches, 1509 measured frames.
animaltrack forecast: all completed native video/query caches, 1691 measured frames.
refer_kitti_v2 forecast: all completed native video/query caches, 1254 measured frames.
tao_amodal forecast: all completed native video/query caches, 92 measured frames.

Workers: four shared A800 GPUs, physical 3/4/6/1, UUIDs and memory limits in the committed configuration. A service runs one native request at a time. Allocation time and sampled utilization are recorded separately. No fourfold speedup is assumed.

Association replay FPS is separate from detector throughput. A sum of measured stage costs is an estimate, never a directly measured end-to-end FPS.
