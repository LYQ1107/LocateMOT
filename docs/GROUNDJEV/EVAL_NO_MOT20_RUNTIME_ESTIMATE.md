# Measured runtime and remaining detection cost

Status: INCOMPLETE. Estimates use actual input-specific native request times, including all TAO vocabulary blocks. They are provisional and exclude future JEV cost.

| Dataset | Cached frames / registered | Requests saved | Measured inputs | FPS | Remaining GPU-hours |
|---|---:|---:|---:|---:|---:|
| sportsmot | 974/26970 | 974 | 32 | 0.363 | 22.510 |
| animaltrack | 1379/13167 | 1379 | 32 | 0.225 | 13.924 |
| refer_kitti_v2 | 855/854073 | 855 | 32 | 0.950 | 426.523 |
| tao_amodal | 79/36562 | 1043 | 45 | 0.004 | 2710.041 |

sportsmot: registered 32-input preflight; mean 2.757 s, p95 2.990 s.
animaltrack: registered 32-input preflight; mean 4.442 s, p95 5.361 s.
refer_kitti_v2: registered 32-input preflight; mean 1.052 s, p95 1.151 s.
tao_amodal: completed full-video caches; registered 32-input preflight unfinished; mean 267.416 s, p95 PENDING s.

Remaining-work forecasts prefer all completed native video/query scopes over the original short preflight. The early completed videos are a small, size-prioritized sample, so these are provisional cost extrapolations, not completion-date guarantees.
sportsmot forecast: all completed native video/query caches, 974 measured frames.
animaltrack forecast: all completed native video/query caches, 1081 measured frames.
refer_kitti_v2 forecast: all completed native video/query caches, 836 measured frames.
tao_amodal forecast: all completed native video/query caches, 45 measured frames.

Workers: four shared A800 GPUs, physical 3/4/6/1, UUIDs and memory limits in the committed configuration. A service runs one native request at a time. Allocation time and sampled utilization are recorded separately. No fourfold speedup is assumed.

Association replay FPS is separate from detector throughput. A sum of measured stage costs is an estimate, never a directly measured end-to-end FPS.
