# Measured runtime and remaining detection cost

Status: INCOMPLETE. Estimates use actual input-specific native request times, including all TAO vocabulary blocks. They are provisional and exclude future JEV cost.

| Dataset | Cached frames / registered | Requests saved | Measured inputs | FPS | Remaining GPU-hours |
|---|---:|---:|---:|---:|---:|
| sportsmot | 974/26970 | 974 | 32 | 0.363 | 19.905 |
| animaltrack | 1081/13167 | 1081 | 32 | 0.225 | 14.914 |
| refer_kitti_v2 | 836/854073 | 836 | 32 | 0.950 | 249.442 |
| tao_amodal | 71/36562 | 946 | 30 | 0.004 | 2685.888 |

sportsmot: registered 32-input preflight; mean 2.757 s, p95 2.990 s.
animaltrack: registered 32-input preflight; mean 4.442 s, p95 5.361 s.
refer_kitti_v2: registered 32-input preflight; mean 1.052 s, p95 1.151 s.
tao_amodal: completed full-video caches; registered 32-input preflight unfinished; mean 264.975 s, p95 PENDING s.

Workers: four shared A800 GPUs, physical 3/4/6/1, UUIDs and memory limits in the committed configuration. A service runs one native request at a time. Allocation time and sampled utilization are recorded separately. No fourfold speedup is assumed.

Association replay FPS is separate from detector throughput. A sum of measured stage costs is an estimate, never a directly measured end-to-end FPS.
