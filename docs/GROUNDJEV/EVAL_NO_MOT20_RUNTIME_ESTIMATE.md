# Measured runtime and remaining detection cost

Status: INCOMPLETE. Estimates use actual input-specific native request times, including all TAO vocabulary blocks. They are provisional and exclude future JEV cost.

| Dataset | Cached frames / registered | Requests saved | Measured inputs | FPS | Remaining GPU-hours |
|---|---:|---:|---:|---:|---:|
| sportsmot | 715/26970 | 715 | 32 | 0.363 | 20.103 |
| animaltrack | 781/13167 | 781 | 32 | 0.225 | 15.285 |
| refer_kitti_v2 | 627/854073 | 627 | 32 | 0.950 | 249.504 |
| tao_amodal | 45/36562 | 596 | 30 | 0.004 | 2687.802 |

sportsmot: registered 32-input preflight; mean 2.757 s, p95 2.990 s.
animaltrack: registered 32-input preflight; mean 4.442 s, p95 5.361 s.
refer_kitti_v2: registered 32-input preflight; mean 1.052 s, p95 1.151 s.
tao_amodal: completed full-video caches; registered 32-input preflight unfinished; mean 264.975 s, p95 PENDING s.

Workers: four shared A800 GPUs, physical 3/4/6/1, UUIDs and memory limits in the committed configuration. A service runs one native request at a time. Allocation time and sampled utilization are recorded separately. No fourfold speedup is assumed.

Association replay FPS is separate from detector throughput. A sum of measured stage costs is an estimate, never a directly measured end-to-end FPS.
