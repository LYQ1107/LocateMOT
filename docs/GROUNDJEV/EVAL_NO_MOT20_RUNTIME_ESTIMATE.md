# Measured runtime and remaining detection cost

Status: INCOMPLETE. Estimates use actual input-specific native request times, including all TAO vocabulary blocks. They are provisional and exclude future JEV cost.

| Dataset | Cached frames / registered | Requests saved | Measured inputs | FPS | Remaining GPU-hours |
|---|---:|---:|---:|---:|---:|
| sportsmot | 32/26970 | 32 | 32 | 0.363 | 20.626 |
| animaltrack | 86/13167 | 86 | 32 | 0.225 | 16.142 |
| refer_kitti_v2 | 32/854073 | 32 | 32 | 0.950 | 249.678 |
| tao_amodal | 0/36562 | 4 | 0 | PENDING | PENDING |

Workers: four shared A800 GPUs, physical 3/4/6/1, UUIDs and memory limits in the committed configuration. A service runs one native request at a time. Allocation time and sampled utilization are recorded separately. No fourfold speedup is assumed.

Association replay FPS is separate from detector throughput. A sum of measured stage costs is an estimate, never a directly measured end-to-end FPS.
