# Two additional shared-service held-out controls — 2026-10-23

Two additional independent controls were launched in parallel after the user requested two more evaluations. Both reuse already-running GroundAnything services, so this adds no SGLang scheduler and does not start or stop another model server.

| owner | GPU/service | model | scope | frames | status at launch |
|---|---:|---|---|---:|---|
| `g1_dancetrack_shared_gpu5_v1` | GPU5 / `8240` | `GroundAnything-DLM-TrackProjG1-GPU5-V2` | DanceTrack `dancetrack0004` validation | 1,203 | launched |
| `g200_animaltrack_shared_gpu7_v1` | GPU7 / `8238` | `GroundAnything-DLM-TrackProj200` | AnimalTrack `chicken_1` test | 306 | launched |

Both use the independent manifests under `heldout_inputs/g2_domain_v1`, the same `run_video.py` tracking decoder, official scoring parser, and official scorer used by the corresponding held-out controls. Outputs are isolated at:

- `/data2/user/LocateMOT_GroundJEV/outputs/groundjev/heldout/g1_dancetrack_shared_gpu5_v1`
- `/data2/user/LocateMOT_GroundJEV/outputs/groundjev/heldout/g200_animaltrack_shared_gpu7_v1`

GPU6 was left untouched because another user's workload was active there. The runtime launch receipt is `/data2/user/LocateMOT_GroundJEV/outputs/groundjev/heldout/parallel_two_shared_launch_20261023.json`. Metrics will be reported only after each owner writes a terminal receipt and `official_metrics.json`.
