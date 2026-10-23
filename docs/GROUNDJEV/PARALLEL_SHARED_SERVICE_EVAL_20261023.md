# GroundJEV shared-service parallel evaluation (2026-10-23)

This run adds two independent validations while the long MOT20 controls continue. Each client reuses the already-running service on its GPU, so no additional SGLang scheduler or model service is created.

| owner | model | task and legal scope | GPU / endpoint | output | status |
|---|---|---|---|---|---|
| `g1_rmot_shared_gpu5_v2` | `GroundAnything-DLM-TrackProjG1-GPU5-V2` | RMOT single-query, `rmot_single_empty_full.json`, 800 validation frames, `training_on_this_scope=false` | GPU5 / `127.0.0.1:8240` (shared with G1 MOT20) | `/data2/user/LocateMOT_GroundJEV/outputs/groundjev/heldout/g1_rmot_shared_gpu5_v2` | running |
| `g200_ovmot_shared_gpu7_v2` | `GroundAnything-DLM-TrackProj200` | TAO video 63 Base/Novel OVMOT, `ovmot_tao_base_novel63_full.json`, 40 registered annotation frames, `training_on_this_scope=false` | GPU7 / `127.0.0.1:8238` (shared with G200 MOT20) | `/data2/user/LocateMOT_GroundJEV/outputs/groundjev/heldout/g200_ovmot_shared_gpu7_v2` | running |

Both use `run_video.py`, the `simple` tracker, `tracking_dense` decoding, explicit official scoring parser policies, `empty_and_record` perception failures, and the matching official scorer. OVMOT uses the repaired parser and vocabulary chunks of 192. The launcher records model/manifest SHA256, the exact command, frame contract, scorer output and a terminal receipt. Partial rows are not metrics.

The launcher is [run_shared_service_scope.py](/data2/user/LocateMOT/scripts/groundjev/run_shared_service_scope.py). The two failed first background starts were non-experiment launch errors (one shell-session reaping and one typo in the model path); their empty launch directories were not used or promoted.

The single blocking waiter watches these two owners together with the existing G1/G200 MOT20 owners. OneJev remains gated on the registered held-out finalizer and the exclusive `(4,5,6,7)` GPU contract.
