# Non-MOT20 frozen data scope

Output root: `/data3/user/LocateMOT-runs/GroundJEV_Eval_NoMOT20_4GPU/request_local_seed_v2`. Model: G2 final step18227 `7c4a81349e333a304e32f42cf1999db289802d8c31eba415b85827038d14801f`.

| Dataset | Videos | Jobs/queries | Query-frames | DLM requests |
|---|---:|---:|---:|---:|
| sportsmot | 45 | 45 | 26970 | 26970 |
| animaltrack | 26 | 26 | 13167 | 13167 |
| refer_kitti_v2 | 3 | 1694 | 854073 | 854073 |
| tao_amodal | 993 | 993 | 36562 | 475306 |

TAO uses all 1,203 fixed LVIS-v1 labels in 13 blocks. Only annotated frames are inferred; no complete physical-video FPS claim.

Every job has a frozen detector manifest and a separate evaluation-only GT manifest. No MOT20 job is registered.

MOT17 labeled videos and all 25 local DanceTrack val videos overlap G2 TRAIN, including RMOT26 cross-dataset copies. They are excluded from independent evaluation. AerialMind available videos all occur in G2 dense TRAIN. RMOT26 local test files do not supply a complete, disjoint frame/identity-GT release. These are explicit coverage limits.

The full row audit and immutable manifests are stored locally as `train_overlap_audit.json`, `jobs.json`, and `jobs/*/manifest.json`; no labels or large caches are published.
