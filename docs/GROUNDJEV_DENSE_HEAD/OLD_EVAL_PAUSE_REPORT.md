# Old evaluation pause report

Status: **PAUSED_BY_USER_FOR_ARCHITECTURE_PIVOT**. User explicitly paused for the spatial dense decoder architecture pivot. This is not evaluation completion or research failure.

Old branch: `codex/g2-four-gpu-no-mot20-eval-20261026`; source `28567e6fd227809d88ca7fba5ad3eb66de3467f2`.
Run: `/data3/user/LocateMOT-runs/GroundJEV_Eval_NoMOT20_4GPU/request_local_seed_v2`.
Full immutable signal/ownership audit: `/data3/user/LocateMOT-runs/GroundJEV_Eval_NoMOT20_4GPU/request_local_seed_v2/architecture_pivot_pause_20261026/pause_audit.json` (SHA256 `80653066ce9d4ce65be12ff2b697cc9d3d21e65586363f8a726b71f52d30d3f6`).

Stopped 121 verified UID 1009 evaluation processes, across PGIDs 124831, 3123376, 3190354, 3303793, 3305851, 3959270, 4104808, 4148689. Recovery owner and its shell were terminated first; dispatcher parents were frozen to prevent claims; current exporter children received 35 seconds to commit atomic frame/chunk transactions; then verified service groups and descendants received TERM, with KILL for [124831, 129552, 4104808, 4105535]. No other user's process was signalled.

The live old code does not recognize the pause marker. Actual termination, not the marker alone, disabled recovery. No matching tmux/screen/cron/systemd launcher remained. Old worktree GPU services and queue owners exited. Other projects still occupy GPUs, including GPU3; they were preserved.

SQLite backup API snapshots: `/data3/user/LocateMOT-runs/GroundJEV_Eval_NoMOT20_4GPU/request_local_seed_v2/architecture_pivot_pause_20261026/queue_before.sqlite` and `/data3/user/LocateMOT-runs/GroundJEV_Eval_NoMOT20_4GPU/request_local_seed_v2/architecture_pivot_pause_20261026/queue_paused.sqlite`. Both integrity checks: ok. Before counts: {'COMPLETE': 244, 'PENDING': 29888, 'RUNNING': 3}; paused counts: {'COMPLETE': 246, 'PENDING': 29887, 'RUNNING': 2}. No queue states were edited. Confirmed interrupted RUNNING units and their last committed frames are in OLD_EVAL_PAUSE_RECEIPT.json; their detector databases also have consistent backups.

|Worker|COMPLETE queue units|Interrupted RUNNING units|
|---|---:|---:|
|A|56|0|
|B|102|1|
|C|49|0|
|D|39|1|

These are atomic queue units, not complete videos. Worker B's interrupted TAO job retained 7 committed frames / 95 chunk requests (last physical frame180); worker D's Animal dolphin job retained269 committed frames (last269). No incomplete unit was promoted to COMPLETE.

Preserved run storage: 0.8299 GiB logical, 0.8498 GiB allocated; detector/cache artifacts 0.0548 GiB logical. No models, caches, predictions, metrics, source manifests or raw data were deleted.

Measured owned GPU residency: 26.1192 GPU-hours over 6.823 recorded hours. Includes idle resident models, excludes foreign jobs, is not kernel-busy time; telemetry begins after initial launch, so this is recorded-window cost, not an invented full-lifetime total.

## Completed legal scopes at pause

|Dataset|Complete detector jobs / scored frames|G2 R50 / R75|Simple HOTA / IDF1 / IDSW|J1 HOTA / IDF1 / IDSW|Coupled HOTA / IDF1 / IDSW|
|---|---:|---:|---:|---:|---:|
|animaltrack|9 / 2620|52.9142 / 33.2097|27.7036 / 29.1223 / 3407|7.8750 / 2.9851 / 275|8.1743 / 6.0929 / 9035|
|refer_kitti_v2|9 / 1881|35.3438 / 26.9117|15.4574 / 14.3253 / 482|9.6712 / 7.1368 / 0|not supported/completed|
|sportsmot|9 / 2396|98.5634 / 87.3665|58.6357 / 55.6920 / 780|13.7220 / 6.7468 / 0|21.7830 / 20.7489 / 781|
|tao_amodal|9 / 156|0.0000 / 0.0000|not supported/completed|not supported/completed|not supported/completed|

All scores are only these completed video/query scopes. Refer-KITTI frames are expression-conditioned query-frames; TAO uses annotated frames. They do not constitute dataset-wide completion. Raw per-arm scopes, frame counts and TETA are preserved in the receipt. J1 and Coupled do not improve the valid Simple baseline in these completed scopes.

TAO retains malformed coordinate/label/separator output in raw per-chunk SQLite records. One rejected chunk causes whole-frame rejection under the frozen original protocol; proposals from accepted chunks were not silently rescued. Joint TETA on completed scope remains zero. This motivates the image-only parallel object decoder; no native language inference was rerun for this pause summary.

## Explicit restoration procedure (not executed)

1. Read `/data3/user/LocateMOT-runs/GroundJEV_Eval_NoMOT20_4GPU/request_local_seed_v2/architecture_pivot_pause_20261026/queue_paused.json` and per-job `.resume.json`, verify model/checkpoint/config identities and free GPU UUIDs; preserve both original SQLite snapshots.
2. Inspect the old worker resource receipts: this externally paused run may lack fresh terminal receipts. Write a separate restoration audit proving every old service PID/start-time is gone; do not falsify historical receipts. The recovery code requires a terminal receipt newer than `service_ready.json`.
3. Only after a future explicit user resume request, remove/archive `pause_requested.json`, verify frozen native decoding/seed/manifest and register legal currently available GPUs. Recover confirmed unfinished owners with `dispatch_4gpu.py --config ... --stage detection --recover-worker <owner>`; this existing code resets only FAILED/RUNNING rows of that owner and reuses committed frame/chunk records. Pending queue and completed units remain preserved. Worker A also had a service-restart/queue-demotion bug; inspect its already-complete video before resetting/replaying the unit.
4. Restart recovery/monitor launchers only after the explicit resume audit. Do not launch broad `--resume` concurrently with recover owners. This pivot does not perform any restoration.

Models preserved: G2 final step18227 and merged bundle, original J1 step1200, Coupled step180, all Simple/JEV predictions and TrackEval/TETA outputs. Large assets stay on the server; Git carries this receipt and reproducible indices.

## Remaining paused ranges

|Dataset|Registered jobs|Complete detection jobs|Unfinished registered jobs|Preserved detector frames (incl partial)|
|---|---:|---:|---:|---:|
|animaltrack|26|9|17|2889|
|refer_kitti_v2|1694|9|1685|1881|
|sportsmot|45|9|36|2396|
|tao_amodal|993|9|984|163|

The five uncommitted automatic Markdown reports from the old worktree are also archived with SHA256 in `architecture_pivot_pause_20261026/original_worktree_reports/` and `uncommitted_reports_manifest.json`. The old source branch remains unchanged; these archived reports are not confused with the newer sealed metric summary.

After the new detector pilot ended, FINAL_RESOURCE_RECEIPT.json reconfirmed zero old-worktree processes and unchanged queue counts (246 COMPLETE /29887 PENDING /2 interrupted RUNNING), with integrity `ok`. The old branch still points to28567e6f. No automatic recovery or queue mutation occurred during the pilot.
