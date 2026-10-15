# Resource cleanup — 2026-10-13

Status: `STOPPED_BY_USER_RESEARCH_PIVOT` for the old U2 route.

Verified `/proc` command lines, ancestry, cwd and start ticks before signaling.
Stopped 13 live owners/children with SIGTERM: supervisor 1313648, queues
1313650–1313653, cache parent 3099009 and workers 3099205–3099210/3099212.
Worker 6 had already exited. No SIGKILL was required. Post-stop old U2 process
count is zero. Other GPU jobs were preserved; several belong to UDMT.
No U2 cron entry or user systemd service was found.

Deleted 4,748 verified `frames_*.pt` feature shards under the exact epoch-1
cache root. Their allocated size was 42,831,618,048 bytes (39.89 GiB).
No process had an open reference. No symlink or cross-mount deletion occurred.
Manifests, protocols and worker receipts remain, but their feature paths now
refer to intentionally removed intermediate data.

| Volume | Used before (bytes) | Used after (bytes) | Net freed (bytes) |
|---|---:|---:|---:|
| /data0 | 3450871926784 | 3450871926784 | 0 |
| /data1 | 3342249783296 | 3342249783296 | 0 |
| /data2 | 3391303626752 | 3348498751488 | 42804875264 |
| /data3 | 3459036491776 | 3459039395840 | -2904064 |

The /data3 increase and small difference between allocated deleted bytes and
/data2 net change reflect concurrent unrelated activity. None of the four
external `LocateMOT_U2_perception_cache` roots existed at audit time.

Old U2 workers held 17,584 MiB in the compute-app snapshot (seven workers).
Those PIDs are absent after stopping. Total device memory still fluctuates
because other projects run concurrently; total GPU utilization is not the
old U2 utilization.

Preserved selected checkpoint:
`/data2/user/LocateMOT/outputs/unified/u2/perception_adaptation/training/epoch_1.pth`
(1,909,880,505 bytes), SHA256
`1d9077ae603f40de0e955a11a88083f914f98faa967bc46cdd06bcdfb5129688`.
CPU loading succeeded with 1,069 state-dict entries. Original weights and all
datasets/GT/evaluation results remain. Dataset roots measured: project data
143 GiB; TAO volumes approximately 45/45/89/179 GiB (symlinks may overlap;
these are not additive totals).

Removed feature shards are not retained in trash; they are regenerable from
the preserved images and foundation. Do not resume their old manifests.
Generated evidence is in `outputs/groundjev/resource_cleanup/`.

2026-10-15 evidence-scope clarification: the durable per-device snapshots in
`process_before.json` and `process_stop.json` sum to14042MiB net memory decrease
across all eight devices, with concurrent unrelated activity (device4increased).
This is directly recomputable from the saved snapshots; it is not an exclusive
causal U2 VRAM total. The17584MiB worker compute-app inventory above is a
historical observed report, not independently reconstructable from those two
per-device tables. Both scopes are retained; no old job is restarted to measure
past cleanup. The checkpoint still exists at1909880505bytes.
