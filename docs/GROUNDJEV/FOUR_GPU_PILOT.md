# Four-GPU Pilot evidence — 2026-10-15

The12Pilot execution deliverables in user task§17 now have evidence at the
registered legal development scopes. This replaces the obsolete2026-10-14
pending note. It is not a successful unified-model promotion, real-time claim,
Novel-recognition result, complete-dataset evaluation or completion of the full
research goal. Follow-on research and existing jobs continue.

|User requirement|Observed evidence|
|---|---|
|1. Stop the old U2 cache route and queue|`resource_cleanup/process_stop.json`: verified owner/worker signals, remaining U2 processes0; no U2 restart|
|2. Record cleanup resources and preserve data/weights|4748regenerable shards/42831618048allocated bytes removed, selected epoch1checkpoint1909880505bytes still present; actual before/after device-memory snapshots and disk report|
|3. Enforce at most four physical project GPUs|Registered set4/5/6/7; explicit visibility wrappers and actual per-run hardware/resource records; GPU5handoff verifies owned groups gone before reuse|
|4. Run real Ground DLM on real frames|Fresh native Full/Compact800each, same1412actual detector candidates and36parser-rejected frames, all800retained|
|5. Load actual pretrained OneJev|Unresumed pretrainedFull1203run records original `models/OneJev-0.8B`; genuine native engine in `decision.py`|
|6. Dynamic identities, typed decisions and persistent memory|PretrainedFull real frame2already has historical numeric-ID options; native probabilities/crops/lifecycle/update and final serialized state present|
|7. Complete real MOT video|DanceTrack0004all1203frames, official TrackEval and rendered video; additional matched Compact/MLP/Set/U2 controls complete|
|8. Legal RMOT and OVMOT loops|Both RMOT800expressions complete; TAO40annotation frames/all1203input categories/official TETA/BaseNovel scopes complete|
|9. Real tracking fine-tuning and before/after|MOT500actual500updates/2000examples/latest state/checkpoint; pretrained/MOT500same-source800official metrics saved, including negative transfer; additional600stages separately recorded|
|10. Same-Ground association baseline|All12two800controls retain identical proposals; Simple/Full/Compact official metrics, no hidden candidate filtering|
|11. Actual native pipeline speed and resources|Fresh same-GPU/same800pair: Full.495914FPS/Compact.687099FPS; all IO/state/render/encoder/finalexport timed, actual5s ownedGPU memory samples|
|12. Publish code/configs/results and verify remote|Verified GitHub commit1a1bf1a2dacb8aceced7e70207629d0928dc18cd/treee638f60c3733ecb04ae600f5da3f41713fc2e413 already contains the actual paired artifacts; subsequent audit publication verified separately|

Paths in the table are under `outputs/groundjev/` unless explicitly shown as
code/model paths. Consolidated source receipts:

- `pilot_milestone_20261015/evidence_audit.json`
- `resource_cleanup/{process_before,process_stop,cache_cleanup_result}.json`
- `pilot_runs/dancetrack0004_pretrained_v3_evidence_fresh_v1/full/{result,official_metrics}.json`
- `multitask_runs/full_rmot_shared600_v1/all12_validated_summary.json`
- `compact/fixed_affinity_controls_v1/validated_summary.json`
- `e2e/full_compact_rmot800_v1/validated_summary.json`

The durable before/after device snapshots sum to a14042MiB net decrease over
all eight devices during concurrent unrelated activity. This is an actual
snapshot delta, not an exclusively attributable U2 VRAM figure. The17584MiB
owned-worker inventory reported at cleanup time is preserved as historical
information; it cannot be reconstructed from those per-device snapshots alone.
Disk allocated-deletion bytes and filesystem net changes also have distinct
scopes; details remain in `RESOURCE_CLEANUP.md`.

Evidence quality limits remain material: one-seed legal development videos,
single RMOT expression pair with18GTboxes/782empty frames, poor recall,
NovelClsA0, no model promotion. Full/Compact speed pair uses fixed arm order,
concurrent other workloads and five-second VRAM samples; startup/init excluded
and separately timed. Short scopes are never called full-dataset results.

Beyond Pilot already completed: matched300step association architectures,
25frozen policy units,24task-subset controls,12context units,12hybrid units,
12matched input-evidence units, all12full800RMOT model controls and the actual
fresh paired speed/resource measurement. Remaining required studies are trained
Full1203controls, original/adapted full9native retention, localization200/96native
frames/18composition controls and naturalMOT20-05full1657/8memory controls.
Their queues/results are tracked in `CURRENT_STATUS.md`; the full goal stays active.
