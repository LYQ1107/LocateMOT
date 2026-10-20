# GroundJEV G2 Clean live progress — step 13158

- **Run:** `/data3/user/LocateMOT-runs/GroundAnythingG2/G2_CLEAN_FULL_COVERAGE_V2/formal_clean_coverage_v2_final`
- **Observed:** 2026-10-21 06:56:12 +08:00
- **Progress:** 13,158 / 18,227 optimizer steps (72.18%)
- **Latest event:** loss `0.1489639424`, gradient norm `10.2172279`, peak event memory `12.828881 GiB`; all recorded numeric fields are finite.
- **Throughput:** 5.222 / 5.204 / 5.201 steps per minute over the last 100 / 500 / 1,000 steps. Linear remaining-time estimate: 16.18–16.24 hours, roughly 23:08–23:10 Asia/Shanghai on 2026-10-21. This is an operational estimate only.
- **Runtime:** parent PID `1799359`, ranks `1799468` and `1799469`, two-rank run bound to physical GPUs 3 and 5.
- **Recovery:** `adapter-step013000.pt` SHA-256 `08cf28930eced61b2086785494728221e113ac75cd7dc6aefe02ab1bc2ebfe59`; `latest_training_state.pt` SHA-256 `4a79cbf3e64e69ffc56aaf964754d0209382b87c2124d46f13506325ca52fd22`. The latest durable state is step 13,000 and contains the registered recovery contract.
- **Gate:** `final_receipt.json` is absent. G1 GPU5 adaptation, G2 bundle export, native TAO/OVMOT joining, three-task OneJev optimization and independent held-out evaluation remain dependency-gated. GPU7 held-out/YesTrack also remains resource-gated by an unrelated external process.
- **Scientific boundary:** no G2 MOT/RMOT/OVMOT metric or generalization claim is made from this training-health snapshot. Historical G0 and G2-PL artifacts are preserved separately.
