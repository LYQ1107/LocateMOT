# Frozen spatial decoder pilot

The old full evaluation is explicitly paused. See [pause report](OLD_EVAL_PAUSE_REPORT.md). Do not run its dispatcher, recovery supervisor or monitor to reproduce this detector experiment.

Read [feature audit](SPATIAL_FEATURE_AUDIT.md), [dataset protocol](DATASET_PROTOCOL.md), [training and metrics](PILOT_TRAINING.md), and [research decision](NEXT_RESEARCH_DECISION.md). JSON receipts provide exact IDs, hashes, scopes and limitations. Weights, full frame indices, spatial caches and original media remain on server under `/data3/user/LocateMOT-runs/G2_Spatial_DenseHead_20261026/`.

## Checkout and environment

```bash
git clone --branch research/g2-parallel-dense-head-20261026 https://github.com/LYQ1107/LocateMOT.git
cd LocateMOT
python3 tools/groundjev/eval_no_mot20/link_local_assets.py --asset-root /data2/user/LocateMOT_GroundJEV
```

Use `/data2/user/LocateMOT_GroundJEV/external/GroundAnything/.venv-serve/bin/python` for native vision, decoder training and tests. It is separate from OneJev. GroundAnything dependency source is pinned to: `8a3b43a1b42952e117960e85d63ff173bb9fba0a`. Tested package pins are in `configs/groundjev/dense_head/requirements.txt`. No FlashAttention extension is required; the audited official vision uses eager attention. Existing model bundles include their custom processor/config/MoonViT implementation and trained projector. G2 final merged file SHA is verified by the real GPU leakage audit.

## Reproduction sequence

1. `prepare_train_data.py` creates or reuses the frozen video-disjoint head split. `prepare_external_comparison.py` reuses final-G2 native caches and freezes evaluation scope; neither downloads images or generates autoregressive boxes.
2. On an available registered GPU run `audit_feature_leakage.py --arm g2`, and likewise g0; then `compare_spatial_features.py`. Set task-local `CUDA_VISIBLE_DEVICES`; inspect the UUID and all other GroundJEV allocations first. At most four physical GPUs total.
3. The checked-in configs describe the completed1,000-step arms. **To run a new experiment, use new output directories/config copies and an audited GPU registration.** Existing completed receipts/checkpoints are preserved and fresh training refuses to overwrite them. The recorded `GPU_REGISTRATION.json` is historical resource evidence, not permission to clear someone else's current GPU processes.
4. `run_matched_pilot.py` owns both arms in one blocking command. It has no old-evaluation recovery path. Each arm trains, chooses checkpoint/threshold on TRAIN-internal DEV, evaluates20 full DEV videos and the frozen6-video external scope. Resume an unfinished arm using its actual config and `train_dense_head.py --resume /path/to/last.pt`; optimizer/scheduler/sampler/RNG are restored.
5. `summarize_pilot.py` validates matching initialization/sampling/steps/checkpoints and emits comparison receipts. `plot_pilot.py` uses separately installed optional plot dependencies under the run directory. `audit_auxiliary_data.py` / `run_auxiliary_gpu_diagnostic.py` do not add TAO/Refer to optimization.

Related numerical checks: `python -m unittest discover -s tests/groundjev/dense_head -v`. Real GPU smoke receipts are the actual feature/gradient/restore evidence; CPU tests are not training or performance results.

Frozen-backbone spatial feature cache is bounded6GiB per arm (12GiB combined), not a full candidate-cache export. No identity training or G2 unfreezing is part of this pilot. Optional SAM has no ready local checkpoint and is skipped.
