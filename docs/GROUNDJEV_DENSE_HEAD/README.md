# Frozen spatial decoder pilot

Four real GPU trials completed 1,000 optimizer steps each, followed by 8,370 internal DEV frames and 1,607 frozen external frames per model. The spatial G2 head reaches R50 28.209% Sports / 44.700% Animal versus native G2 98.697% / 67.582%, with more false positives. It should not replace native G2 or trigger full identity training. The internal DEV was in G2 TRAIN; external videos were used in prior research, not untouched final TEST.

The old full evaluation is explicitly paused. See [pause report](OLD_EVAL_PAUSE_REPORT.md). Do not run its dispatcher, recovery supervisor or monitor to reproduce this detector experiment.

Read [feature audit](SPATIAL_FEATURE_AUDIT.md), [dataset protocol](DATASET_PROTOCOL.md), [training and metrics](PILOT_TRAINING.md), and [research decision](NEXT_RESEARCH_DECISION.md). JSON receipts provide exact IDs, hashes, scopes and limitations. Weights, full frame indices, spatial caches and original media remain on server under `/data3/user/LocateMOT-runs/G2_Spatial_DenseHead_20261026/`.

[Complete numeric comparison](PILOT_RESULTS.json), [native misses recovered and coverage lost](DETECTION_ERROR_ATTRIBUTION.md), and [final GPU release / paused queue receipt](FINAL_RESOURCE_RECEIPT.json) preserve the measured outcomes. Four-model train+evaluation residency was 11.6844 GPU-hours; a conservative upper estimate including the full functional-audit reserve is 12.6844, below24. All pilot GPU contexts exited, old queue counts stayed unchanged, and unrelated GPU jobs were preserved.

![Actual training losses and sampled internal-DEV AP; DEV was seen in G2 TRAIN](PILOT_CURVES.png)

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
5. The additional spatial-attention pair uses `g0_spatial_prior_control.yaml` and `g2_spatial_prior_pilot.yaml` and a separate `spatial_prior_v2` output root. Run its owner with `run_matched_pilot.py --run-root /path/to/spatial_prior_v2 --shared-budget-root /path/to/original_run`; the shared budget covers both pairs. Its sigma was frozen on internal DEV before any external head result was read. The original global-query results are retained.
6. After all four receipts are complete, `attribute_detection_errors.py` counts native misses recovered and native coverage lost; then `summarize_pilot.py` validates matching initialization/sampling/steps/checkpoints and emits final comparisons and decisions. `plot_pilot.py` uses separately installed optional plot dependencies under the run directory. `audit_auxiliary_data.py` / `run_auxiliary_gpu_diagnostic.py --run-root /path/to/spatial_prior_v2` do not add TAO/Refer to optimization.

Related numerical checks: `python -m unittest discover -s tests/groundjev/dense_head -v`. Real GPU smoke receipts are the actual feature/gradient/restore evidence; CPU tests are not training or performance results.

The global-query pair has a frozen feature-cache limit of 6 GiB per arm; the spatial-attention pair has 4 GiB per arm. All four together stay within 20 GiB. No identity training or G2 unfreezing is part of this pilot. Optional SAM has no ready local checkpoint and is skipped.

The real GPU checkout test used the public Git tree recorded in `CLEAN_PUBLIC_GIT_SNAPSHOT_GPU_RECEIPT.json`, loaded an actual trained checkpoint and ran real Sports/Animal images. It also verified an online TRAIN detection-loss gradient into the projector without an optimizer update. All recorded devices are NVIDIA A800-SXM4-80GB; `GPU_HARDWARE_IDENTITY.json` preserves their UUIDs. Cached frozen features in the pilot do not constitute joint G2 fine-tuning.
