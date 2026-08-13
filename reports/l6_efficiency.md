# Stage L6 Efficiency Report

Status: COMPLETE.

| Item | Value |
|---|---:|
| Model | UIDM-Large |
| Total params | 15.0M |
| Trainable params | 15.0M |
| Peak VRAM / GPU | ~12.8 GB (batch 8, H=16) |
| GPUs | 3 × A40 40GB (DDP) |
| Training steps | 4,200 (epoch 18) |
| Training wall time | ~5.8 h |
| Training loss (final) | 0.74 |
| Inference: MOT17 3 videos | ~16 s (TrackerEval runner) |
| Inference FPS (proxy) | ~20–30 FPS on A40 (per-frame model + cache I/O) |

FLOPs proxy: per frame, set transformer over ≤124 tokens × 6 layers
(d=384, FFN 1536) ≈ 0.9 GFLOPs + PBD encoder (2048→384) ≈ 0.2 GFLOPs
per candidate; total ≈ 1–2 GFLOPs/frame at typical N=20–60.
