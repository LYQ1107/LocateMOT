# Stage L1-A DanceTrack Candidate Report（LocateAnything-3B pilot）

生成时间：2026-08-07

Pilot 视频：dancetrack0052（低密度）、dancetrack0082（高密度）、dancetrack0096（高密度），每视频 30 帧。
协议：LocateAnything-3B（commit 783f656d）hybrid generation，max_new_tokens=1024，in_token_limit=4096，seed=20260806。

| query | Recall@0.3 | Recall@0.5 | Recall@0.7 | Precision | candidates/frame | duplicate rate | avg seconds/frame | FPS | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| d1: "Locate all the instances that matches the following description: person." | 0.9411 | 0.9166 | 0.7911 | 1.0356 | 21.88 | 0.0060 | 3.10 | 0.32 | 10.69 |
| d2: "... a person." | 0.6018 | 0.5370 | 0.4282 | 1.5721 | 14.41 | 0.0049 | 4.21 | 0.24 | 10.69 |
| d3: "... people." | 0.9411 | 0.9152 | 0.7764 | 1.0403 | 21.78 | 0.0050 | 3.05 | 0.33 | 10.69 |

## 结论

- d1 与 d3 的 Recall@0.5 均 >0.90，d2 明显更差（0.54）。
- 选择 **d1（"person."）** 作为固定 person query，在 calibration 上继续确认后冻结。
- Recall@0.5=0.9166 ≥ 0.70，进入全量缓存。
- 无小目标（<32×32）样本；高密度帧（GT≥10）Recall@0.5=0.9734，说明高密度不是主要瓶颈。
- 候选/帧 21.9 与 GT 密度接近；duplicate rate 0.6%。
- 推理速度 ~0.32 FPS/GPU，峰值显存 ~10.7GB（A100 40GB），适合多 GPU 并行缓存。
