# Stage L1-C Association-Controlled Results（UAF 完成）

更新：2026-08-10。DanceTrack val（25 视频，25,508 帧，TrackEval official），
Association-Controlled protocol：所有候选必须输出，仅 track ID 可变；
DetA ≈ 0.947（C0–C4 全部一致，≤0.001 tie-break 差异）。

## 1. 基线主表

| variant | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | IDSW/1k det |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 (IoU) | 0.608 | 0.947 | 0.390 | 0.529 | 0.894 | 3,554 | 45.9 |
| C1 (motion/OC-SORT) | 0.630 | 0.947 | 0.419 | 0.566 | 0.897 | 2,916 | 37.6 |
| C2 (raw PBD cosine) | 0.384 | 0.947 | 0.155 | 0.319 | 0.836 | 15,616 | 201.4 |
| C3 (IoU+PBD fixed) | 0.587 | 0.947 | 0.364 | 0.529 | 0.896 | 2,978 | 38.4 |
| C4 (frozen B6) | 0.383 | 0.948 | 0.155 | 0.308 | 0.836 | 16,456 | 212.3 |
| UA (final, margin=3.5) | 0.355 | 0.947 | 0.133 | 0.270 | 0.787 | 26,804 | 345.7 |

注：AC 协议下 DetA 全部≈0.947（所有候选输出），因此 HOTA/AssA 差异完全来自
association。C2 的 raw PBD cosine 在 DanceTrack 上很差（IDSW 15.6k），
说明 PBD 特征必须结合 geometry/motion/set 竞争。

## 2. 完整性校验

- 25/25 val 视频的 C0/C1/C2/C3 输出 boxes 逐帧一致（sha 校验通过）。
- C4 输出 boxes 一致；DetA 0.9475。

## 3. UAF NEW-margin 校准（calibration split，8 视频）

| margin | HOTA | AssA | IDF1 | IDSW |
|---|---:|---:|---:|---:|
| 0.0 | 0.306 | 0.099 | 0.212 | 17,465 |
| 1.0 | 0.329 | 0.115 | 0.236 | 9,327 |
| 2.0 | 0.339 | 0.122 | 0.246 | 6,514 |
| 3.0 | 0.353 | 0.132 | 0.278 | 4,971 |
| 3.5 | 0.354 | 0.133 | 0.286 | 4,342 |
| 4.0 | 0.352 | 0.131 | 0.287 | 4,013 |
| 5.0 | 0.351 | 0.130 | 0.284 | 4,049 |

选择 margin=3.5（AssA 最高；与 4.0 的 IDF1/IDSW 差异 <0.002/<330）。

## 4. 结论（UAF pilot gate）

- UAF（frozen LocateAnything + UA decoder，7.9M 参数，50k 步）在
  association-controlled DanceTrack val 上 AssA 0.133 / IDF1 0.270 /
  IDSW 26,804，三项均未超过 C4（0.155 / 0.308 / 16,456）。
- 因此 §51 UAF pilot gate 不通过：
  “相对旧 B6 AssA/IDF1 明显提高、IDSW 明显下降”不满足。
- 科学结论：当前 contextual association 设计（short-history track token +
  relation-bias cross-attn + K+1）在该协议下未证明优于 IoU/motion/B6；
  不继续堆容量。Route B（LoRA）仍作为诊断路线执行，以回答
  “LoRA 是否改变 association 特征质量”。

## 5. 下一步

- C2/C3 阈值在 calibration 校准后更新（当前为默认值）。
- LoRA（UAL）完成后填 UAL 行。
