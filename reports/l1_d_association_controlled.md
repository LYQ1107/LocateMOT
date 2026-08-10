# Stage L1-D Association-Controlled Results

日期：2026-08-10。协议：与 L1-C 相同——全部候选输出、只改 track ID、
DetA/boxes 一致；TrackEval official（MOTChallenge 2D box）。

## 1. DanceTrack val（25 视频，25,508 帧）

| method | HOTA | DetA | AssA | IDF1 | IDSW | Frag |
|---|---:|---:|---:|---:|---:|---:|
| C0 IoU | 0.6078 | 0.9473 | 0.3899 | 0.5291 | 3,554 | 5,283 |
| C1 Motion (OC-SORT) | 0.6301 | 0.9470 | 0.4193 | 0.5660 | 2,916 | 5,221 |
| C2 raw PBD | 0.3836 | 0.9466 | 0.1555 | 0.3188 | 15,616 | 5,621 |
| C3 IoU+PBD | 0.6103 | 0.9469 | 0.3934 | 0.5367 | 2,981 | 5,254 |
| C4 frozen B6 | 0.3827 | 0.9475 | 0.1546 | 0.3083 | 16,456 | 5,427 |
| UA (failed UAF) | 0.3548 | 0.9472 | 0.1329 | 0.2704 | 26,804 | 5,610 |
| **L1DK base** | **0.6280** | 0.9470 | **0.4165** | **0.5630** | **2,558** | 5,209 |
| L1DK_d03 (residual) | 0.6149 | 0.9466 | 0.3993 | 0.5503 | 2,579 | 5,217 |

结论：
1. L1DK base（Kalman IoU+PBD 融合）是当前最强 AC 基座：
   AssA 与 C1 相当（0.4165 vs 0.4193），IDSW 比 C1 低 12.3%
   （2,558 vs 2,916），比 C3 低 14.2%。
2. EGRA residual（L1DK_d03）在 calibration 上 +1.9pp AssA，但
   **在 val 不迁移**：AssA −1.7pp、IDF1 −1.3pp、IDSW +21。
3. raw PBD（0.1555）仍是全表最弱之一；L1D 相对 raw PBD 提升巨大，
   但相对自己的 base 没有正向增量。

## 2. DanceTrack calibration（8 视频，8,024 帧）

| method | AssA | IDF1 | IDSW |
|---|---:|---:|---:|
| L1DK base | 0.4241 | 0.5713 | 512 |
| L1DK_d03 | 0.4428 | 0.5925 | 532 |

注：calibration 上的 +1.9pp 是过拟合信号，val 上不成立。

