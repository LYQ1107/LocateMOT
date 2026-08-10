# Stage L2 GPT Handoff

日期：2026-08-10。项目：LocateMOT（不是 TrackOCD）。

## 一句话结论

Stage L2 验证了核心科学问题（local correctness 与 future trajectory
utility 不同构），但 **oracle headroom 不足**：即使拥有 privileged
future，端到端整视频 AssA 提升 < 0.1pp（DanceTrack）且 BDD/MOT17
为负；判定 `L2_ORACLE_HEADROOM_LOW`，未启动大型 TUM 训练，按任务书
直接进入失败分析与最终报告。

## 关键数字

### Baseline（四域 AC 公平矩阵，官方 TrackEval）

| Variant | DanceTrack AssA | MOT17 AssA | MOT20 AssA | BDD AssA | Macro |
|---|---:|---:|---:|---:|---:|
| C0 IoU | 0.3899 | 0.4504 | 0.2071 | 0.3044 | 0.3380 |
| C1 Motion | 0.4193 | 0.5530 | 0.2869 | 0.3019 | 0.3903 |
| C2 PBD | 0.1555 | 0.0975 | 0.0740 | 0.1659 | 0.1232 |
| C3 IoU+PBD | 0.3934 | 0.3856 | 0.2171 | 0.2255 | 0.3054 |
| **L1DK base** | **0.4165** | **0.6010** | 0.2779 | **0.3292** | **0.4062** |
| L1DK_d03 | 0.3992 | 0.5922 | **0.2864** | 0.2841 | 0.3905 |

BEST_STRONG_BASE = L1DK base（0.4 IoU + 0.2 PBD + 0.4 Kalman motion，
thr 0.25）。

### Oracle headroom

单事件窗口（冻结 base policy rollout）：

- DanceTrack val：H4 +0.38pp → H32 +0.74pp（mean windowed AssA），
  frac better 7.5%→21.9%（1,000 事件）；
- BDD：H2 +0.96pp → H16 +1.01pp，frac better 18%→62%（745 事件）。

端到端 privileged greedy oracle（整视频 TrackEval 同款 AssA）：

- dancetrack0004：+0.02pp；dancetrack0005：+0.06pp；
- BDD×3：均值 −0.88pp（+2.62/−4.76/−0.50）；
- MOT17-02-SDP：−2.32pp；
- IDSW 全部变差（DanceTrack 149→151、68→74；BDD 260→283；
  MOT17 783→866）。

### Local vs future mismatch

- DanceTrack H32：219/1000 事件 future-best ≠ base；
  local_correct_future_bad 128 / local_wrong_future_good 60；
- BDD H16：460/745 事件 future-best ≠ base；
  local_correct_future_bad 173 / local_wrong_future_good 110；
- 多 horizon 排序一致性：H16 vs H32 73.7%（DanceTrack）、
  H8 vs H16 67.7%（BDD）。

### 历史污染审计（EGRA 修正）

- DanceTrack val：1,295 修正，helpful 334（25.8%）/ harmful 264 /
  same_gt 213 / other 484；57% 位于 past IDSW≥4 轨迹且 helpful 率更低；
- BDD：1,799 修正，helpful 163（9.1%）/ harmful 247 / same_gt 748 /
  other 641；59.8% 位于 purity<0.5 轨迹，harmful 175 > helpful 74。

## 方法学可信度

- replay 与官方基线 100% 一致（MOT17-04-SDP 3589/3589）；
- 本项目 windowed AssA/IDF1 与官方 TrackEval 整视频数值完全一致
  （DanceTrack 0004/0005、MOT17-04 验证）；
- 文献审计（TDLP/SambaMOTR/TRACT/UniTrack/PathConsistency/QuoVadis/
  FDTA/HATReID-MOT/HNCD-MOTR 全部实际 clone 阅读）：
  **NO DIRECTLY EQUIVALENT VERIFIED METHOD FOUND**。

## 为什么没有训练 TUM

任务书明确停止条件：oracle 相对 BEST_STRONG_BASE 无 ~1pp AssA
headroom 且 IDSW 无改善 → `L2_ORACLE_HEADROOM_LOW`，不训练大模型。
本阶段 oracle 整视频 headroom < 0.1pp（DanceTrack），端到端 IDSW
变差，因此停止。

## 未完成 / 下一步建议

1. 若继续该方向：换效用定义（整序列 ID 映射 + IDSW 惩罚）或
   换协议（允许 ID 重映射的 full-tracker）后重新验证 oracle；
2. 否则回到 L1DK base 的 full-tracker 工程路线；
3. TAO cache 缺失，后续需补 cache 才能做 TAO 域；
4. 无阻塞问题；所有产物已写入仓库（见 STAGE_L2_FINAL_REPORT.md §64）。
