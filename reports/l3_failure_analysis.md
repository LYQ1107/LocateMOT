# Stage L3 — Failure Analysis

日期：2026-08-10。

## 1. 失败结论

```text
L3_REGIME_NOT_SUPPORTED（pilot）
REGIME_ROUTER_DATASET_SHORTCUT
```

Stage L3 主方法（latent tracking regime 条件化 shared core）在
四域 AC pilot 上未通过 Gate A：U1 相对 U0 只在 MOT20 微正
（+0.08pp），DanceTrack −1.19pp、MOT17 −1.91pp、BDD −0.89pp；
macro AssA −0.98pp。

## 2. 证据链

1. Regime 信号审计（`reports/l3_regime_signal.md`）：
   per-regime 方法偏好确实变化（MOT17 高运动桶 C1 胜 +6.7pp、
   BDD 高密度桶 C0 胜、低密度桶 EGRA 胜），但幅度小、样本少；
2. U0（naive shared learned）已超 L1DK：macro AssA 0.4013 vs 0.3944
   （MOT17 +1.67pp、MOT20 +1.72pp、BDD −0.70pp、DanceTrack +0.04pp）；
3. U1（regime 条件化）反而低于 U0：macro 0.3915；
4. Routing shortcut 审计：z_regime 的 domain classifier 准确率 96.6%，
   域质心距离远大于域内标准差 → z 主要编码 dataset 身份；
5. B（spec/prompt）未接入训练：因为 A 的 Gate 未通过，且 SAM3/GLEE
   已覆盖 prompt 接口统一（`reports/l3_novelty_collision_audit.md`）。

## 3. 根因

### 3.1 regime 特征与 dataset 天然共线

BDD 5fps → gap 大；DanceTrack/MOT17/MOT20 → gap 1。density/PBD 歧义
也与 benchmark 强相关。用这些统计学 z，FiLM 必然学到 dataset 偏置。

### 3.2 association 任务的可条件化空间小

L2 oracle 已证明 L1DK 的整视频 AC headroom < 0.1pp；U0 又吸收了
多域数据的大部分可学信号。剩下的 regime 差异（MOT17 高运动桶）幅度
大但窗口样本极少（30 窗口），不足以驱动端到端训练。

### 3.3 训练目标仍是 local correctness

U0/U1 都用 row/col CE（local）。L2 已证明 local 与 trajectory utility
不同构；因此 regime 条件化即使学到了，也不能保证改善 TrackEval。

## 4. 为什么不做的事

- 不换 MoE / 24 层 / 500M（任务书禁止失败后堆容量）；
- 不加入 dataset 平衡重采样后重试 U1（shortcut 已确认，重采样不能
  消除 regime 特征与 dataset 共线）；
- 不把 dataset-specific adapter 作为主结果；
- 不把 prompt 接口硬塞进已失败的核心。

## 5. 可保留的科学产出

1. **U0 是有效的 shared learned baseline**：一个 checkpoint 在
   DanceTrack/MOT17/MOT20/BDD（多类 GT）上达到/超过 L1DK
   （macro AssA +0.69pp），且无 dataset-specific 参数；
2. 负迁移审计：naive shared 的负迁移小（per-domain 最优差 ≤1pp），
   说明 L3 的“latent regime 必要性”在当前 AC 协议下证据不足；
3. 多类 BDD 协议：现有 manifest 已含 11 类 GT，可直接多类评估；
4. 2025/2026 审计：Claim 3（latent regime）未见等价实现，但 pilot
   证明其在本协议下无正收益——novelty 空洞不等于方法有效。

## 6. 下一步建议（单一）

若继续 Unified MOT 主线，应先回答“U0 的 BDD −0.7pp 与 DanceTrack
IDSW +30 的来源”，用 **类别/密度感知的 spec-conditioned U0**
（B 轴，不依赖 latent regime）验证统一接口；否则回到 U0 作为
统一 checkpoint 的工程收口（full tracker + LODO 基线）。
