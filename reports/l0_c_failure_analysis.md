# Stage L0-C：失败分析

## 1. B4 训练 NaN（已修复）

- Evidence：B4 loss 从 step 200 起为 NaN，权重含 NaN。
- Root cause：`nn.MultiheadAttention` 的 `key_padding_mask` 语义用反（True=忽略），导致全有效样本被全部 mask；叠加空样本全 mask 与 contrastive 对零向量 normalize。
- Fix：mask 取反、空样本补 dummy key、contrastive 改 cosine margin。
- Validation：修复后 loss 1.22→0.19，无 NaN。

## 2. 预计算慢

- Evidence：冷启动 PrecomputedPairSet 在磁盘争用时可长达 30–50 分钟。
- Root cause：每样本读取 2 个 safetensors，受共享服务器 IO 影响。
- Fix：预计算张量持久化到 `outputs/l0_c/precomputed/*.pt`，后续训练直接加载。

## 3. Candidate 瓶颈

- Evidence：generic recall@0.5=0.528，category_guided=0.862；held-out 中 693 个 reference 处于 candidate_missing。
- 影响：e2e accuracy 上限受候选覆盖限制；conditional 评估中 MOSE 仅 0.293。

## 4. 多目标竞争

- Evidence：5–8 target conditional acc 0.229。
- 方向：需要更强的 reference 间竞争建模或更多多目标训练数据。

## 5. NO_MATCH 长间隔

- Evidence：gap>64 时 NO_MATCH F1 仅 0.438（e2e 0.571）。
- 方向：长间隔目标容易丢失 reference 信息。

## 6. 评测脚本问题（已修复）

- conditional accuracy 曾输出计数而非比率（cond_total 漏加）；NO_MATCH 假阳性未统计。已修复并重跑。
