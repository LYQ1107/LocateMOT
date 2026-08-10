# Stage L5 — Cross-Spec Results

日期：2026-08-11。

## 在线 drift（video-level 全局 ID 对齐分歧率）

| 域 | U0 | Route A ep40 | 相对变化 |
|---|---:|---:|---:|
| BDD val | 53.2% | 28.7% | -46% |
| Dance val | 37.9% | 29.3% | -23% |

事件分类（ep40，与 U0 相同视频）：

- Type1（P0/P1 均正确且一致）主导；Type2（P0 wrong/P1 correct）
  在 U0 中为 1554/2598（val，旧 dom_GT 指标），cur_GT 指标下大部分
  case 转为 Type1；
- 剩余 drift 主要来自早期 association switch 造成的链分歧。

未执行 unseen-spec-type 泛化（box/point/visual），见 NOT_EXECUTED。
