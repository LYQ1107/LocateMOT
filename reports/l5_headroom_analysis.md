# Stage L5 — Headroom Analysis（GT-anchored cross-spec evidence selection）

日期：2026-08-11。

## 1. 目的

回答：如果允许模型在 ALL 与 restricted 证据之间选择更正确的关联，
identity consistency 的理论 headroom 有多大？

## 2. L4 已证实的 gap（官方 AC 协议）

见 `reports/l4_cross_spec_inconsistency.md` 与 `l4_restriction_audit`：

- DanceTrack instance：P0 (Track-All-Then-Filter) AssA 0.5592 / IDSW 799
  vs P1 (Pre-Filter) AssA 0.8406 / IDSW 72；
- BDD：car drift 32.9%、truck 39.7%、bus 42.6%、pedestrian 48.8%；
- DanceTrack：person drift 32.2%、instance drift 31.1%；
- TAO：car drift 24.1%、instance drift 14.3%。

## 3. L5 小集 u0 baseline drift（val，scorer=base）

`outputs/l5/drift_base_val.json`：

| 域 | common candidates | drift | drift rate |
|---|---:|---:|---:|
| BDD100K | 370 | 172 | 46.5% |
| DanceTrack | 2228 | 1517 | 68.1% |
| 合计 | 2598 | 1689 | 65.0% |

事件分类（ALL vs restricted 的 common candidate 级别）：

| Type | 含义 | 数量 |
|---|---|---:|
| 1 | P0 correct / P1 correct / same | 787 |
| 2 | P0 wrong / P1 correct | 1554 |
| 3 | P0 correct / P1 wrong | 2 |
| 4 | both wrong same way | 122 |
| 5 | both wrong differently | 133 |

## 4. Oracle 结论

- Type 2（1554/2598 = 59.8%）说明：如果模型能在 restricted 证据下
  选择 P1 的正确关联（而不是复制 P0），identity consistency 可以直接
  下降约 60 个百分点中的大部分；
- Type 4/5（255 个）是双视图都错的 hard case，需要 temporal state 提供
  单视图内无法获得的证据；
- Type 3 只有 2 个：ALL 视图几乎不会比 restricted 更正确，这与
  「restricted evidence 减少竞争噪声」的观察一致。

因此 GT-anchored + evidence-adaptive 路线存在明确、可量化的 headroom。

## 5. GT-anchored identity oracle（小集）

假设模型对每个 common candidate 都能选择「正确 GT 身份」的 track，
则 Type1/2/3（P0/P1 至少一个正确且可对齐）全部消除，只剩 Type4/5
（双视图同错/异错）无法由跨视图证据选择解决：

| 集合 | Type4+5 占比（oracle floor） |
|---|---:|
| val 小集 | (122+133)/2598 = 9.8% |
| train 小集 | (658+419)/8018 = 13.4% |

train/val baseline drift 分别为 44.9% / 65.0%，因此可学习的 headroom
约为 31–55 个百分点；temporal identity state 的目标是把双视图都错的
case 也进一步压缩（需要单视图内的时间证据）。
