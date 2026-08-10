# Stage L4 — Specification / Restriction 任务定义

日期：2026-08-10。

## 1. Specification 是 set-restriction operator

给定视频的候选流

```
X = {x_{t,i}}   （候选 box + 特征流）
```

specification `s` 定义候选保留集合：

```
R_s(X) = {x_{t,i} ∈ X : x_{t,i} 满足 s}
```

本阶段已实现（PRIVILEGED_SPEC_ORACLE，训练/诊断用途）：

| spec | 定义 | 类别来源 |
|---|---|---|
| `ALL` | 保留全部候选 | 显式 ALL |
| `cat:<name>` | 保留 GT 匹配候选的类别为 name | BDD manifest 真实 11 类 canonical names；DanceTrack/MOT 单类 fallback `person` |
| `inst:<gid,...>` / `inst:auto` | 保留指定 GT 身份（auto=每视频最长 top-k 轨迹） | GT 身份（诊断 oracle） |

未实现（本阶段不伪造）：

- box / point / visual exemplar / referring：需要官方 prompt encoder 与
  真实 benchmark 数据；Stage L4-A 没有合法可复用实现，标注
  `NOT_EXECUTED`，不把 synthetic prompt 当作主结果。

## 2. 等价性定义

`≈` 只在 common objects 上比较，使用 permutation-invariant
co-identity agreement（见 `docs/l4_consistency_metric_audit.md`）：

```
T_theta(R_s(X)) ≈ R_s(T_theta(X))
```

语义等价的 `s_a ~ s_b`（指向同一对象集合）还要求：

```
T(R_sa(X)) ≈ T(R_sb(X))
```

## 3. Dataset / Spec 矩阵

| Domain | 候选来源 | 已运行 specs | 说明 |
|---|---|---|---|
| BDD100K train（200 视频，8001 帧，11 类 GT） | LocateAnything cache | ALL + 11 category | 主证据 |
| DanceTrack val（25 视频） | LocateAnything cache | ALL + person + inst:auto | 第二证据（instance） |
| DanceTrack calibration（8 视频） | 同 cache | ALL + inst:auto（训练 pairs） | 训练 |
| MOT17 / MOT20 train | 同 cache | ALL + inst:auto（训练 pairs） | 训练/评估 |

BDD category 用官方 11 类 canonical names（bicycle, bus, car, motorcycle,
other person, other vehicle, pedestrian, rider, trailer, train, truck）；
不捏造 supercategory 层级。

## 4. 禁止事项

- 主推理结果不允许用 GT membership 过滤候选（本阶段所有 category/
  instance 结果均标注 PRIVILEGED_SPEC_ORACLE）；
- 不声称 open-vocabulary / referring（无官方 split / benchmark）；
- 不把 P0（post-filter）包装成创新。
