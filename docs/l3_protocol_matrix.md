# Stage L3 — 协议矩阵

日期：2026-08-10。

| Dataset | Regime | Classes | FPS/时间 | 标注密度 | Main metric | Prompt support | 是否可共同训练 | Train/Eval 角色 |
|---|---|---|---|---|---|---|---|---|
| DanceTrack | dense same-class，外观歧义高，非线性运动/交叉 | 1（person） | 30fps，帧间隔 1 | dense box+ID | HOTA/AssA/IDF1/IDSW（官方 TrackEval） | ALL / person text | ✅ | calib=train；val=主评估 |
| MOT17 | standard pedestrian | 1（person） | 30fps（采样） | dense box+ID | HOTA/AssA/IDF1/IDSW | ALL / person text | ✅ | train + 跨域评估 |
| MOT20 | extreme crowd/遮挡 | 1（person） | 25fps（采样） | dense box+ID | HOTA/AssA/IDF1/IDSW | ALL / person text | ✅ | train + 跨域评估 |
| BDD100K | multi-class driving，ego-motion，5fps | 11 类 | 5fps 采样 | dense box+ID（全类） | per-class AC + macro；官方 BDD eval（若可行） | ALL / category text / open-vocab | ✅ | train + 多类评估 |
| TAO / C-TAO | long-tail/open-world/sparse | 数百类（TAO categories） | 1fps 级别 | sparse federated | TETA / official TAO | ALL / category / open-vocab | 待 cache 修复 | 延迟纳入 |
| STORM-Bench | referring MOT（VidOR） | 80 类 | 1fps | referring expression→tracks | HOTA/IDF1（官方 bench） | referring text | 待数据下载 | 诊断级 |
| RMOT26 | query-driven MOT | — | — | grounded queries | MCP/MOTP（官方） | text query | 待数据下载 | 诊断级 |

## 统一训练采样原则

- dataset-balanced + video-balanced + category/long-tail-balanced +
  regime-balanced + prompt-type-balanced；
- loss 按任务/域归一化；
- 禁止 raw concat；
- 禁止 dataset ID 作为输入。

## AC 协议

- 所有主要对比使用 Association-Controlled：同一候选集
  （boxes/scores/features）、同帧数、同输出数量，只改 IDs；
- hash 校验候选集一致性；
- 主要指标 TrackEval（HOTA/AssA/IDF1/IDSW）。

## 输出

`outputs/l3/manifests/`（构建脚本生成）。
