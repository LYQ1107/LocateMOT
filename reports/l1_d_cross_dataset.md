# Stage L1-D Cross-Dataset Results

同一 checkpoint（L1DK_d03）与同一基座（L1DK base）在 4 个域上的
Association-Controlled 对比（TrackEval；BDD/MOT17/MOT20 的 GT 由
frozen manifest 生成，标注为 custom MOTChallenge 格式）。

| 域 | method | HOTA | AssA | IDF1 | IDSW |
|---|---|---:|---:|---:|---:|
| BDD100K（200 视频，8,001 帧，5fps） | L1DK base | 0.3878 | 0.3292 | 0.3167 | 12,149 |
| BDD100K | L1DK_d03 | 0.3603 | 0.2841 | 0.2889 | 11,151 |
| MOT17（3 视频，240 帧） | L1DK base | 0.6569 | 0.6010 | 0.5784 | 276 |
| MOT17 | L1DK_d03 | 0.6525 | 0.5922 | 0.5775 | 274 |
| MOT20（2 视频，160 帧） | L1DK base | 0.4864 | 0.2779 | 0.3232 | 3,736 |
| MOT20 | L1DK_d03 | 0.4937 | 0.2864 | 0.3916 | 2,408 |

## Macro（4 域等权）

| 指标 | base | L1D | 变化 |
|---|---:|---:|---:|
| AssA | 0.4062 | 0.3905 | −1.6pp |
| IDF1 | 0.4453 | 0.4521 | +0.7pp |
| IDSW relative（各域相对变化均值） | — | — | −10.9% |

## 解读

- 方向不一致：MOT20 大胜（IDSW −35.5%，IDF1 +6.8pp）；BDD/MOT17
  AssA 下降；DanceTrack val 全线下滑。
- 因此残差模型不满足“unified 方向一致”要求，不能作为 one-checkpoint
  统一模型；L1DK base 作为统一基座（macro AssA 0.4062）。
- LODO 未执行：pilot 未通过（residual < base on DanceTrack val），
  按任务书 §55 仅在 pilot 成功后执行 zero-finetune generalization。

