# Stage L7 GPT Handoff（自包含，可交网页版 GPT 继续）

## 项目与结论一句话

LocateMOT Stage L7 验证：不同 WHAT-TO-TRACK specification 可以共享
同一个 HOW-TO-TRACK 因果身份动力学核心。OVMOT 正信号已获得
（`L7_OVMOT_SUPPORTED`），RMOT 因数据阻塞未执行。

## 关键数字

OVMOT（TAO val，官方 TETA，L6 核心冻结 + 0.69M CLIP 投影器，
零 OVMOT 训练数据）：

| 分类前端 | All TETA | AssocA | ClsA | Base/Novel AssocA |
|---|---|---|---|---|
| Detic 标签 | 31.48 | 29.51 | 0.14 | 29.54 / 29.31 |
| CLIP 余弦 | 33.94 | 29.51 | 7.51 | 29.54 / 29.31 |
| GT oracle | 63.45 | 29.51 | 96.05 | 95.91 / 97.11 |

- 随机基线 AssocA ≈ 8.2；Base≈Novel 无差距是核心证据。
- ClsA 瓶颈在 frozen perception；替换分类前端不影响身份动力学。

closed-set：L6 PBD core 仍是普通 MOT 主结果（Macro AssA 0.4922）；
统一 CLIP checkpoint 四域回归 Macro AssA 0.4290（代价见报告第 23 节）。
Dance repair（cue reliability）一次 iteration 失败（Dance AssA
0.2522/IDSW 9251 vs L6 0.3248/5290），已冻结普通 MOT。

## 运行中 / 待办

1. 无持久记忆消融已完成：All AssocA 24.32（-5.19pp），Base 24.22 /
   Novel 25.10；已填入报告第 26 节。
2. 报告 `reports/STAGE_L7_FINAL_REPORT.md` 已收口（自包含）。
3. 当前无运行中进程（GPUs 已释放）。

## 收口结论

`L7_OVMOT_SUPPORTED / RMOT_NOT_EXECUTED`。主结果：冻结 L6 UIDM core
+ 0.69M CLIP 投影器在 TAO 官方 TETA 上 AssocA 29.5（随机基线 8.2），
Base≈Novel；替换分类前端不改 AssocA（WHAT/HOW 解耦）；stateless
-5.2pp。Dance repair 一次迭代失败已冻结普通 MOT。

## 阻塞（如实记录，不伪造）

- RMOT：官方 KITTI tracking 帧本地缺失、官方下载需登录；STORM 缺
  VidOR 帧。协议/接口设计已写（`docs/l7_rmot_protocol.md`）。
- joint OVMOT 训练：本地 Detic SwinB 推理 roi_align OOM（51.7GB），
  官方 openmmlab CDN 被网络策略阻断；TAO train dets 未生成。

## 重要路径

- 模型：`locatemot/models/l6_uidm.py`（UIDM、app_dim、cue mixture）
- 训练：`tools/train_l6_uidm.py`（--app-key/--freeze-core/--stateless）
- 数据：`outputs/l7/data/{tao_val,clip_closed,clip_eval}`
- 评估：`tools/eval_l7_ovmot.py`、`tools/eval_l7_closed_clip.py`、
  `references/l7/TETA/scripts/run_ovmot.py`
- checkpoint：`outputs/l7/checkpoints/ovmot_probe/latest.pt`
- 状态：`outputs/l7/state.json`、`research_log.md`

## 与已冻结方向的边界

不回：universal ReID、dataset router/MoE、future utility、retrospective
revision、residual correction、specification hard consistency、MOTSynth、
C-TAO、普通 MOT 刷分。继续方向：Unified MOT（OVMOT→RMOT→joint）。
