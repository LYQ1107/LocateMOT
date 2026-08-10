# Stage L5 — GPT Handoff

日期：2026-08-11。

## 一句话结论

Route A（GT-anchored temporal identity transformer）把 BDD/Dance 的
在线 cross-spec identity drift 分别降低 46%/23%，但 BDD IDSW +12.3%、
MOT17/MOT20 AssA 下降，未通过 full-scale 判据；
Route B（sequence-local ID prediction）小集不迁移，证伪。

## 关键文件

- 最终报告：`reports/STAGE_L5_FINAL_REPORT.md`
- Route A：`reports/l5_route_a.md`、`docs/l5_temporal_identity_design.md`
- 数据：`docs/l5_clip_dataset.md`、`outputs/l5/clips/*.pkl`
- 文献：`docs/l5_reference_audit.md`、`reports/l5_novelty_collision_audit.md`
- 失败分析：`reports/l5_failure_analysis.md`
- 研究日志：`research_log.md`；状态机：`outputs/l5/state.json`

## 重要 bug 与修复（防止重蹈）

1. L4 TrackEval 数据目录复用旧 U0 → per-tag split 修复；
2. torch 2.5 的 3D attn_mask + -inf padding 产生 NaN → causal 2D mask +
   zero padding；
3. collate padding 参与 CE/argmax → masked_fill；
4. u0 target 用 history 多数 GT 被早期 switch 污染 → track_cur_gt
   （当前帧 GT 框 IoU 锚定）；
5. drift 指标错误（per-frame 对齐 vs video-level 全局对齐）→ 已统一为
   全局 Hungarian 对齐。

## 下一步唯一建议

把 Route A 的 temporal state 与 trajectory-level 在线一致性损失结合
（模型自身 rollout 的 track-chain 跨 spec 对齐，Gumbel-Sinkhorn 近似），
在完整 BDD/Dance/MOT17/MOT20 上 2–4 GPU 训练；这是唯一有正机制证据
且未被证伪的路线。
