# Stage L1-B Pilot Failure Analysis

## 状态

`L1_B1_IDENTITY_SIGNAL_NOT_SUPPORTED`（pilot）

## 失败现象

1. Identity Adapter（full：PBD+region+geometry+gen，InfoNCE）在全部 6 个
   数据集的 Same-Category R@1 上不优于最佳 raw 基线。
2. PBD-only 变体只在 DanceTrack 提升（0.919→0.968，+4.9pp）；MOT17 反而
   下降（0.970→0.910）；YT-VOS/MOSE/TAO 不提升。
3. 未达到 pilot gate：需要至少三个 dataset family 方向一致且同类别 R@1
   明显优于 raw（+5pp 以上）。
4. v2（加入 BDD，1,064 身份，30 epochs）：full adapter 在 BDD +3.0pp、
   TAO +8.7pp、DanceTrack +6.5pp；MOT17 -3.6pp、MOT20 -4.7pp、
   YT-VOS -1.3pp、MOSE -1.7pp；macro 仅 +1.0pp。跨 family 方向仍不一致。

## 原因判断（按证据排序）

1. **raw PBD 已含较强 instance 信息（dense person）**：DanceTrack/MOT17/
   MOT20 raw R@1 0.87–0.97，Adapter 几乎没有可提升空间；MOT17 R1 甚至
   0.97，接近该协议上限。
2. **region 特征稀释**：full adapter 全部数据集差于 pbd-only；region
   raw R@1 仅 0.02–0.61，主要编码类别语义而非实例。
3. **小样本过拟合/跨数据集迁移不足**：704 个身份、每 epoch 307 身份；
   adapter 学到的变换在训练过的 dense 分布上有效（DanceTrack +4.9pp），
   在 deformable/sparse（YT-VOS/MOSE）与 long-tail（TAO）上不迁移。
4. **pilot 覆盖不足**：TAO 23、YT-VOS 44 个可用身份，统计噪声大；不能
   排除全量数据下方向成立，但按规格必须先如实关闭 pilot。
5. v2 观察：扩大数据后 road multi-class（BDD/TAO）一致改善，说明 adapter
   在 multi-class 方向可能有效；dense person 接近 raw 上限且被干扰，
   deformable 仍无信号——数据规模不是唯一变量，结构与目标需重新设计。

## 修改与结果变化

- full→pbd-only：DanceTrack +6.5pp（0.903→0.968），YT-VOS -4.6pp。
- 结论：结构消融（full→pbd）与数据扩充（v1→v2）都不能扭转跨 family
  不一致；方向从“仅 DanceTrack”变为“road multi-class + DanceTrack”。

## 是否保留

- 保留 pilot 基建（cache、split、评估脚本、adapter 代码与 checkpoints），
  作为后续更大规模或改目标的复测基础。
- 不保留“Identity Adapter 已成功”的结论。
