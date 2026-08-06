# Stage L0-C：数据划分报告

日期：2026-08-07

## 冻结划分来源

- train：`GLEE_PMOT_stage1/stage_u_unified_tracker/unified_train_split.json`（6066 videos）
- calibration：`.../unified_calibration_split.json`（300 videos）
- held-out：`.../stage_d_full/identity_heldout_split.json`（1071 videos）

以上均为旧项目已冻结的实际 manifest；本项目未根据聊天内容重新伪造划分，只做分层子采样。

## 本项目实际使用的子集

| 划分 | 视频数 | YouTube-VOS | MOSE | identity 数* | hash |
|---|---:|---:|---:|---:|---|
| train | 400 | 200 | 200 | 778 | d007b496c97f4dea4714aad04f51f186f3577892e6fc09adc096a34f1f661862 |
| calibration | 80 | 40 | 40 | 157 | acac14f43d36a453975a8079c20dc9cccf49549fc627af5d00fa11923bacfdb1 |
| held-out | 150 | 75 | 75 | 325 | e52777f3909807dc290a5a75cc132e4727058d2cd697c30a3a97adb8e833eec4 |

*identity 数来自冻结 manifest 的 target_count 之和（视频级目标数）。

## 重叠检查

- train ∩ calibration = 0
- train ∩ held-out = 0
- calibration ∩ held-out = 0

选择 train 时显式排除了 calibration 与 held-out 视频，因此三者互斥。

## 文件存在性

- 检查 630 个视频的 JPEGImages 与 Annotations 目录：0 个缺失。
- cache 构建实际 job 数：3780（详见 cache 报告）。

## 说明

- 划分文件：`configs/data/l0_c_{train,calibration,heldout}_videos.json`
- 划分脚本：`tools/build_l0_c_pair_manifest.py --phase splits`
