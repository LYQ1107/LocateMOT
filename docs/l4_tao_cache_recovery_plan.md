# Stage L4 — TAO Cache Recovery Plan

日期：2026-08-10。

## 1. 现状

- Manifest：`outputs/l1_c/fixed_candidate_manifest/tao_amodal_train.jsonl`
  （105 videos / 4200 frames；2256 帧有候选，1944 帧无候选）。
- 缓存：`/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L1B/
  cache_dla/tao_amodal/train/<SOURCE>/<video_id>/<frame>/pilot.*`
  （source ∈ {BDD, AVA, YFCC100M, HACS, LaSOT}）。
- 4200 个 `.complete` 全部存在；safetensors/meta 同目录。
- 根因：manifest 的读取 key 是
  `tao_amodal/<video_id>/<frame>/pilot`，而真实路径多了一层
  `train/<SOURCE>/`。缓存本身没有损坏。

## 2. 恢复方案（已执行）

**不修改共享缓存、不重跑 LocateAnything、不复制数据。**

1. 新增 `tools/fix_tao_manifest.py`：对每个 video_id 在
   `cache_dla/tao_amodal/train/<SOURCE>/` 下定位一次来源，写入
   `outputs/l4/manifests/tao_amodal_train_l4.jsonl`，每行增加
   `cache_key = tao_amodal/train/<SOURCE>/<video_id>/<frame>/pilot`；
2. `build_candidates`（`tools/l4_restriction_audit.py`、
   `tools/eval_l3.py`）优先使用 `entry["cache_key"]`；
3. 验证：U0 audit 在 TAO ALL 上成功运行 105 视频，
   pairs=7,522（公共候选观测），ALL vs ALL agree=1.0。

## 3. 结果（frozen U0，PRIVILEGED_SPEC_ORACLE）

| Spec | Pairs | drift | P0 AssA | P1 AssA | P0 IDSW | P1 IDSW |
|---|---:|---:|---:|---:|---:|---:|
| ALL | 7,522 | 0.0000 | 0.4174 | 0.4174 | 870 | 870 |
| baby | 578 | 0.0675 | 0.2501 | 0.2625 | 83 | 69 |
| car_(automobile) | 798 | 0.2406 | 0.0963 | 0.1315 | 164 | 70 |
| dog | 217 | 0.1152 | 0.0225 | 0.0285 | 32 | 25 |
| cat | 168 | 0.0119 | 0.0431 | 0.0436 | 5 | 2 |
| inst:auto | 2,230 | 0.1430 | 0.4044 | 0.4616 | 359 | 217 |

结论：TAO（open-world long-tail）同样表现出 restriction 敏感性
（car 24%、instance 14%）与 P1 改善；作为第三域证据加入
Problem Signal。

## 4. 遗留

- 无候选帧（1944/4200）在 AC 协议下只能作为空帧；open-world
  detection 不是本项目 AC 范围；
- 若后续做 TETA/TAO 官方协议，需要额外 detection 输出与
  TrackEval TETA 适配（不在 Stage L4 主 AC 范围内）。
