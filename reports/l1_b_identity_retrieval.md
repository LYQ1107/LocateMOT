# Stage L1-B Pilot Identity Retrieval（Raw ObjectToken Baselines）

数据：pilot cache 1,860 帧（DanceTrack 360 / MOT17 240 / MOT20 160 /
TAO-Amodal 240 / YT-VOS 360 / MOSE 500），LocateAnything-3B 冻结，
commit 783f656d。candidate→GT 用逐帧最大 IoU 匹配。
协议：query=identity 首个观测；gallery=同 identity 其它观测（正）+ 同视频
其它 identity（hard）+ 跨视频其它 identity（easy）。cosine 排序。

## Same-Category R@1 / mAP（核心表）

| dataset | R0 PBD-box-end | R1 PBD-coord | R2 region | R3 fused(未训练) |
|---|---:|---:|---:|---:|
| dancetrack | 0.919 / 0.942 | 0.855 / 0.935 | 0.016 / 0.810 | 0.613 / 0.829 |
| mot17 | 0.946 / 0.926 | 0.970 / 0.945 | 0.151 / 0.820 | 0.825 / 0.763 |
| mot20 | 0.869 / 0.801 | 0.774 / 0.807 | 0.384 / 0.784 | 0.495 / 0.685 |
| tao_amodal | 0.870 / 0.841 | 0.870 / 0.869 | 0.609 / 0.833 | 0.739 / 0.748 |
| ytvos | 0.750 / 0.895 | 0.591 / 0.869 | 0.250 / 0.832 | 0.568 / 0.815 |
| mose | 0.551 / 0.835 | 0.435 / 0.810 | 0.087 / 0.755 | 0.377 / 0.740 |

## R4 IdentityToken（pilot-trained adapter）

同一 query/gallery 协议，adapter 训练 seed=20260806，704 个身份，20–30 epochs。

| dataset | R4 full R@1 / mAP | R4 pbd-only R@1 / mAP | best raw R@1 |
|---|---:|---:|---:|
| dancetrack | 0.903 / 0.941 | 0.968 / 0.938 | 0.919 (R0) |
| mot17 | 0.934 / 0.927 | 0.910 / 0.912 | 0.970 (R1) |
| mot20 | 0.862 / 0.794 | 0.845 / 0.812 | 0.869 (R0) |
| tao_amodal | 0.826 / 0.878 | 0.826 / 0.863 | 0.870 (R0/R1) |
| ytvos | 0.705 / 0.873 | 0.659 / 0.881 | 0.750 (R0) |
| mose | 0.493 / 0.814 | 0.551 / 0.815 | 0.551 (R0) |

## v2（加入 BDD100K，1,064 身份，per-dataset cap 120，30 epochs）

| dataset | best raw R@1 | R4 full R@1 | R4 pbd R@1 |
|---|---:|---:|---:|
| dancetrack | 0.919 | 0.984 (+6.5pp) | 0.935 |
| mot17 | 0.970 | 0.934 | 0.934 |
| mot20 | 0.869 | 0.822 | 0.801 |
| bdd100k | 0.740 | 0.770 (+3.0pp) | 0.640 |
| tao_amodal | 0.870 | 0.957 (+8.7pp) | 0.783 |
| ytvos | 0.747 | 0.734 | 0.722 |
| mose | 0.588 | 0.571 | 0.580 |

macro best-raw 0.815 → R4 full 0.825（+1.0pp）。数据增加后 adapter 对
road multi-class（BDD/TAO）与 DanceTrack 有效，但 dense person
（MOT17/MOT20）与 deformable（YT-VOS/MOSE）仍不提升。跨 family 方向
仍不一致，pilot gate 未通过。

完整 ROC-AUC / PR-AUC 见
`outputs/l1_b/same_category_retrieval.csv` 与
`outputs/l1_b/raw_token_retrieval.csv`。

## 结论

1. R0/R1（PBD box-end / coordinate）是当前最强 raw 特征；R2（region）单独
   不具备 top-1 判别力；R3（未训练 fused projection）反而弱于 R0。
2. dense person（DanceTrack/MOT17/MOT20）raw 基线已经很高（R@1 0.87–0.97），
   提升空间有限；TAO/YT-VOS/MOSE 是 Identity Adapter 的主要目标。
3. 这为 R4（IdentityToken）提供公平同协议 baseline。
4. R4 pilot 结果：full adapter 在全部数据集上不优于 best raw；pbd-only
   只在 DanceTrack 上 +4.9pp（0.919→0.968），其余数据集不提升或下降。
   未达到“至少三个 dataset family 方向一致且 +5pp”的 pilot gate。

## Pilot Decision

`L1_B1_IDENTITY_SIGNAL_NOT_SUPPORTED`（pilot，v1 与 v2 均未通过）——见
`l1_b_failure_analysis.md` 与 `STAGE_L1_B_FINAL_REPORT.md`。

## 资源

- Cache：/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L1B/cache_dla
- 脚本：tools/eval_l1b_retrieval.py、tools/repair_l1b_cache_gt.py
