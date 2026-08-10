# Stage L1-B0 Multi-Dataset Identity Audit

统计来源：真实本地标注文件（脚本 `tools/l1_b_dataset_audit.py`），
完整数字见 `outputs/l1_b/dataset_statistics.json`。

## 可用性总表

| dataset | status | family | 训练身份可用 |
|---|---|---|---|
| DanceTrack | AVAILABLE | same-class dense | 是（train/val；test 无公开 GT） |
| MOT17 | AVAILABLE | same-class dense | 是（train；test 无公开 GT） |
| MOT20 | AVAILABLE | same-class dense | 是（mot20_train 4 序列） |
| YT-VOS 2019 | AVAILABLE | deformable/sparse | 是（train；valid 官方 eval） |
| MOSEv2 | AVAILABLE | deformable/sparse | 是（train；valid 标注隐藏） |
| C-TAO (COVTrack) | AVAILABLE（派生数据，许可证待核） | multi-class long-tail | 是（TAO 格式） |
| BDD100K | AVAILABLE（masa 本地 tracking labels + images） | road multi-class | 是（200 train + 200 val 视频有本地图像） |
| TAO official | MISSING_PUBLIC | multi-class long-tail | 否（本地未找到官方 TAO 标注） |
| MOTSynth | FORBIDDEN_BY_SPEC | - | 否 |

## 关键统计

### DanceTrack

| split | videos | frames | identities | track med | obj/frame |
|---|---:|---:|---:|---:|---:|
| train | 40 | 41,796 | 419 | 791 | 8.35 |
| val | 25 | 25,508 | 273 | 830 | 8.83 |
| test | 35 | 38,551 | N/A（无公开 GT） | - | - |

单类别 dense、全标注、track 连续。track_id 按视频复用，身份键必须为
(video_id, track_id)。

### MOT17 / MOT20

| dataset | split | videos | frames | identities | obj/frame max |
|---|---:|---:|---:|---:|---:|
| MOT17 | train | 7（去 3 detector 重复） | 5,316 | 546 | 52 |
| MOT20 | train | 4 | 8,931 | 2,215 | 220 |

person-only、dense、带 visibility/ignore 语义；MOT20 为高密度人群。

### YT-VOS 2019

| split | videos | frames | identities | median track | same-cat 多实例视频 |
|---|---:|---:|---:|---:|---:|
| train | 3,471 | 94,440 | 6,459 | 21 | 914 |
| valid | 507 | 13,694 | 1,063 | 20 | 112 |

65 类，sparse annotation（未标注≠不存在），mask 可用，实例可变形。

### MOSEv2

| split | videos | frames | identities | median track |
|---|---:|---:|---:|---:|
| train | 3,666 | 311,843 | 7,631 | 42 |
| valid | 433 | 66,526 | N/A（标注隐藏，仅首帧 mask） | - |

无类别标签（meta 只有 objects id 列表）；需 mask 读取确定身份。

### C-TAO（COVTrack，TAO 格式）

| file | videos | frames | identities | 类别(使用中) | obj/frame |
|---|---:|---:|---:|---:|---:|
| ctao_base.json | 500 | 490,210 | 2,632 | 208 | 3.1 |

带 box+mask、track_id、neg_category_ids / not_exhaustive_category_ids
（sparse 语义必须保留），license 需核（派生数据）。

## 标注语义结论

1. 未标注≠背景：YT-VOS/MOSE/TAO/C-TAO 都 sparse，训练必须用 supervision
   mask，禁止把缺标注帧当负样本。
2. DanceTrack/MOT17/MOT20 全标注 dense，可直接 identity supervision。
3. BDD100K tracking 标签在
   /data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/bdd/annotations/box_track_20
   与 images/track 本地可用：train 200 视频（有图像）/ 39,418 帧 /
   15,558 身份 / 11 类（car/bus/truck/pedestrian/rider 等）；train 标签
   共 1,400 视频，本地图像只有 200，pilot 用有图像的 200 个；val 200 个。
   identity supervision 可用。
4. TAO 官方标注本地缺失；C-TAO 是 TAO 格式的可用替代，但先核许可证。
5. MOTSynth 规格禁止，不进入任何统计/训练。

## Pilot 数据集选择建议（待 storage/标注核验后定）

- A same-class dense：DanceTrack + MOT20（+MOT17）
- B multi-class/long-tail：C-TAO（或官方 TAO 若可下载）
- C deformable/sparse：YT-VOS + MOSE

至少覆盖三种 family 即可启动 pilot，不强制全用。
