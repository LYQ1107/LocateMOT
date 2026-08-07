# Stage L1-A DanceTrack Split Report

生成时间：2026-08-07，seed=20260806

划分规则：video-level disjoint；train/calibration 来自官方 train（40 视频），official val（25 视频）全程 held-out，不用于训练/阈值/checkpoint 选择。

## train: 32 videos, 33772 frames, 280094 GT boxes
| video_id | frames | gt_frames | gt_boxes | mean_density | max_density |
|---|---:|---:|---:|---:|---:|
| dancetrack0074 | 1203 | 1203 | 5304 | 4.41 | 5 |
| dancetrack0055 | 1203 | 1203 | 5820 | 4.84 | 5 |
| dancetrack0023 | 1483 | 1483 | 13025 | 8.78 | 9 |
| dancetrack0001 | 703 | 703 | 4912 | 6.99 | 7 |
| dancetrack0024 | 763 | 763 | 4512 | 5.91 | 6 |
| dancetrack0086 | 603 | 603 | 9496 | 15.75 | 16 |
| dancetrack0029 | 1263 | 1263 | 7886 | 6.24 | 7 |
| dancetrack0032 | 604 | 604 | 3360 | 5.56 | 6 |
| dancetrack0062 | 1203 | 1203 | 6227 | 5.18 | 6 |
| dancetrack0044 | 1203 | 1203 | 13389 | 11.13 | 13 |
| dancetrack0039 | 1242 | 1242 | 6092 | 4.91 | 5 |
| dancetrack0016 | 2163 | 2163 | 12613 | 5.83 | 6 |
| dancetrack0012 | 1203 | 1203 | 13032 | 10.83 | 12 |
| dancetrack0075 | 803 | 803 | 5442 | 6.78 | 7 |
| dancetrack0068 | 1203 | 1203 | 6002 | 4.99 | 5 |
| dancetrack0096 | 603 | 603 | 15787 | 26.18 | 40 |
| dancetrack0053 | 1204 | 1204 | 5954 | 4.95 | 5 |
| dancetrack0052 | 1203 | 1203 | 4462 | 3.71 | 4 |
| dancetrack0057 | 622 | 622 | 3004 | 4.83 | 6 |
| dancetrack0045 | 1203 | 1203 | 15830 | 13.16 | 14 |
| dancetrack0082 | 603 | 603 | 12370 | 20.51 | 24 |
| dancetrack0087 | 1003 | 1003 | 10699 | 10.67 | 11 |
| dancetrack0008 | 883 | 883 | 6850 | 7.76 | 8 |
| dancetrack0006 | 1202 | 1202 | 10559 | 8.78 | 9 |
| dancetrack0051 | 1203 | 1203 | 10808 | 8.98 | 9 |
| dancetrack0069 | 1403 | 1403 | 8259 | 5.89 | 6 |
| dancetrack0066 | 1202 | 1202 | 6010 | 5.00 | 5 |
| dancetrack0080 | 1201 | 1201 | 12040 | 10.03 | 16 |
| dancetrack0098 | 1203 | 1203 | 8426 | 7.00 | 8 |
| dancetrack0037 | 1203 | 1203 | 8305 | 6.90 | 7 |
| dancetrack0020 | 583 | 583 | 20174 | 34.60 | 40 |
| dancetrack0027 | 403 | 403 | 3445 | 8.55 | 12 |

## calibration: 8 videos, 8024 frames, 68836 GT boxes
| video_id | frames | gt_frames | gt_boxes | mean_density | max_density |
|---|---:|---:|---:|---:|---:|
| dancetrack0083 | 603 | 603 | 14994 | 24.87 | 25 |
| dancetrack0002 | 1203 | 1203 | 9257 | 7.69 | 8 |
| dancetrack0061 | 1203 | 1203 | 6015 | 5.00 | 5 |
| dancetrack0099 | 603 | 603 | 6206 | 10.29 | 11 |
| dancetrack0015 | 1203 | 1203 | 10720 | 8.91 | 9 |
| dancetrack0033 | 803 | 803 | 6237 | 7.77 | 8 |
| dancetrack0072 | 1203 | 1203 | 5905 | 4.91 | 5 |
| dancetrack0049 | 1203 | 1203 | 9502 | 7.90 | 8 |

## val: 25 videos, 25508 frames, 225148 GT boxes
| video_id | frames | gt_frames | gt_boxes | mean_density | max_density |
|---|---:|---:|---:|---:|---:|
| dancetrack0004 | 1203 | 1203 | 4586 | 3.81 | 4 |
| dancetrack0005 | 1203 | 1203 | 4739 | 3.94 | 4 |
| dancetrack0007 | 1203 | 1203 | 9471 | 7.87 | 8 |
| dancetrack0010 | 1203 | 1203 | 7029 | 5.84 | 6 |
| dancetrack0014 | 1203 | 1203 | 12655 | 10.52 | 12 |
| dancetrack0018 | 503 | 503 | 2935 | 5.83 | 8 |
| dancetrack0019 | 2402 | 2402 | 16504 | 6.87 | 7 |
| dancetrack0025 | 803 | 803 | 7079 | 8.82 | 9 |
| dancetrack0026 | 302 | 302 | 3993 | 13.22 | 18 |
| dancetrack0030 | 1263 | 1263 | 7572 | 6.00 | 6 |
| dancetrack0034 | 923 | 923 | 11142 | 12.07 | 14 |
| dancetrack0035 | 703 | 703 | 5263 | 7.49 | 8 |
| dancetrack0041 | 1003 | 1003 | 16738 | 16.69 | 22 |
| dancetrack0043 | 183 | 183 | 1675 | 9.15 | 12 |
| dancetrack0047 | 1203 | 1203 | 9410 | 7.82 | 8 |
| dancetrack0058 | 1601 | 1601 | 11050 | 6.90 | 7 |
| dancetrack0063 | 1000 | 1000 | 7614 | 7.61 | 8 |
| dancetrack0065 | 702 | 702 | 3505 | 4.99 | 5 |
| dancetrack0073 | 703 | 703 | 8470 | 12.05 | 14 |
| dancetrack0077 | 1203 | 1203 | 10827 | 9.00 | 9 |
| dancetrack0079 | 1202 | 1202 | 14523 | 12.08 | 18 |
| dancetrack0081 | 984 | 984 | 18809 | 19.11 | 20 |
| dancetrack0090 | 1004 | 1004 | 13142 | 13.09 | 14 |
| dancetrack0094 | 603 | 603 | 11749 | 19.48 | 23 |
| dancetrack0097 | 1203 | 1203 | 4668 | 3.88 | 4 |

## Calibration selection rationale

按 GT mean density 分桶（tercile），从低/中/高各选 2/3/3 个视频，保证 calibration 覆盖低密度、中密度、高密度场景。
train/calibration 各自 video-level disjoint；同一视频不会同时出现在两个 split。
