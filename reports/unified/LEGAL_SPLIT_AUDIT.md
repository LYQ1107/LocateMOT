# LocateMOT-U legal split audit (v2)

The split is deterministic with seed `20261001` and source manifest SHA256 `f520fce033d44c18ff31358dcea8cffec4006ae1ba6d4fb7ec14059b9b57b112`.
It is frozen before any U1 formal evaluation or adaptation training.
Official evaluation videos are represented as metadata and were not opened.

## Assignment contract

Video quotas are fit 60%, calibration 20%, validation 20% (with at least two validation and one calibration video per dataset). High-volume videos are assigned first by a fixed objective over log expression volume, unique physical target volume, and moving-track ratio; SHA256(seed, dataset, video) breaks ties.

## refer_kitti_v1

| split | videos | expressions | valid target refs | missing source refs | unique physical targets |
|---|---:|---:|---:|---:|---:|
| fit | 9 | 314 | 75184 | 0 | 7541 |
| calibration | 3 | 178 | 139866 | 0 | 10283 |
| validation | 3 | 168 | 73980 | 0 | 5231 |

| video | frames | expressions | valid refs | missing refs | moving track ratio | size small/medium/large |
|---|---:|---:|---:|---:|---:|---|
| 0001 | 447 | 66 | 37764 | 0 | 1.000 | 645/1606/585 |
| 0002 | 233 | 58 | 14732 | 0 | 1.000 | 694/359/30 |
| 0003 | 144 | 34 | 3750 | 0 | 1.000 | 114/192/57 |
| 0004 | 314 | 49 | 9382 | 0 | 0.968 | 453/365/63 |
| 0006 | 270 | 18 | 4646 | 0 | 1.000 | 285/177/75 |
| 0007 | 800 | 76 | 34940 | 0 | 1.000 | 450/1166/709 |
| 0008 | 390 | 68 | 14502 | 0 | 1.000 | 684/319/39 |
| 0009 | 803 | 58 | 36846 | 0 | 1.000 | 893/1408/559 |
| 0010 | 294 | 50 | 6956 | 0 | 1.000 | 205/420/32 |
| 0012 | 78 | 22 | 1716 | 0 | 1.000 | 193/11/0 |
| 0014 | 106 | 34 | 4218 | 0 | 1.000 | 203/327/50 |
| 0015 | 376 | 36 | 10522 | 0 | 1.000 | 400/510/703 |
| 0016 | 209 | 13 | 19262 | 0 | 1.000 | 279/1720/840 |
| 0018 | 339 | 42 | 22632 | 0 | 0.944 | 305/814/239 |
| 0020 | 837 | 36 | 67162 | 0 | 0.965 | 1805/2835/676 |

## refer_kitti_v2

| split | videos | expressions | valid target refs | missing source refs | unique physical targets |
|---|---:|---:|---:|---:|---:|
| fit | 9 | 4350 | 600470 | 157 | 5506 |
| calibration | 3 | 1693 | 1038793 | 0 | 10988 |
| validation | 3 | 1694 | 602729 | 0 | 6768 |

| video | frames | expressions | valid refs | missing refs | moving track ratio | size small/medium/large |
|---|---:|---:|---:|---:|---:|---|
| 0000 | 154 | 596 | 37480 | 32 | 1.000 | 14/178/72 |
| 0001 | 447 | 614 | 270630 | 0 | 1.000 | 645/1606/585 |
| 0002 | 233 | 449 | 84223 | 0 | 1.000 | 694/359/30 |
| 0003 | 144 | 442 | 56231 | 0 | 1.000 | 114/192/57 |
| 0006 | 270 | 618 | 109716 | 0 | 1.000 | 285/177/75 |
| 0007 | 800 | 711 | 180203 | 0 | 1.000 | 450/1166/709 |
| 0008 | 390 | 620 | 103510 | 0 | 1.000 | 684/319/39 |
| 0009 | 803 | 605 | 231851 | 0 | 1.000 | 893/1408/559 |
| 0010 | 294 | 572 | 49560 | 0 | 1.000 | 205/420/32 |
| 0012 | 78 | 339 | 21897 | 0 | 1.000 | 193/11/0 |
| 0014 | 106 | 361 | 38346 | 5 | 1.000 | 203/326/51 |
| 0015 | 376 | 478 | 108281 | 0 | 1.000 | 400/492/721 |
| 0016 | 209 | 505 | 314245 | 0 | 1.000 | 279/1709/851 |
| 0017 | 145 | 353 | 99507 | 120 | 1.000 | 28/282/468 |
| 0020 | 837 | 474 | 536312 | 0 | 0.965 | 1906/2752/658 |

## V2 anomaly contract

V2 source-target-ID-missing references counted from legal expression/label joins: `157`. Missing references remain in the source rows and are excluded only from valid-positive denominators; they are not replaced, converted to negatives, or silently dropped.

Machine-readable output: `outputs/unified/protocol/legal_video_split_v2.json`.

`official_test_labels_read=false`, `screening_gt_used=false`, and `training_started=false`.
