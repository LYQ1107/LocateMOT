# LocateMOT-U U1R2 class-aware generic data audit

`prepare_u1_generic_adaptation_v2.py` builds one row per physical
`video/frame` from legal fit only. V1/V2 rows sharing a physical image are
unioned and duplicate objects are removed before training. Classes are recovered
by same-frame maximum-IoU matching to official KITTI tracking labels, because
Refer and official track IDs are separate namespaces.

| quantity | value |
|---|---:|
| unique physical images | 2,545 |
| unique boxes | 10,502 |
| V1 source rows | 8,757 |
| V2 source rows | 5,508 |
| V1/V2 overlap removed | 3,424 |
| conflicting V1/V2 boxes | 580 |
| duplicate official-object rows removed | 318 |
| missing class matches | 21 |
| low-confidence geometry matches | 1,226 |

The fixed mapping is `Car -> car`, `Van -> van`, `Truck -> truck`, `Bus -> bus`,
`Pedestrian -> pedestrian`, `Person_sitting -> person`, `Cyclist -> cyclist`,
and `Tram -> tram`; `Misc` and `DontCare` are ignored. The observed fit counts
are car 6,787, pedestrian 3,700, cyclist 13, and van 2; no unsupported class
was fabricated. Low-confidence matches are retained and reported rather than
silently dropped. The manifest uses source top-left extents without evaluation
clipping.

Evidence: `outputs/unified/u1r2/generic_train_manifest.json`,
`generic_train_manifest.jsonl`, `generic_category_mapping.json`, and
`generic_label_map.json`. Official test labels remain unread.
