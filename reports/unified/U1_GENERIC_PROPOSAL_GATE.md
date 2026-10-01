# U1 formal generic proposal gate

- Execution code commit: `75264a762824469725ff963f99d0fef42691af9d`
- Frozen legal split SHA-256: `8ee806048ccbb227919f27edf68cf0bebc1fa50f824a8108878fc5f39fc6e1c0`
- Foundation checkpoint SHA-256: `55949c9c0f46339a73b415334765615d491ee6ed739ed3f568142b7fc5581143`
- Preprocessing: LoadImageFromFile -> FixScaleResize(800,1333,keep_ratio=True) -> PackDetInputs -> DetDataPreprocessor(bgr_to_rgb=True); boxes restored to original coordinates.
- Fixed vocabulary: `car . van . truck . bus . tram . pedestrian . person . cyclist . bicycle . motorcycle .`
- Unit: `dataset/video/frame_id/target_id`; duplicate expression references were not re-counted.
- Scope: all image frames in frozen calibration and validation videos; frames without a materialized label file contribute no GT unit.

## Gate result

- Status: **FORMAL_COMPLETE**
- Gate pass: **False** (required V1 and V2 Top150 Recall@IoU0.50 >= 0.90)
- Images evaluated: `4201`
- Physical GT units: `33526`
- Unique physical objects: `5992` (minimum 500 satisfied)

| scope | Top100 @.25/.50/.75 | Top150 @.25/.50/.75 | Top300 @.25/.50/.75 | Top900 @.25/.50/.75 |
|---|---|---|---|---|
| overall | 0.4541/0.0603/0.0052 | 0.5113/0.0732/0.0054 | 0.6171/0.1018/0.0064 | 0.7753/0.1671/0.0097 |
| refer_kitti_v1 | 0.4739/0.0662/0.0043 | 0.5309/0.0803/0.0045 | 0.6326/0.1116/0.0056 | 0.7871/0.1789/0.0094 |
| refer_kitti_v2 | 0.4367/0.0551/0.0060 | 0.4940/0.0669/0.0062 | 0.6034/0.0932/0.0070 | 0.7650/0.1567/0.0100 |
| small | 0.5183/0.0817/0.0106 | 0.5855/0.0975/0.0107 | 0.6882/0.1307/0.0121 | 0.8144/0.1928/0.0162 |
| medium | 0.4717/0.0602/0.0042 | 0.5290/0.0726/0.0046 | 0.6416/0.1022/0.0054 | 0.8040/0.1646/0.0085 |
| large | 0.3232/0.0316/0.0004 | 0.3666/0.0417/0.0004 | 0.4597/0.0618/0.0009 | 0.6507/0.1384/0.0041 |

The three numbers in each cell are Recall@IoU0.25, Recall@IoU0.50, and Recall@IoU0.75. Size bins use GT box height: small `<32 px`, medium `32–96 px`, large `>96 px`. Top100/300/900 are diagnostics; Top150 is the registered main gate.

### Main Top150 values

- V1: IoU0.25=0.5309, IoU0.50=0.0803, IoU0.75=0.0045
- V2: IoU0.25=0.4940, IoU0.50=0.0669, IoU0.75=0.0062

The measured IoU0.50 recalls are far below the internal 0.80 adaptation early-pass and the final 0.90 gate. No U2 or tracking-core work is authorized by this result. Raw worker shards remain outside the repository.

This is the unadapted foundation measurement.  The controlled U1-A
adaptation checkpoints and the final stop decision are recorded separately in
`reports/unified/U1_GENERIC_ADAPTATION.md` and
`reports/unified/U1_FAILURE_DECOMPOSITION.md`; the legal split, vocabulary,
and inference contract are unchanged.
