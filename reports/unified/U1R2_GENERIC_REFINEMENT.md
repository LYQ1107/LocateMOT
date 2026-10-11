# LocateMOT-U U1R2 generic refinement

The corrected epoch-4 checkpoint (SHA-256
`94fcc1d1b5e29ab24b8b926ae868736449832248a9415e9a3796f8b98ed3db80`) was the
only refinement starting point. The run used the class-aware, deduplicated
physical-frame manifest, FP32/TF32, AdamW weight decay `1e-4`, gradient clip
`0.1`, and the bounded low learning rates: decoder/head `2e-5`, multimodal
encoder `1e-5`, Swin stages 2/3 `2e-6`, and BERT `0`. Stages 0/1 and BERT were
frozen. No BF16 or AMP path was used.

The first R2 epoch passed the repaired generic gate, so no second epoch and no
BDD/COCO mixed fallback were started. The selected generic checkpoint is
`outputs/unified/u1r2/generic_refinement_epoch1/epoch_1.pth` (SHA-256
`f197bbbf26e784a6a808fe2bf0481f4432d5e7bca6c38d43df9f9e887cbbe1af`). It has
4,201 legal images, 33,526 physical GT units, and 5,992 physical objects.

| scope | Top150 @.25 | Top150 @.50 | Top150 @.75 |
|---|---:|---:|---:|
| V1 | 0.995234 | 0.982716 | 0.890259 |
| V2 | 0.993086 | 0.971837 | 0.748440 |
| overall | 0.994094 | 0.976943 | 0.815009 |

The size-stratum Top150 IoU0.50 recalls are small `0.937467`, medium
`0.989984`, and large `0.997825`. This confirms that the large-object residual
identified in the source audit is repaired by the bounded class-aware
refinement, without changing the architecture.

Evidence: `outputs/unified/u1r2/generic_refinement_epoch1/`,
`outputs/unified/u1r2/formal_generic_refinement_epoch1/result.json`, and
`outputs/unified/u1r2/U1A_SELECTED_CHECKPOINT.json`.
