# LocateMOT-U foundation checkpoint audit

Date: 2026-10-01

The candidate is the MMDetection conversion
`weights/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth`, SHA256
`55949c9c0f46339a73b415334765615d491ee6ed739ed3f568142b7fc5581143`.
The configuration is
`third_party/mmdetection/configs/grounding_dino/grounding_dino_swin-b_finetune_16xb2_1x_coco.py`.

The load audit uses an explicit whitelist and fails if any backbone, encoder,
decoder, bounding-box head, or language-fusion parameter is absent.  The exact
load result is:

```text
missing:
  dn_query_generator.label_embedding.weight
unexpected:
  language_model.language_backbone.body.model.embeddings.position_ids
  language_model.language_backbone.body.model.pooler.dense.bias
  language_model.language_backbone.body.model.pooler.dense.weight
core_missing_keys: []
```

`dn_query_generator.label_embedding.weight` is a training-only denoising
initialization parameter; it is the sole permitted missing key.  The three
unexpected keys are BERT position/pooler metadata and are explicitly
whitelisted.  The audit status is
`PASS_CONDITIONAL_TRAINING_ONLY_MISSING`; it is not a claim that a training
checkpoint has been produced.  Machine-readable evidence is in
`outputs/unified/protocol/foundation_checkpoint_audit.json`.

No official-test labels or screening ground truth were read.
