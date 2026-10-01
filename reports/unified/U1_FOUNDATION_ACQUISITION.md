# U1 foundation acquisition — 2026-10-01

The U1 foundation assets are now present. The original checkpoint at
`weights/grounding_dino_swin-b.pth` was downloaded from the official
GroundingDINO v0.1.0-alpha2 release URL used by the audited ReferDINO Swin-B
configuration. The local MMDetection runtime uses the converted OpenMMLab
checkpoint at `weights/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth`.

| Field | Value |
|---|---|
| Model | `groundingdino_swinb_cogcoor` |
| Original path | `weights/grounding_dino_swin-b.pth` |
| Original size / SHA256 | `938057991` / `46270f7a822e6906b655b729c90613e48929d0f2bb8b9b76fd10a856f3ac6ab7` |
| MMDetection path | `weights/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth` |
| MMDetection size / SHA256 | `935971702` / `55949c9c0f46339a73b415334765615d491ee6ed739ed3f568142b7fc5581143` |
| Status | loaded and label-free forward smoked |

The local adapter built the Swin-B model with `232996763` parameters and
loaded the converted checkpoint with eight expected missing initialization
keys and three BERT pooler/position metadata keys reported as unexpected. A
CUDA forward on legal V1 video `0004`, frame `000063.png`, with the text
specification `car . pedestrian . cyclist .` returned 300 finite predictions,
finite boxes and finite scores. This is a foundation compatibility smoke, not
a proposal-quality or tracking result. Proposal and expression grounding must
pass before any tracking head is trained. No Refer-KITTI official-test labels
or screening ground truth were read.
