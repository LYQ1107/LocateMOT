# L83 literature and code audit

Checked 2026-09-02.  These are primary-source references and local source
audits, not claims of official reproduction.  No external checkpoint, label,
training script, or tracker is imported into L83.

| Source | Paper | Official repository/revision | Read/reuse classification |
|---|---|---|---|
| MMDetection/MM-Grounding-DINO | [MM-Grounding-DINO](https://arxiv.org/abs/2303.05499) and [MMDetection v3.3.0](https://github.com/open-mmlab/mmdetection/tree/v3.3.0) | official tag `v3.3.0`; the prompt records commit `44ebd17b145c2372c4b700bfb9cb20dbd28ab64a`, while the local checkout has no verifiable HEAD | Read-only implementation reference for native query/reference, valid ratios, decoder layers and refinement. L83 reuses only the existing local L82 runtime/data adapter; no native top-k or class score is used. |
| FlexHook | [arXiv:2503.07516](https://arxiv.org/abs/2503.07516) | [buptLwz/FlexHook](https://github.com/buptLwz/FlexHook), audited revision `bd1acc38634b28525d54dc6e0fcb38335f0029f9` | Read configs/model/data for multi-expression batching and temporal engineering. Structural inspiration only; no code/weights/data are imported. |
| DKGTrack | [ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Language_Decoupling_with_Fine-grained_Knowledge_Guidance_for_Referring_Multi-object_Tracking_ICCV_2025_paper.html) | [acyddl/DKGTrack](https://github.com/acyddl/DKGTrack), audited revision `197f354443bd1e7b490d204456a7654b7d1e4ccd` | Read data/model/config interfaces. Static/motion ideas are inspiration only; no verified motion-language labels are transferred. |
| ReferDINO | [ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/html/Liang_ReferDINO_Referring_Video_Object_Segmentation_with_Visual_Grounding_Foundations_ICCV_2025_paper.html) | [iSEE-Laboratory/ReferDINO](https://github.com/iSEE-Laboratory/ReferDINO), audited revision `3cfc01f57dff97f7d801b1bd54c251e0f34fcef8` | Read temporal grounding modules as structural context. No segmentation labels, weights, or code are used. |
| STORM | [arXiv:2604.10527](https://arxiv.org/abs/2604.10527) | [amazon-science/storm-referring-multi-object-grounding](https://github.com/amazon-science/storm-referring-multi-object-grounding), audited revision `0d87c3ba52a024ffb0ea9c533ec278ae5361f4fa` | Task-composition/benchmark context only; the public repository was not treated as a complete model implementation and no data are mixed. |
| COAL | [arXiv:2605.14795](https://arxiv.org/abs/2605.14795) | No independently verified official training repository found in this audit | Paper-level counterfactual/discriminability inspiration only. L83 exact real-label flips are not claimed as the first counterfactual RMOT method. |
| TempRMOT | [arXiv:2406.05039](https://arxiv.org/abs/2406.05039) | [zyn213/TempRMOT](https://github.com/zyn213/TempRMOT), audited revision `6a65640d849fdee4a32bb055945ee34c3b0edeb1` | Temporal RMOT framing only; no code/weights are reused. |
| iKUN | [arXiv:2312.16245](https://arxiv.org/abs/2312.16245) | [dyhBUPT/iKUN](https://github.com/dyhBUPT/iKUN), audited revision `4db56bfaec703590e0fdfd1684d9769467a67e05` | RMOT formulation reference only; frozen L69 transfer is not an official iKUN reproduction. |

L83's scientifically distinct elements are the exact real-query matrix,
target-bag duplicate invariance, dual-axis label supervision, corrected
unique-bag metrics, and the layer-wise diagnostic.  These are hypotheses under
test, not novelty or SOTA claims.  No verified token/span-to-region or
motion-language annotation is present: both remain `UNALIGNED`.
