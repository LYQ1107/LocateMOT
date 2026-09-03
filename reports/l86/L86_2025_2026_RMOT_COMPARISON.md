# L86 literature and comparability notes

The following primary sources were used as structural context only. No public
checkpoint, external label, detector, tracker, or training code was imported
into L86.

| work | paper / repository revision | relationship to L86 |
|---|---|---|
| TempRMOT | [arXiv:2406.05039](https://arxiv.org/abs/2406.05039), [repository commit `6a65640`](https://github.com/zyn213/TempRMOT/commit/6a65640d849fdee4a32bb055945ee34c3b0edeb1) | Temporal RMOT and sequence-evaluation inspiration only. |
| DKGTrack | [ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/html/Ding_DKGTrack_Dynamic_Key_Guided_Tracking_for_Open-Vocabulary_Multi-Object_Tracking_ICCV_2025_paper.html), [repository commit `197f354`](https://github.com/acyddl/DKGTrack/commit/197f354443bd1e7b490d204456a7654b7d1e4ccd) | Dynamic-key/open-vocabulary tracking context; its detector and training stack were not used. |
| FlexHook | [arXiv:2503.07516](https://arxiv.org/abs/2503.07516), [repository commit `bd1acc3`](https://github.com/buptLwz/FlexHook/commit/bd1acc38634b28525d54dc6e0fcb38335f0029f9) | Multi-frame/multi-expression structural inspiration only. |
| ReferDINO | [ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_ReferDINO_Referring_Video_Object_Tracking_with_Open-Vocabulary_Detection_ICCV_2025_paper.html), [repository commit `3cfc01f`](https://github.com/iSEE-Laboratory/ReferDINO/commit/3cfc01f57dff97f7d801b1bd54c251e0f34fcef8) | Query-conditioned detection context; no RVOS branch or weights were transplanted. |
| STORM | [arXiv:2604.10527](https://arxiv.org/abs/2604.10527), [public repository](https://github.com/amazon-science/storm-referring-multi-object-grounding) | Recent RMOT/task-composition context; no verified local revision or weights were used. |
| COAL | [arXiv:2605.14795](https://arxiv.org/abs/2605.14795) | Recent semantic-injection context; no dependency was imported. |
| GroundingDINO/MMDetection | [MMDetection v3.3.0 tree, commit `44ebd17`](https://github.com/open-mmlab/mmdetection/tree/44ebd17b145c2372c4b700bfb9cb20dbd28ab64a) | Runtime/source family behind the frozen local representation; local checkout HEAD is not verifiable. |

L86 actually reused only local audited L69/L85 data interfaces, the frozen Z1
cache, local model definitions, and the local TrackEval API. The public
methods above are not claimed reproductions and their benchmark results are
not comparable to this internal frozen-bank validation. Public systems may
train a detector, use different proposals, or use official benchmark splits;
L86 does none of those. Token/span-to-region and motion-language annotations
are unavailable, so the corresponding statuses remain `UNALIGNED`.
