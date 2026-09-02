# L85 literature and comparability notes

The following are primary-source/official-code references inspected for
structural context. They are not claims of reproduction, and no external
weights are imported into L85.

| Work | Primary source / code | L85 use and limitation |
|---|---|---|
| TempRMOT | [arXiv:2406.05039](https://arxiv.org/abs/2406.05039), [repository](https://github.com/zyn213/TempRMOT) | Temporal RMOT inspiration; local reference commit recorded in the stage provenance. No external model is used. |
| DKGTrack | [ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/html/Ding_DKGTrack_Dynamic_Key_Guided_Tracking_for_Open-Vocabulary_Multi-Object_Tracking_ICCV_2025_paper.html), [repository](https://github.com/acyddl/DKGTrack) | Dynamic-key/track context only; its full training/evaluation dependencies are not transplanted to frozen L69. |
| FlexHook | [arXiv:2503.07516](https://arxiv.org/abs/2503.07516), [repository](https://github.com/buptLwz/FlexHook) | Open-vocabulary temporal association inspiration; not a reused checkpoint or official reproduction. |
| ReferDINO | [ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_ReferDINO_Referring_Video_Object_Tracking_with_Open-Vocabulary_Detection_ICCV_2025_paper.html), [repository](https://github.com/IDEA-Research/ReferDINO) | Query-conditioned detection context; L85 keeps L69 acquisition fixed and does not claim comparable detector training. |
| STORM | [arXiv:2604.10527](https://arxiv.org/abs/2604.10527) | Recent RMOT/TCL direction noted as inspiration; no verified local weights or independent reproduction is used. |
| COAL | [arXiv:2605.14795](https://arxiv.org/abs/2605.14795) | Recent open-vocabulary tracking context; no local training checkout is treated as a dependency. |
| MMDetection | [v3.3.0 commit](https://github.com/open-mmlab/mmdetection/tree/44ebd17b145c2372c4b700bfb9cb20dbd28ab64a) | Runtime implementation reference used by the already audited L84 capture; local checkout HEAD is not independently verifiable. |

The L85 files actually reuse only local data/runtime contracts and small
adapter interfaces; the papers provide structural inspiration. Public code,
weights, and metrics are not assumed identical to the frozen-bank setting.
There is no verified token/span-to-region or motion-language annotation in
this project, so that status remains `UNALIGNED`. TrackEval comparability is
limited to legal, explicitly named full-video internal validation unless a
verified public split is later authorized.

## Revision-level provenance checked for L85

The following revisions were checked on 2026-09-02 against the public
repositories. They document what was inspected as a structural reference;
none of the external model weights or training code was imported into L85.

| source | paper/repository revision | relationship to L85 |
|---|---|---|
| GroundingDINO/MMDetection | [MMDetection v3.3.0 tree](https://github.com/open-mmlab/mmdetection/tree/44ebd17b145c2372c4b700bfb9cb20dbd28ab64a), commit `44ebd17b145c2372c4b700bfb9cb20dbd28ab64a` | The local GroundingDINO runtime follows this public implementation family. The local checkout has no independently verifiable HEAD; no external detector weights were copied. |
| ReferDINO | [ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_ReferDINO_Referring_Video_Object_Tracking_with_Open-Vocabulary_Detection_ICCV_2025_paper.html), [repository commit](https://github.com/iSEE-Laboratory/ReferDINO/commit/3cfc01f57dff97f7d801b1bd54c251e0f34fcef8), `3cfc01f57dff97f7d801b1bd54c251e0f34fcef8` | Temporal/query-conditioned reference only; no RVOS mask branch or checkpoint is reused. |
| DKGTrack | [ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/html/Ding_DKGTrack_Dynamic_Key_Guided_Tracking_for_Open-Vocabulary_Multi-Object_Tracking_ICCV_2025_paper.html), [repository commit](https://github.com/acyddl/DKGTrack/commit/197f354443bd1e7b490d204456a7654b7d1e4ccd), `197f354443bd1e7b490d204456a7654b7d1e4ccd` | Dynamic-key and multi-frame engineering inspiration; its detector/training/evaluation stack is not transplanted to the frozen L69 bank. |
| FlexHook | [paper](https://arxiv.org/abs/2503.07516), [repository commit](https://github.com/buptLwz/FlexHook/commit/bd1acc38634b28525d54dc6e0fcb38335f0029f9), `bd1acc38634b28525d54dc6e0fcb38335f0029f9` | Multi-frame/multi-expression batching inspiration only; no external weights or code path is used. |
| TempRMOT | [arXiv:2406.05039](https://arxiv.org/abs/2406.05039), [repository commit](https://github.com/zyn213/TempRMOT/commit/6a65640d849fdee4a32bb055945ee34c3b0edeb1), `6a65640d849fdee4a32bb055945ee34c3b0edeb1` | Temporal RMOT and TrackEval layout reference only; no official result is claimed for L85. |
| STORM | [arXiv:2604.10527](https://arxiv.org/abs/2604.10527), [public repository](https://github.com/amazon-science/storm-referring-multi-object-grounding) | Task-composition/end-to-end RMOT inspiration; no verified local commit, weights, or complete reproduction was available, so no code is reused. |
| COAL | [arXiv:2605.14795](https://arxiv.org/abs/2605.14795) | Counterfactual/semantic-injection context only; no dependency or checkpoint is used. |

The public papers' benchmark numbers are not directly comparable to this
L85 result: L85 uses an internal query-sequence wrapper around a frozen L69
candidate bank and reports full-video validation HOTA, while public systems
may train the detector, use different proposal pools, or use official split
protocols. L85 therefore reports its own TrackEval numbers without claiming a
public-method reproduction or a fair leaderboard ranking.
