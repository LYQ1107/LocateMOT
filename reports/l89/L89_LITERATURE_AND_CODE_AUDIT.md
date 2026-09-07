# L89 literature and code audit

Audit date: 2026-09-07  
Project: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`  
Thread: `01a02014-fce8-7f51-8414-e7ed6ab44745`

This is a provenance audit, not a claim of reproducing any external method.
No external checkpoint, dataset, label, or source module is imported into
L89. The only implementation inputs are the frozen local L85 Z1 cache, frozen
L69 bank, local pure GroundingDINO/BERT token path used to create a new cache,
and the unchanged local L87-A loss.

## Sources checked

| Source | Primary paper / official repository and verified revision | What was checked | L89 use | Missing/limits |
|---|---|---|---|---|
| FlexHook | [CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Rethinking_Two-Stage_Referring-by-Tracking_in_Referring_Multi-Object_Tracking_Make_it_Strong_CVPR_2026_paper.html); [official repository](https://github.com/buptLwz/FlexHook), `bd1acc38634b28525d54dc6e0fcb38335f0029f9` | Repository README and current main revision; its explicit correspondence/conditioning motivation | Structural inspiration only: explicit candidate/query correspondence. No FlexHook file or checkpoint copied. | External environment, pretrained assets and tracker outputs are not part of this checkout. |
| DKGTrack | [ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Language_Decoupling_with_Fine-grained_Knowledge_Guidance_for_Referring_Multi-object_Tracking_ICCV_2025_paper.html); [official repository](https://github.com/acyddl/DKGTrack), `197f354443bd1e7b490d204456a7654b7d1e4ccd` | Paper abstract/method description and repository revision | Inspiration for fine-grained language/region interaction only. L89 does not add static/motion decomposition. | No external DKGTrack weights or supervision; `static_motion_alignment=UNALIGNED`. |
| STORM | [CVPR Findings 2026 paper](https://openaccess.thecvf.com/content/CVPR2026F/html/Lu_STORM_End-to-End_Referring_Multi-Object_Tracking_in_Videos_CVPRF_2026_paper.html); [official benchmark repository](https://github.com/amazon-science/storm-referring-multi-object-grounding), verified main `0d87c3ba52a024ffb0ea9c533ec278ae5361f4fa` | Paper and repository metadata; task-composition and end-to-end motivation | Context only. L89 intentionally does not use STORM-Bench/TCL or external data. | No claim of full public implementation reproduction; no STORM data or weights loaded. |
| COAL | [arXiv:2605.14795](https://arxiv.org/abs/2605.14795) | Primary paper record and stated counterfactual/observation alignment motivation | Context only; no synthetic counterfactuals or COAL code. | No official repository was verified, so no code/weight reuse is claimed. |
| PropVG | [ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Dai_PropVG_End-to-End_Proposal-Driven_Visual_Grounding_with_Multi-Granularity_Discrimination_ICCV2025_paper.pdf); [official repository](https://github.com/Dmmm1997/PropVG), `7f8ccd783a1721891aa503ab689c142db0f223d2` | Paper/repository record describing proposal and multi-granularity discrimination | Structural inspiration for candidate discrimination only. L89 does not transplant CRS/MTD. | No external weights, code or datasets used. |
| ReferDINO | [ICCV 2025 paper/repository record](https://github.com/iSEE-Laboratory/ReferDINO); [official repository](https://github.com/iSEE-Laboratory/ReferDINO), `3cfc01f57dff97f7d801b1bd54c251e0f34fcef8` | Repository revision and downstream grounding framing | Context only; L89 freezes the local GroundingDINO/Z1 contract. | No ReferDINO model or labels used. |
| WeDetect / WeDetect-Ref | [official repository](https://github.com/WeChatCV/WeDetect), `dd302dba0069ace1b05816bafbc3fa1dbd6aa68c` | Repository availability and shared-space retrieval framing | Prior structural context only; no code or checkpoint. | No verified task-specific local weights were used. |
| TRACT / OVTR | [TRACT paper](https://arxiv.org/abs/2503.08145); [OVTR repository](https://github.com/jinyanglii/OVTR), `500e72c19bf5f7f8717546911a5639fdc26bfee5` | Trajectory/open-vocabulary framing and repository revision | Prior context only. Ordinary OVMOT remains untouched. | No external model, data or production integration. |
| iKUN / TempRMOT / RMOT | [iKUN repository](https://github.com/dyhBUPT/iKUN), `4db56bfaec703590e0fdfd1684d9769467a67e05` (master); [TempRMOT repository](https://github.com/zyn213/TempRMOT), `6a65640d849fdee4a32bb055945ee34c3b0edeb1` (main); [RMOT repository](https://github.com/wudongming97/RMOT), `d4fedb35538e79a743ff78ff946abc6c84453cab` (master) | Repository metadata where available; historical RMOT framing | Context only; no external code or weights. | No external code, weights, or benchmark outputs were imported. |

The verified local implementation references are the tracked L85 runtime,
L86 temporal encoder, L87-A loss, and L88 data/runtime files. L89 will copy
neither their source nor any old checkpoint; it will import only the
unchanged `l87a_loss` and instantiate its own model/data integration.

## L89 boundary conclusion

The common theme in the sources is explicit language-conditioned interaction
with candidate/region or trajectory representations. L89 isolates that idea
as one QSC-D change: candidate self-attention followed by pure-language-token
cross-attention. It does not combine external motion-language annotation,
counterfactual data, proposal generation, LoRA, cardinality, or a new tracker.

The local GroundingDINO checkout is used as a frozen implementation/runtime,
not as an official verified upstream commit. Its checkpoint-load warning about
the language-model `position_ids` key and the local source reference will be
recorded in cache provenance. There is no verified token/span-to-region mask
or static/motion annotation in LocateMOT; both remain `UNALIGNED`.
