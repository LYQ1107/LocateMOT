# R0 recent method and code audit

Audit date: 2026-09-09.  The links below are primary paper pages or the
authors' repositories/commits.  They are structural references only; R0 does
not copy their checkpoints, datasets, tracker code, or benchmark numbers.

## FlexHook — CVPR 2026

- Paper: https://openaccess.thecvf.com/content/CVPR2026/html/Li_Rethinking_Two-Stage_Referring-by-Tracking_in_Referring_Multi-Object_Tracking_Make_it_Strong_CVPR_2026_paper.html
- Official repository: https://github.com/buptLwz/FlexHook
- Audited revision: `bd1acc38634b28525d54dc6e0fcb38335f0029f9`.
- The repository commit is verifiable on GitHub and updates the README paper
  reference.  The paper/repository informed the choice to treat feature
  construction, local visual observation, language-conditioned cues, and
  active pairwise correspondence as first-order design questions.  No
  FlexHook source is imported by R0 and no FlexHook weight is used.
- Missing/limited for this migration: the original training/data/dependency
  contract is not transplanted to the frozen L69 bank.  R0 is therefore only
  a structural probe, not an official reproduction.

## DKGTrack — ICCV 2025

- Paper: https://openaccess.thecvf.com/content/ICCV2025/html/Li_Language_Decoupling_with_Fine-grained_Knowledge_Guidance_for_Referring_Multi-object_Tracking_ICCV2025_paper.html
- Official repository: https://github.com/acyddl/DKGTrack
- Audited revision: `197f354443bd1e7b490d204456a7654b7d1e4ccd`.
- The repository revision is verifiable and changes a training-epoch setting;
  the paper/repository are used only for the documented region-level and
  motion-aware correspondence motivation.  R0 does not copy DKGTrack code,
  language decomposition, or its reported results, and does not use its
  missing external weights/dependencies.

## STORM — CVPR Findings 2026

- Paper: https://openaccess.thecvf.com/content/CVPR2026F/html/Lu_STORM_End-to-End_Referring_Multi-Object_Tracking_in_Videos_CVPRF_2026_paper.html
- Structural use: the paper is cited only for the data/supervision and
  end-to-end RMOT motivation.  STORM external data and implementation are not
  added to R0.  The Open Access page was not fetchable in this environment
  (HTTP 403), so no source-code or weight reuse is claimed.

## COAL — 2026

- Primary paper: https://arxiv.org/abs/2605.14795
- The arXiv abstract explicitly discusses the sparse-supervision versus
  discriminability contradiction.  It is context for why R0 uses dense native
  frame supervision, not a source of VLM/LLM synthetic labels.  No COAL code,
  checkpoint, or external label is used.

## PropVG — ICCV 2025

- Paper: https://openaccess.thecvf.com/content/ICCV2025/html/Dai_PropVG_End-to-End_Proposal-Driven_Visual_Grounding_with_Multi-Granularity_Discrimination_ICCV2025_paper.html
- Official repository: https://github.com/Dmmm1997/PropVG
- The paper/repository motivate multi-granularity discrimination only.  No
  PropVG code, weights, proposal generator, or word-level annotations are
  used in R0.  Token/span-to-region and static/motion language alignment are
  `UNALIGNED` in this project.

## Local implementation and fairness boundary

R0 actually reuses only the already audited local interfaces:

- `locatemot/rmot/l82_grounding_runtime.py`: `build_groundingdino`, the local
  MMDetection 3.3.0 config/checkpoint/BERT initialization and frozen detector;
- `locatemot/models/l82_grounding_reference.py`: existing normalized box
  conversion and `grid_sample` coordinate convention;
- `tools/l89d_fullvideo_common.py`, `tools/l86_infer_fullvideo.py`, and
  `tools/l89_trackeval_matrix.py`: native timeline/key and TrackEval wrapper
  contracts, read-only;
- the frozen L69 bank and the separately audited pure language cache as data,
  never as modified source.

The R0 implementation is new under `locatemot/models/r0_*`,
`locatemot/rmot/r0_*`, and `tools/r0_*`.  It does not import a competing
benchmark method.  GroundingDINO remains a local checkout with no verified
official commit in the existing audit; the local checkpoint is reused only as
the frozen detector specified by LocateMOT.  R0 does not train GroundingDINO,
UIDM, the tracker, or any ordinary MOT/OVMOT path, and it does not claim an
official RMOT reproduction.
