# Stage L10 — Literature and Code Audit (2026)

Date: 2026-08-17

Every entry below was verified by opening the official page / cloning
and reading the linked repository.  Nothing is cited from chat memory
alone.  The L8/L9 audits (`reports/l8_literature_and_code_audit.md`,
`reports/l9_literature_and_code_audit.md`) remain valid for MOTIP, iKUN,
TransRMOT, OVTR, TRACT, AED and QTrack; this document adds the 2025/2026
references that directly affect L10's OVMOT supervision scaling and
RMOT breadth.

## 1. COVTrack / C-TAO — ICCV 2025

- Paper: "COVTrack: Continuous Open-Vocabulary Tracking via Adaptive
  Multi-Cue Fusion", Zekun Qian, Ruize Han, Zhixiang Wang, Junhui Hou,
  Wei Feng; ICCV 2025, pp. 10054-10063.
- Paper URL:
  https://openaccess.thecvf.com/content/ICCV2025/html/Qian_COVTrack_Continuous_Open-Vocabulary_Tracking_via_Adaptive_Multi-Cue_Fusion_ICCV_2025_paper.html
- Official GitHub: https://github.com/zekunqian/COVTrack
  - Local clone: `LocateMOT_reference_repos/covtrack`
  - Commit: `9b0ced5779ee36f5dd73dbe39b5ae5d57abb4b3b`
  - License: Apache-2.0
- Model/data release: https://huggingface.co/clarkqian/COVTrack
  (README documents `ctao_dataset/ctao_base.json`,
  `ctao_base_and_novel.json`, `ctao_public.pth`, etc.)
- Mechanism inspected (files):
  - `ovtrack/datasets/tao_dataset.py` (C-TAO COCO-video loader,
    key-image sampling, `extra_sample_ratio`)
  - `configs/uncertainty-ovtrack-teta/ovtrack_r50_ctao_train.py`
    (C-TAO training config: `extra_sample_ratio=8`, ref sampler scope 30)
  - `ovtrack/models/roi_heads/ovtrack_roi_head.py`,
    `ovtrack/models/trackers/ovtracker.py` (multi-cue fusion and
    confidence-aware association)
  - `tools/qa/validate_ctao_annotations.py` (C-TAO annotation QA)
- Observed implementation: C-TAO completes missing frame-level TAO
  annotations (490,210 frames, 1,489,637 boxes, 2,588 tracks across the
  same 500 TAO train videos), giving ~26x annotation density; COVTrack
  itself trains a detection+association model on C-TAO with adaptive
  fusion of appearance/motion/semantic cues.
- Direct relevance to L10: C-TAO is the strongest verified 2025 source
  of continuous OVMOT identity supervision for TAO train.  We use its
  annotations (base categories) as the GT for L10's DLA-candidate
  alignment.  We do **not** copy COVTrack's fusion model; UIDM remains
  the shared identity-dynamics core (closed-set + OV + referring).
- Local copy of the C-TAO annotation file (already on this server from
  the official HF release): `.../OCD_OVMOT/data/external_annotations/
  covtrack/ctao_base.json` (546,599,277 bytes; sha256 prefix
  `4269b1a5`).  Evaluation stays on official TAO val.

## 2. COVTrack++ — arXiv 2026

- Paper: "COVTrack++: Learning Open-Vocabulary Multi-Object Tracking
  from Continuous Videos via a Synergistic Paradigm", Zekun Qian, Wei
  Feng et al.; arXiv:2603.24016 (submitted 2026-03-25).
- Paper URL: https://arxiv.org/abs/2603.24016
- Official GitHub: **not yet available** — the abstract states "code and
  dataset will be publicly available".
- Relevance: extends C-TAO/COVTrack with a bidirectional detection-
  association synergy.  Confirms that continuous TAO supervision is an
  active 2026 direction; no code is available to inspect, so it is not
  used as an implementation reference.

## 3. TempRMOT / Refer-KITTI-V2 — arXiv 2024

- Paper: "Bootstrapping Referring Multi-Object Tracking", Yani Zhang,
  Dongming Wu, Wencheng Han, Xingping Dong; arXiv:2406.05039.
- Paper URL: https://arxiv.org/abs/2406.05039
- Official GitHub: https://github.com/zyn213/TempRMOT
  - Local clone: `LocateMOT_reference_repos/temp_rmot`
  - Commit: `6a65640d849fdee4a32bb055945ee34c3b0edeb1`
  - License: **none detected** in the repo root (recorded as unknown;
    no code is copied into LocateMOT).
- Files inspected:
  - `datasets/refer_kitti.py` (KITTI labels_with_ids loader,
    expression-conditioned RMOT training; label format
    `class_id track_id x1 y1 w h` in KITTI-normalized coordinates)
  - `datasets/README.md` (Refer-KITTI-V2 data organization; images from
    official KITTI tracking benchmark, expressions/labels from the
    authors' Google Drive)
  - `configs/temp_rmot_train.sh`, `configs/temp_rmot_test.sh`
    (official train/test entry points, MOTR-style)
  - `datasets/data_path/seqmap.txt` (861 official evaluation
    sequence+expression entries used by TrackEval)
  - `TrackEval/` (official RMOT evaluator, KITTI-format)
- Observed dataset facts:
  - Refer-KITTI-V2 starts from 2,719 manual annotations and expands them
    with GPT-3.5 to 9,758 total annotations using 617 different words.
  - It reuses the 21 KITTI tracking training sequences (0000-0020) with
    an official split: 17 training sequences (5,171 frames) and 4
    held-out evaluation sequences (0005/0011/0013/0019, 861 expressions
    in `seqmap.txt`).
  - Local data present: `/data1/LWR/vranlee/MFT2025/REFER-MFT25/
    refer-kitti-v2/` with `expression/` (21 sequence dirs, per-expression
    JSON with `label` frame→track-id map, `sentence`, `raw_sentence`,
    `ignore`) and `labels_with_ids/image_02/<seq>/<frame>.txt`.
  - KITTI tracking images were **missing** locally; Stage L10 started a
    download of the official `data_tracking_image_2.zip` from the KITTI
    benchmark S3 mirror (see `reports/l10_refer_kitti_and_v2_audit.md`).
- What we use: the official Refer-KITTI-V2 data layout and evaluator
  (TrackEval KITTI format) as a second RMOT benchmark.  We do **not**
  reuse TempRMOT's architecture (MOTR-style Deformable-DETR); the UIDM
  consumes expression sentences through the shared spec encoder.

## 4. ReaMOT — arXiv 2025

- Paper: "ReaMOT: A Benchmark and Framework for Reasoning-based
  Multi-Object Tracking", Sijia Chen, Yanqiu Yu, En Yu, Wenbing Tao;
  arXiv:2505.20381 (2025).
- Paper URL: https://arxiv.org/abs/2505.20381
- Official GitHub: https://github.com/chen-si-jia/ReaMOT
  - Local clone: `LocateMOT_reference_repos/reamot`
  - Commit: `1695160007e57f30e7d758ea087bafe3d649e841`
  - License: MIT
- Inspected: README + repo contents.  The benchmark/framework is not yet
  fully released ("upon acceptance ... fully open-source"); only
  comparison material is present.  No usable code/data to adopt.
- Relevance: confirms reasoning-level referring tracking is an active
  2025 direction; not used in L10.

## 5. CRMOT — AAAI 2025

- Paper: "Cross-View Referring Multi-Object Tracking", Sijia Chen, En
  Yu, Wenbing Tao; AAAI 2025.
- Paper URL: https://ojs.aaai.org/index.php/AAAI/article/view/32219 ;
  arXiv:2412.17807
- Official GitHub: https://github.com/chen-si-jia/CRMOT
  - Local clone: `LocateMOT_reference_repos/crmot`
  - Commit: `50ffe32d1ce6938c0b80c2c5559c542b184249af`
  - License: MIT
- Inspected: README, dataset/eval layout (CAMPUS/DIVOTrack-based CRTrack,
  13 scenes, 221 descriptions, cross-view).
- Relevance: cross-view RMOT is outside LocateMOT's single-view unified
  scope; recorded as related work only.

## 6. QTrack — arXiv 2026 (re-verification)

Already audited in L9 (`reports/l9_literature_and_code_audit.md`):
QTrack (arXiv:2603.13759, RMOT26 benchmark, Apache-2.0) remains the
closest 2026 query-driven-RMOT reference.  It uses a 3B VLM + RL
(TPA-PO), not a shared identity-dynamics core across MOT/OVMOT/RMOT, so
it does not pre-empt LocateMOT's claim.  No changes for L10.

## 7. Novelty status after the L10 audit

After inspecting COVTrack/C-TAO (ICCV 2025), COVTrack++ (2026),
OVTR, TRACT, AED, QTrack, MOTIP, iKUN, TransRMOT, TempRMOT/Refer-KITTI-V2,
ReaMOT and CRMOT, we did **not** identify a published, verifiable system
that:

1. uses **one trained identity-dynamics core** with persistent memory,
   lifecycle, Existing/NEW/NO-MATCH and set-level competition;
2. serves **one shared checkpoint** across closed-set MOT, open-vocabulary
   MOT and referring-expression MOT;
3. is driven by a unified frozen observation space (PBD identity token +
   CLIP semantic token + specification token).

COVTrack is the closest OVMOT supervision advance (C-TAO), and QTrack is
the closest 2026 language-conditioned tracker, but neither unifies the
three formulations in one identity-dynamics core.  The claim remains
phrased as "we did not identify ...", not "first".
