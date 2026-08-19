# Stage L12 — Point / Box / Mask Prompt Tracking Literature & Code Audit

Date: 2026-08-19
Project: LocateMOT (Stage L12, prompt-seeded unified identity dynamics)

## Purpose

Extend the unified specification space from semantic discovery (closed
category / open category / referring expression) to seeded
specifications (point / box / mask), while keeping ONE shared UIDM and
ONE shared checkpoint.  This audit records verified 2024–2026 official
code and benchmarks for the prompt-to-identity interface.

## Verified references

### 1. SAM 2 — Segment Anything in Images and Videos

- Paper/status: arXiv 2408.00714 (2024); Meta official.
- Official repo: https://github.com/facebookresearch/sam2
- Local data: `/data1/LWR/vranlee/SERVER_ONLY/avis/SAM2`
- License: Apache-2.0.
- Mechanism (verified from repo/README):
  - Promptable video predictor: add point / box / mask prompts on any
    frame; memory bank propagates object masks/features to later frames.
  - Multi-object support via `add_new_points_or_box` per object id.
  - Evaluated on DAVIS, YouTube-VOS, MOSE, LVOS, BURST, SA-V.
- L12 relevance: prompt -> region interface; memory bank / per-object
  state informs our seeded identity token design.  We do NOT copy SAM2
  memory code; UIDM remains the identity core.

### 2. SAM 3 — unified promptable segmentation and tracking

- Status: announced 2025; official repo
  https://github.com/facebookresearch/sam3 (third-party mirrors exist;
  weights require HF request).
- Mechanism (verified via HF transformers docs / mirrors):
  - Promptable Visual Segmentation (PVS): point / box / mask / text
    prompts on images and videos; video tracking with object ids.
- L12 relevance: strongest available localizer for point/box/mask ->
  region.  If the SAM3 checkpoint is not locally available/authorized,
  we fall back to SAM2 (locally present) and record the substitution.

### 3. SAM-PT — Segment Anything Meets Point Tracking

- Paper: WACV 2025 (arXiv 2307.01197).
- Official repo: https://github.com/SysCV/sam-pt
- Mechanism: point propagation (PIPS-style) + SAM mask decoding;
  zero-shot video segmentation from point seeds.
- L12 relevance: point-prompt seeding design evidence (deterministic
  interior point derived from a mask is an acceptable controlled
  prompt-type eval, but is not an official point protocol).

### 4. Tracking-Anything-with-DEVA / DEVA

- Paper: ECCV 2024 (arXiv 2309.05490).
- Official repo: https://github.com/hkchengrex/Tracking-Anything-with-DEVA
- Mechanism: decoupled video segmentation: class-agnostic mask
  propagation + open-vocabulary recognition; object-centric memory.
- L12 relevance: decoupling "where" (localization/mask) from "who"
  (identity) supports our UIDM/localizer separation.

### 5. BURST benchmark

- Paper: WACV 2023 (arXiv 2209.12118).
- Official repo: https://github.com/Ali2500/BURST-benchmark
- Content: 2,914 videos, 482 object classes, mask-level annotations,
  6 tasks (point/box/mask/first-frame/zero-shot etc.).
- L12 relevance: one of the few benchmarks with explicit point/box/mask
  task protocols; candidate if data is obtainable.

### 6. DAVIS 2017

- Paper: CVPR 2017 challenge paper (arXiv 1704.00675).
- Official eval: https://github.com/davisvideochallenge/davis-2017
- Local data: `/data1/LWR/vranlee/SERVER_ONLY/avis/DAVIS/DAVIS`
  (Annotations 2016/2017, ImageSets, JPEGImages/480p; 819 MB).
- Protocol: first-frame mask supervision, multi-object (2017), J&F
  metrics.
- L12 relevance: PRIMARY benchmark if no official point/box protocol is
  locally available; we derive bbox + one deterministic interior point
  from the official first-frame GT mask and explicitly label this as a
  CONTROLLED prompt-type evaluation (not official point/box protocol).

### 7. YouTube-VOS

- Paper: ECCV 2018 (arXiv 1809.03327).
- Official: https://youtube-vos.org
- Protocol: sparse annotation structure; multi-stage evaluation; J&F.
- L12 relevance: fallback/extension benchmark; not confirmed local.

### 8. MOSE

- Paper: ICCV 2023 (arXiv 2302.01872).
- Official: https://MOSE.video
- Content: 2,149 clips / 5,200 objects / 36 categories; complex scenes,
  occlusion-heavy; MOSEv2 exists (2026).
- L12 relevance: occlusion/long-term stress test for identity
  persistence; not confirmed local.

### 9. LVOS

- Paper: ICCV 2023 (arXiv 2211.10181); LVOS v2 (2025).
- Content: 720 videos / 296,401 frames / 407,945 annotations.
- L12 relevance: long-term occlusion; not confirmed local.

## Local asset status

- DAVIS 2017: PRESENT (Annotations/2017, ImageSets/2017,
  JPEGImages/480p).
- SAM2: PRESENT (`/data1/LWR/vranlee/SERVER_ONLY/avis/SAM2` and
  `sam2long` outputs).
- SAM3: official repo not confirmed locally; weights require HF
  authorization -> fallback to SAM2 unless authorization is obtained.
- BURST / YouTube-VOS / MOSE / LVOS: NOT confirmed local (download
  blockers if needed; DAVIS suffices for the controlled experiment).

## Design implication for L12

1. DAVIS 2017 val is the primary prompt benchmark (multi-object,
   local, license-clean).
2. First-frame official GT masks -> three controlled prompt types:
   mask (GT), bbox = tight bbox(mask), point = one deterministic
   interior point (e.g., mask centroid or top-left interior point).
   This is documented as controlled prompt-type evaluation, not an
   official point/box benchmark.
3. Localization/region interface: SAM2 (or SAM3 if available) maps
   point/box/mask -> region crop; the existing PBD/CLIP encoder maps
   region -> unified seed identity token for UIDM.
4. UIDM (frozen first, then joint fine-tune) handles identity
   persistence, absence (NO_MATCH), reactivation, and multi-object
   competition; no VOS-specific tracker is created.
