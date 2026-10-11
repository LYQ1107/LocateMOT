# LocateMOT-U U2 data audit

Recovery snapshot: 2026-10-12. Authoritative inventory:
`outputs/unified/u2/protocol/data_manifest.json`.

| Dataset | Status | Registered scope / acquisition |
|---|---|---|
| bdd100k | INCOMPLETE_LOCAL_DATA | All 1,400/200 raw tracking annotation videos and 39,973 validation images recovered and validated; original train-image extraction remains incomplete |
| dancetrack | AVAILABLE | All three official archives downloaded and CRC-extracted; existing 32/8/25 videos and 33,772/8,024/25,508 frames match exactly |
| mot17 | AVAILABLE | 2,664 train / 2,652 dev frames; FRCNN copies only; exact half protocol frozen |
| mot20 | AVAILABLE | 4,468 train / 4,463 dev frames; exact half protocol frozen |
| kitti_legal_development | AVAILABLE | Existing legal U1 development scope and reference retained |
| tao | AVAILABLE | Every modal train/val annotated image exists; four-volume acquisition continues for remaining release resources |
| ctao | AVAILABLE | Base-only annotation audited; all 490,210 continuous training frames exist |

TAO archives and extracted frames stay on their registered /data0..3/user volume;
the project data directory contains a symlink view. An existing or partially populated
directory does not establish completion: TAO train/val annotation image coverage and
C-TAO continuous training image coverage are required.

The official DanceTrack source is [noahcao/dancetrack](https://huggingface.co/datasets/noahcao/dancetrack)
(CC-BY-4.0), pinned to `a0ba42ac690c41e9850a20e76a0b9450a6fb6a47`.
The base-only C-TAO source is [COVTrack](https://github.com/zekunqian/COVTrack)
at model release `a6c1a4673a45425d9375f75d1427df7d22590837`; no external tracker weights are used.
Source file sizes, completed hashes and download state are retained under the U2 acquisition directory.
The LVIS v1 vocabulary contains all 1,203 categories with original frequency metadata, sourced from
[Detectron2 category metadata](https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/datasets/lvis_v1_categories.py).

Direct networking is used. aria2 failed its TLS handshake before receiving bytes; an OpenSSL
curl range request succeeded (HTTP 206), and the official data download resumed with curl.

No Refer-KITTI official evaluation or benchmark test labels were parsed. Track training remains
ineligible because BDD tracking inputs are incomplete and measured MOT gates failed.

C-TAO base-only annotation audit: 490,210 training frames, 1,489,637 boxes, 500 videos; class-supervised boxes are f/c only (1,378,391 frequent, 111,246 common). All image paths start with train/. TAO validation has 36,375 annotated frames and 113,112 GT boxes. Its v0.5→v1 synset mapping leaves 314 boxes in six categories unmapped; retain this explicit count when registering full portability and official TETA conversion, rather than silently dropping them.

Frozen portability scopes now cover all 36,375 TAO validation images and all 17,131
original C-TAO training-audit images. The interpolated C-TAO images remain in the
490,210-frame identity-training scope. Full TAO/C-TAO measurement runs on GPUs 1–6;
full DanceTrack validation runs on GPU 7. Separate real-image smoke runs passed with
the immutable U1R2 foundation. Full-run recall remains unreported until exact coverage completes.

The validated BDD annotation mirror is
[jaffe03195/bdd100k_sot](https://huggingface.co/datasets/jaffe03195/bdd100k_sot/tree/694197099692fe1254e93215e73e68ece8b1f41f),
archive `bdd100k.zip`, SHA `6f5b9d47837925a64f1280af588b05800392a2ab163e7c6110195d70c13781a5`.
Its ZIP directory contains raw `labels/box_track_20/train` (1,400 JSON videos),
raw `labels/box_track_20/val` (200 JSON videos), and validation images. Only these
original directories are selected. Converted annotations, filtered labels and SOT labels
are excluded from acquisition output. Archive SHA and selected-member CRC passed;
every validation image's size and CRC match the original release archive (MD5
`743ab4c7b5ff8eeebd60bbfd15b20bd6`). Original category aliases, ignored/crowd objects
and inclusive Box2D coordinates match the cached BDD toolkit and Scalabel functions
on every validation object. There are 442,814 positive GT units, 18,625 crowd objects
and 4,073 objects from official ignored categories. All eight positive categories are present.
The registered training/validation scope contains 278,079/39,973 frames. Complete
training image coverage is still required for overall data availability. The BDD full
validation audit is queued after the DanceTrack GPU worker and includes a real-image smoke.
