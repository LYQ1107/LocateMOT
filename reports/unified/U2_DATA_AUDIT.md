# LocateMOT-U U2 data audit

Recovery snapshot: 2026-10-11. Authoritative inventory:
`outputs/unified/u2/protocol/data_manifest.json`.

| Dataset | Status | Registered scope / acquisition |
|---|---|---|
| bdd100k | MISSING_LOCAL_DATA | Cloud folder contains static detection resources; original MOT 2020 train/val image downloads running; tracking labels still required |
| dancetrack | MISSING_LOCAL_DATA | Official Hugging Face train1/train2/val acquisition running; existing 32/8/25 protocol preserved |
| mot17 | AVAILABLE | 2,664 train / 2,652 dev frames; FRCNN copies only; exact half protocol frozen |
| mot20 | AVAILABLE | 4,468 train / 4,463 dev frames; exact half protocol frozen |
| kitti_legal_development | AVAILABLE | Existing legal U1 development scope and reference retained |
| tao | INCOMPLETE_LOCAL_DATA | Four-volume cloud acquisition and CRC-checked extraction running; official modal train/val coverage audited |
| ctao | INCOMPLETE_LOCAL_DATA | Base-only annotation downloaded and audited; complete training-frame coverage still pending |

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
ineligible both because required inputs are incomplete and because measured MOT gates failed.

C-TAO base-only annotation audit: 490,210 training frames, 1,489,637 boxes, 500 videos; class-supervised boxes are f/c only (1,378,391 frequent, 111,246 common). All image paths start with train/. TAO validation has 36,375 annotated frames and 113,112 GT boxes. Its v0.5→v1 synset mapping leaves 314 boxes in six categories unmapped; retain this explicit count when registering full portability and official TETA conversion, rather than silently dropping them.
