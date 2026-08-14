# Stage L8 — Refer-Dance / RMOT Protocol Audit

Date: 2026-08-14
Project: LocateMOT (`/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`)

## 1. Task definition (RMOT)

Referring multi-object tracking (RMOT) is defined by Wu et al. (CVPR 2023,
"Referring Multi-Object Tracking", arXiv:2303.03366): given a video and a
natural-language expression, output all object trajectories matching the
expression. It is multi-target: an expression may refer to zero, one, or
many objects.

## 2. Official protocol source

- RMOT paper/repo: https://github.com/wudongming97/RMOT
  (official TransRMOT implementation + Refer-KITTI + TrackEval RMOT runner)
- Refer-Dance dataset contributed by iKUN:
  "iKUN: Speak to Trackers without Retraining" (Du et al., CVPR 2024,
  arXiv:2312.16245, https://github.com/dyhBUPT/iKUN)

Refer-Dance construction per iKUN paper §4.1:

- 40 videos with 39 distinct descriptions for training;
- 25 videos with 17 distinct descriptions for testing;
- descriptions focus on motion and dressing status (e.g., "dancing person
  with black T-shirt and green pants");
- metrics: HOTA series (HOTA / DetA / AssA), MOTA, IDF1, evaluated with
  TrackEval (`run_mot_challenge.py`, METRICS=HOTA, threshold=0.5).

## 3. Released artifacts in the local Refer-Dance zip

Source zip:
`/data3/testdata/vranlee/.MOTSynth.partial/Refer-Dance/1665795909_薛定谔的小旺财/Refer-Dance.zip`

Contents verified by listing + extraction (2026-08-14):

| Entry | Count / size | Notes |
|---|---|---|
| `expression/` | 2051 json files | 40 train seqs × 39 expressions + 25 val seqs × 17 expressions (1985 total files with sentence+label; remaining entries are dirs) |
| `gt_template/` | 425 `gt.txt` (25 seqs × 17 expressions) | Only **40** non-empty per-expression GT files; the other 385 are empty placeholders |
| `labels.json` | 1 file | per-seq per-object-id per-frame `bbox` (normalized xywh), `expression_raw`, `category` |
| `seqmap.txt` | 425 lines | format `<video>+<expression>`; the 25 val sequences, each × 17 expressions |
| `videos/train`, `videos/val` | 40 + 25 names | mp4 copies of the same DanceTrack sequences (not needed for evaluation; removed locally) |
| `DanceTrack/training/image_02/` | 65 sequences | identical to official DanceTrack train+val image frames |
| `DanceTrack/labels_with_ids/image_02/` | per-frame txt | `class id x1 y1 x2 y2` normalized (verified against `labels.json`) |

Note: the zip is stored under a path containing `.MOTSynth.partial`, but the
content is the public Refer-Dance dataset (DanceTrack images + referring
annotations), not MOTSynth-generated data. We verified the file names,
JSON schemas and the iKUN paper description. MOTSynth data is not used.

## 4. Expression / GT semantics

- `expression/<seq>/<expression>.json`:
  `{"label": {frame_str: [object_id, ...]}, "sentence": "..."}`.
  `label` maps a frame to the object ids that match the expression in that
  frame. Empty `label` means the expression has no target in that video.
- `labels.json[seq][object_id][frame_str]`:
  `{"bbox": [x_norm, y_norm, w_norm, h_norm], "expression_raw": [...],
  "category": [...]}`.
- `gt_template/<seq>/<expression>/gt.txt`: TrackEval MOT-format per-expression
  GT rows: `frame, id, bb_left, bb_top, bb_width, bb_height, conf=1, class=1,
  visibility=1`. Pixel coordinates are on a **1920×1080** canvas
  (verified: normalized bbox × 1920/1080 equals the gt.txt row exactly),
  while DanceTrack images are 960×540.
- Object ids in `expression label` and in `labels.json` are the same
  DanceTrack GT ids used by the L6/L7 candidate manifests
  (`gt_ids` / `gt_boxes` / `matched`), so per-frame candidate targetness can
  be computed directly.

## 5. Evaluation layout (verified from official TrackEval RMOT variant)

`TrackEval/trackeval/datasets/mot_challenge_2d_box.py` in
`wudongming97/RMOT` (commit `d4fedb35`) handles RMOT as follows:

- seqmap line format: `<video_id>+<expression_id>` (single field);
- GT file: `{gt_folder}/{video_id}/{expression_id}/gt.txt`;
- tracker file: `{trackers_folder}/{video_id}/{expression_id}/predict.txt`
  (the official code expects GT and predictions to coexist under the same
  tracker folder; iKUN's `generate_final_results` symlinks `gt.txt` and
  writes `predict.txt` accordingly);
- sequence length: if the sequence name contains "MOT", read `seqinfo.ini`;
  otherwise it uses a hardcoded Refer-KITTI image path. **This hardcoded
  path must be patched for Refer-Dance** (we use a patched copy in
  `references/l8/TrackEval_rmot`, see below).

## 6. Official published Refer-Dance results (for protocol-comparable baselines)

From iKUN paper (CVPR 2024, arXiv:2312.16245), Refer-Dance comparison
(Figure 5):

| Method | Detector | HOTA | DetA | AssA |
|---|---|---|---|---|
| TransRMOT | Deformable-DETR | 9.58 | 4.37 | 20.99 |
| iKUN | ByteTrack+NKF | 29.06 | 25.33 | 33.35 |

The iKUN detection pipeline is ByteTrack+NKF on DanceTrack person
detections, not LocateAnything. Our LocateAnything candidate set is a
different detector input; external comparisons are therefore indicative
and must be reported with this protocol caveat.

## 7. LocateMOT L8 evaluation plan

1. Run the shared UIDM (with Unified Observation Token and RMOT spec
   conditioning) over the same `dancetrack_val` candidate manifest used in
   L6/L7 (`outputs/l1_c/fixed_candidate_manifest/dancetrack_val.jsonl`,
   25 videos, 25,508 frames, 218,580 candidates, LocateAnything-3B person
   detections).
2. For each query `(video, expression)` in `seqmap.txt`, keep UIDM outputs
   whose candidate relevance score passes a threshold, write
   `predict.txt` in TrackEval MOT format (pixel boxes on the 1920×1080
   canvas, i.e., candidate boxes from the manifest already are on this
   canvas for the val split).
3. Run the patched official TrackEval RMOT runner with `METRICS=HOTA`,
   `THRESHOLD=0.5`, `SEQMAP_FILE=seqmap.txt`, GT layout as above.
4. Report HOTA / DetA / AssA / DetRe / DetPr / AssRe / AssPr / LocA
   (HOTA metric output), plus MOTA and IDF1 (CLEAR/Identity metrics) if
   produced by the same runner.

Queries with empty `gt.txt` (385/425) contribute no GT and are excluded by
the evaluator when GT is empty; the paper's numbers correspond to the 40
queries with targets. We additionally report the subset of 40 GT queries
explicitly.

## 8. Patches / local changes to the official evaluator

- Local copy: `references/l8/TrackEval_rmot` (copied from
  `wudongming97/RMOT/TrackEval` at commit `d4fedb35`).
- Patch 1: replace the hardcoded Refer-KITTI image path in
  `mot_challenge_2d_box.py` with a configurable `IMG_ROOT` so Refer-Dance
  sequence lengths are computed from the local image folders.
- No metric formula changes.

