# GroundJEV fixed-candidate cross-sequence diagnostic — MOT20-05

The fixed-candidate readout trained on the MOT20-03 TRAIN repair split was
frozen and evaluated without updates on a different MOT20-05 TRAIN clip
(frames 766–893, 128 frames). The same native candidate and region-feature
contract was used; the run made 128 cached feature calls and loaded no second
VLM.

TrackEval reports:

| HOTA | DetA | AssA | IDF1 | IDSW | FP | FN |
|---:|---:|---:|---:|---:|---:|---:|
| 0.115666 | 0.147839 | 0.090593 | 0.071458 | 341 | 19 | 21,407 |

The replay produced 3,916 predictions on 126 frames at 9.06 association FPS.
The earlier MOT20-03 temporal repair receipt was HOTA 0.302815 / IDF1
0.257729 / IDSW 18, so the cross-sequence drop exposes readout and proposal
distribution generalization failure. Both clips are registered TRAIN scope and
the native feature cache predates the terminal G2 Clean checkpoint; this is a
diagnostic, not an independent G2 Clean or MOT20 TEST result.

The TrackEval process printed an unrelated BURST/`pycocotools` import warning,
but the requested `MotChallenge2DBox` HOTA/CLEAR/Identity evaluation completed
successfully. Full provenance is in the paired JSON receipt.
