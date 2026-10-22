# AnimalTrack scorer regression — 2026-10-22

The runtime `evaluate_video.py` was exercised on a one-frame real AnimalTrack
scope using the official `chicken_1` GT and an empty prediction file. TrackEval
completed with `OFFICIAL_SCOPE_EVALUATION_COMPLETE` and produced HOTA/IDF1=0,
FN=20. This is a parser/scorer contract regression only, not a model result.
The full 306-frame GPU owner remains the only source of AnimalTrack performance.
