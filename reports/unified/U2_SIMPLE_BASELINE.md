# LocateMOT-U U2 simple baseline

The causal IoU/motion/identity-cosine Hungarian runtime and full five-dataset
evaluation driver are implemented. Fixed pre-benchmark registration is
`protocol/simple_baseline.json`: weights .25/.35/.40, match margin .45,
center-distance cap 2, birth score .10, appearance EMA .90 and max_age 30.
Every cached candidate enters association; low-score observations can recover
existing IDs. No learned model or category feature enters identity association.
Lost tracks are retained causally and no unobserved motion box is emitted.

Unmodified upstream TrackEval is pinned at
`12c8791b303e0a0b50f753af204249e622d0281a`; the expression-modified TempRMOT
reader is not used. HOTA/DetA/AssA use all 19 official alpha thresholds;
sequence combination and class averaging use official implementations. BDD raw
aliases, inclusive box extents and ignore attributes follow the already verified
toolkit mapping; original KITTI IDs and visibility metadata are retained.

The synthetic contract passed all five public file readers and metrics:
perfect tracks score 1 with IDSW 0; an ID switch lowers AssA and records IDSW 2.
It also validates coincident-box appearance identity, occlusion without ghost
output, save/reload and rejection of incomplete F0 cache eligibility.
This is plumbing evidence, not benchmark performance. The formal baseline has
not launched. A blocking queue waits for complete perception selection before
building the ordinary cache and evaluating all five complete dev sets.

F0 MOT17/MOT20 Top150 gates fail; BDD, original KITTI and TAO/C-TAO full audits
remain pending. No benchmark HOTA, DetA, AssA, IDF1 or IDSW is claimed.
