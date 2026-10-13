# LocateMOT-U2 status

Updated 2026-10-13. This file records the public, reproducible status of the
U2 shared persistent track-core run. Datasets, checkpoints, caches, and raw
video/ground-truth files stay outside the repository.

## Completed

- The epoch-1 perception foundation was selected after the full unchanged
  TAO/CTAO portability audit and the U1 retention checks. The selected
  foundation digest is
  `1d9077ae603f40de0e955a11a88083f914f98faa967bc46cdd06bcdfb5129688`.
- TAO measured 36,375 images: base recall `0.9742855845` and novel recall
  `0.9460255152` (2,892 / 3,057 novel hits). CTAO measured 17,131 images with
  base recall `0.9729080225`.
- The selected foundation's formal generic scope is 4,201 images with
  Top-150 recall at IoU .5 `0.9787925789`. The expression audit covers 7,160 query frames
  and 7,199 target references; V1/V2 Top-100 recall is `1.0` and
  `0.9785374828` (mean query AP50 diagnostics `0.7237566558` / `0.6556441180`). The fixed 500-pair language counterfactual audit passed
  (`0.7301285093` true versus `0.4067468580` shuffled AP50).
- The five ordinary MOT perception streams are complete. Their Top-150
  coverage is BDD `0.9365738211`, Dance `0.9990361895`, MOT17 `0.8780664316`,
  MOT20 `0.8191362249`, and legal KITTI `0.9463937008`.

## Historical execution, stopped by user research pivot

The selected epoch-1 foundation was building a GT-free ordinary candidate cache
for approximately 396,865 frames. The cache stores reusable frame-level
features and candidate proposals for the simple and learned tracking stages;
it prevents each later tracker experiment from rerunning the detector/backbone
and keeps all tracker comparisons on identical inputs. The cache is being
built on GPUs 0--7 by the existing supervisor. Formal tracking training has
not started yet.

## Abandoned U2 stages (do not launch)

1. Validate the complete ordinary cache and run the simple persistent-track
   baseline.
2. Run the registered 6,000-step MOT-only training/evaluation and apply the
   ordinary retention gates.
3. Launch the joint OVMOT stage only if the MOT-only result is eligible.
4. Run causal retention, checkpoint self-containment, and final packaging
   gates. U3 readiness is reported only if every registered gate passes.

The current state is `STOPPED_BY_USER_RESEARCH_PIVOT`. The 2026-10-13 user
task replaces this route with GroundJEV-MOT. Verified owners exited and feature
shards were removed; datasets/checkpoints/results remain. See
`docs/GROUNDJEV/RESOURCE_CLEANUP.md`. No final U2 tracking score or U3-ready claim.
