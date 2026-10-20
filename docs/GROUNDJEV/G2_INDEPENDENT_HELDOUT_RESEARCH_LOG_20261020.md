# GroundJEV independent held-out queue addendum — 2026-10-20

- Registered a video-disjoint held-out gate before the four-GPU OneJev resume.
- Scopes: Refer-KITTI v2 sequence 0007 single/multi (800 frames each) and TAO-Amodal validation video 105 (40 frames).
- The streaming exact-path audit against the 583,251-row G2 Clean TRAIN corpus found zero overlap in all three scopes. TAO video105 contains Base categories baby/bowl/bun and Novel category shaker.
- Added `prepare_independent_ovmot_scope.py` and `queue_g2_independent_heldout_eval.py`. The queue waits for the terminal G2 Clean bundle, runs G0 and G2 with identical parser/decoder/Simple/scorer contracts on GPU7, and records official metrics, rejection counts, output counts and FPS.
- Updated `queue_onejev_full_joint_resume.py` so OneJev waits for this held-out owner to reach a terminal state before claiming GPUs 4–7. A held-out failure is retained as evidence and does not deadlock the main training objective.
- No final held-out metric is claimed until the owner writes `G2_CLEAN_FULL_INDEPENDENT_HELDOUT_COMPLETE` and `comparison.json`.
