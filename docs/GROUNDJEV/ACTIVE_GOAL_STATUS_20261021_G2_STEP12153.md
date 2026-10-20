# GroundJEV active goal snapshot — G2 step 12153 — 2026-10-21

The protected formal G2 Clean owner is alive on physical GPUs 3/5 and has reached event step `12153/18227`. The latest durable checkpoint remains `adapter-step012000.pt`; the terminal receipt is absent. The recent event window is approximately `11.614 s/step`, leaving about `19.6 h` by linear extrapolation. The G2 manifest is the registered 583,251-row TRAIN corpus with SHA `f682519a…06e71`.

G0 MOT20-05-05 validation-half export is terminal for all 1,657 registered frames. Its native compact control reports HOTA `0.073346961`, DetA `0.090269828`, AssA `0.059674271`, IDF1 `0.059598451` and IDSW `1933` at `complete_registered_validation_scope_natural_visibility_challenge`. This is a validation-half control, not a full MOT20 TEST claim.

The active owner states are dependency waits with no recorded failure: G1 GPU5 is waiting for G2 success; the G2 bundle waits for G2 and G1 terminal receipts; independent held-out waits for the bundle and exclusive GPU7; native YesTrack waits for exclusive GPU7; the zero-drop three-task compact OneJev queue waits for G2/G1/bundle/held-out and exclusive GPUs 4/5/6/7, with zero optimizer updates so far. The historical v1 joint queue is superseded and the old independent four-GPU resume is paused.

The authoritative process audit found no live MOT-only optimizer. The protected G2 trainer and its two ranks were not modified.

This snapshot is runtime evidence, not a performance claim for G2, G1, OneJev, YesTrack, RMOT or OVMOT.
