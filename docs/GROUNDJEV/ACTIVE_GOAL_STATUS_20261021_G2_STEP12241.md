# GroundJEV active goal snapshot — G2 step 12241 — 2026-10-21

Protected G2 Clean is at `12241/18227` on GPUs 3/5. The latest durable checkpoint remains `adapter-step012000.pt`; the terminal receipt is absent. The last 500 steps average `11.626 s/step`, leaving about `19.3 h` by linear extrapolation.

G0 MOT20-05 is terminal for its registered 1,657-frame validation-half scope: HOTA `0.073346961`, DetA `0.090269828`, AssA `0.059674271`, IDF1 `0.059598451`, IDSW `1933`, native FPS `0.061369784`. It is not a full TEST result.

G1, bundle, held-out and YesTrack owners remain dependency/resource waits. The compact three-task OneJev owner is paused before optimizer launch (`optimizer_updates=0`) because the registered manifest still has OVMOT teacher-control rows and no native OVMOT TRAIN rows.

This is runtime evidence, not a performance claim for G2, G1, OneJev, YesTrack, RMOT or OVMOT.
