# GroundJEV active goal snapshot — G2 step 12195 — 2026-10-21

The protected formal G2 Clean owner has reached event step `12195/18227` on physical GPUs 3/5. The latest durable checkpoint is `adapter-step012000.pt` and `latest_training_state.pt` is present; no terminal receipt exists. The last 500 steps average `11.618 s/step`, leaving about `19.5 h` by linear extrapolation. The registered G2 manifest has 583,251 TRAIN rows and SHA `f682519a…06e71`.

G0 MOT20-05 remains terminal for its complete registered 1,657-frame validation-half natural-visibility scope. Its fresh native-source control reports HOTA `0.073346961`, DetA `0.090269828`, AssA `0.059674271`, IDF1 `0.059598451`, IDSW `1933` and native end-to-end FPS `0.000000000`. This is a validation-half control, not a full MOT20 TEST claim.

G1 GPU5, clean bundle export, independent held-out evaluation, and native YesTrack remain registered dependency/resource waits with no recorded failure. The compact three-task OneJev queue is paused before any optimizer update (`optimizer_updates=0`): its current 131,990-row manifest contains MOT/RMOT native rows plus OVMOT Base teacher-control rows, while native OVMOT TRAIN candidate/annotation rows are still absent. The queue guard now refuses optimizer launch until the manifest explicitly declares native OVMOT training mode.

The previous process audit found no live MOT-only optimizer; that audit was not repeated for this snapshot. This file records runtime evidence and gate state, not a performance claim for G2, G1, OneJev, YesTrack, RMOT or OVMOT.
