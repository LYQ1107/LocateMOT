# OneJev TrackJEV MOT20-05 step8227

- Scope: complete registered MOT20-05 validation scope, 1,657 frames.
- Model: merged native OneJev three-task checkpoint `merged-final-step8227`.
- Candidate input: frozen independent G2 Clean replay; runtime did not read ground truth.
- Official TrackEval metrics: HOTA 0, DetA 0, AssA 0, IDF1 0, IDSW 0, FP 0, FN 319,062.
- Perception rejections: 65/1,657 (3.92275%).
- Association-replay FPS: 0.0779955; full wall time: 21,265.44 s.
- Interpretation: no accepted tracks were produced, so this is a negative TrackJEV result and does not support a JEV improvement claim.

For comparison, the earlier G2 Clean + fixed Simple-tracker MOT20-05 result improved HOTA from 34.2615% to 44.4739%, IDF1 from 29.0533% to 40.0865%, and reduced IDSW from 13,712 to 8,464. That improvement is attributed to G2 perception, not the final JEV decoder.
