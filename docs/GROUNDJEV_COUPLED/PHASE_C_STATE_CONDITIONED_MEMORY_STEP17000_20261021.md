# GroundJEV Phase C: identity plus previous-box state interface

This bounded experiment kept the official G2 step17000 adapter frozen and
projected the previous-frame visual summary together with the previous legal
box `(x1,y1,x2,y2)/999` into the existing causal G2 prompt. The same shared
G2 image summaries fed the typed JEV readout. No independent VLM or MOT
association network was used.

- Sequences: MOT20-03 frames 82–86 (ID 27) and MOT20-05 frames 13–17 (ID 17)
- Six train transitions, last frame held out in each sequence
- 12 optimizer steps, coordinate-token loss weight 5.0
- 10 G2 visual encodes, 4,075,271 trainable interface/readout parameters
- G2 base SHA256: `3cf09355cde8cf4877cdc76e0a72301d572c1d6263b2831f8a03368285bb65e2`
- Coupled checkpoint SHA256: `72595d566a460fe9c595cfe8751249b2ddd5ffd61633c1b4fbfef004237229c1`

The final teacher-forced grounding loss was `1.70039` and typed JEV
MATCH/ADMIT/Yes-No/ACTIVE accuracy was 1.0 on both short sequences. Greedy
C2 grounding nevertheless remained mean IoU `0.0000` on both sequences. The
first coordinates stayed at the frozen G2 baseline (`<1>`/`<0>`), so adding a
numeric box state to one memory token is insufficient to control the causal
coordinate head. This is a real coupling negative result, not a tracking metric.
