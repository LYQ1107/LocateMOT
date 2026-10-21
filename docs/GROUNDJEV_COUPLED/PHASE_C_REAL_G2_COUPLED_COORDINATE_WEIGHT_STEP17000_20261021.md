# GroundJEV Phase C: coordinate-weighted shared-backbone coupling

## Protocol

- Project root: `/data2/user/LocateMOT`
- Frozen G2 adapter: `adapter-step017000.pt`
- Base model SHA256: `3cf09355cde8cf4877cdc76e0a72301d572c1d6263b2831f8a03368285bb65e2`
- Training: resume the real two-sequence coupling checkpoint from step 48 through step 60
- Sequences: MOT20-03 frames 82–86 (target ID 27) and MOT20-05 frames 13–17 (target ID 17)
- Six train transitions, last frame of each sequence held out
- One GroundAnything visual encode per frame; 10 visual encodes total; no independent VLM
- G2 parameters frozen; trainable interface/readout parameters: 4,075,271
- Grounding coordinate-token loss weight: 5.0; JEV structured loss weight: 1.0
- Coupled checkpoint SHA256: `e24e96396e9a4ce7fc3be55de5737bd50f8fa0b6108dccd74f8411d0b8a64578`

## Result

The run completed successfully (`optimizer_step_start=48`, `optimizer_step_end=60`). The final teacher-forced losses were grounding `2.0984815558`, JEV `0.0007789339`, and total `2.0992604896`.

The structured JEV readout remained correct on both short sequences (MATCH/ADMIT/Yes-No/ACTIVE accuracy 1.0), but greedy causal C2 grounding stayed at mean IoU `0.0000` on both MOT20-03 and MOT20-05. This is a bounded real coupling result and does not establish MOT/RMOT/OVMOT improvement.

## Token diagnosis

The no-update trace used four additional G2 visual encodes. Coordinate-token mean loss improved under C2 memory from the C0 values `7.0573` to `2.5554` on MOT20-03 and from `8.2541` to `2.5421` on MOT20-05. The first coordinate remains the bottleneck: `<214>` and `<339>` are still not greedy (the model emits `<1>` or `<|box_start|>`), so valid downstream box geometry is not recovered. This supports changing the coordinate/output interface or curriculum rather than adding blind steps.

Formal G2 training was untouched. No official video-level metric claim is made.
