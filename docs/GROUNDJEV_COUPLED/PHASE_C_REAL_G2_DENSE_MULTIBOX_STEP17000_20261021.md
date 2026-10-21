# GroundJEV Phase C: corrected dense multi-box G2 contract

The earlier one-box pilot rows were not a valid dense G2 contract: they asked
for all persons but supplied one target box as the answer. This diagnostic
keeps every annotated person box in the answer and uses a multi-box causal
decoder that constrains each four-coordinate tuple and stops only at
`<|im_end|>`.

- Frozen G2 adapter: step 17000
- Base SHA256: `3cf09355cde8cf4877cdc76e0a72301d572c1d6263b2831f8a03368285bb65e2`
- MOT20-03 and MOT20-05, three real frames each
- Six visual encodes, no independent VLM, no parameter updates
- G2 parameters untouched

The model returned a legal first box followed immediately by `<|im_end|>` on
all six frames. Mean recall at IoU 0.5 was `0.01035` on MOT20-03 and `0.00404`
on MOT20-05. This is a corrected dense-front-end baseline, not a tracking or
coupling gain. Future coupled Grounding updates must retain the full dense
answer contract or use an explicitly valid referring/visual-prompt task.
