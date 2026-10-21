# GroundJEV Phase C: native visual-prompt memory diagnostic

The official GroundAnything `build_visual_prompt` protocol was evaluated as a
causal identity-memory interface. For each current frame, the previous legal
box was rendered as a native reference-box prompt; the current image was
encoded exactly once. The formal G2 step17000 adapter stayed frozen.

- MOT20-03 frames 82–86, target ID 27
- MOT20-05 frames 13–17, target ID 17
- 10 visual encodes, zero independent VLM instances, zero parameter updates
- G2 base SHA256: `3cf09355cde8cf4877cdc76e0a72301d572c1d6263b2831f8a03368285bb65e2`
- Receipt SHA256: `78c94d13afbe4b49ed392e97a7175e3cf65ebc7643e3f5d011dd1cf25ad4b35a`

All four C2 frames in each sequence returned legal parsed boxes, but C2 mean
IoU was `0.0000` for MOT20-03 and MOT20-05. The visual prompt changed labels
and some token paths but did not repair the frozen coordinate geometry. This
is a bounded protocol diagnostic and does not establish MOT/RMOT/OVMOT gain.
