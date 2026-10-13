# Four-GPU Pilot

At most four physical GPUs across all project processes. Other projects
already occupy devices, so selecting a device does not claim exclusive use.
Use explicit task-local visibility and record actual peaks/latencies.

First validate actual GroundAnything DLM + OneJev on a legal MOT17/Dance clip,
then a full video. Fixed identical candidates enable Simple/Core/Full/SFT
comparisons. Next generate hard tracking decisions and train pretrained
OneJev with own-state rollout, checkpoint saving/reloading, before/after
official metrics. Add legal Refer-KITTI and TAO Base/Novel video loops.
Record FPS including perception and invalid/truncated outputs.

Pilot is incomplete until all 12 requirements in the user task are backed by
real results. Interactive prompting and paper-scale experiments follow later.


2026-10-14 update:legal RMOT short single/multi/empty loops and exact-candidate
Simple controls now have unchanged TrackEval metrics. Corrected GT-format audit gives4/5single-query coverage (V1 ignores it)
and5/30multi-query coverage; see CURRENT_STATUS for valid metrics. TAO
1203class registration,96-class chunking loop/TETA is active on physical5 with
preloaded DLM service6; formal500step training4 and fullDanceTrack6 continue.
Native DLM feature export,200step frozen-backbone temporal adapter and real
feature-memory sanity completed; calibrated retrieval equals unadapted native.
Final pilot still needs complete MOT video, formal500before/after, completed
OVMOT metrics, sufficient resource records and verified publication. Native
feature V5 finite check is active; Compact/retention/multitask research follows.
