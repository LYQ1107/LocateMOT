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
