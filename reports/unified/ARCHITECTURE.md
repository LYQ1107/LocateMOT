# LocateMOT-U architecture contract — U0

`UnifiedSpecTrack` is the composition root for one release checkpoint. Its
forward contract is:

```python
outputs = model(
    frames=frames,
    task_type=task_type,
    specification=specification,
    online_state=online_state,
)
```

The shared path is:

```text
frames
  -> SharedVisualFoundation (MM-GroundingDINO-B target)
  -> UniversalPerceptionDecoder
  -> UniversalTrackDecoder
  -> shared online state / persistent track memory
```

The specification path is:

```text
task + text/class/prompt
  -> SpecificationEncoder
  -> SemanticRouter(static, motion, relation, global)
  -> ConditioningHook / MotionReasoner / RelationReasoner
  -> Reliability CueFusion
  -> task head
```

Ordinary MOT, OVMOT, RMOT, and prompt tracking therefore use the same visual
foundation, track queries, identity dynamics, track memory and online-state
schema. Task adapters and heads only condition or decode the shared state.
They may not own a second backbone or a second tracker.

The current `locatemot/unified/models/` implementation is a shape-checking U0
skeleton. `FoundationConfig` records the MM-GroundingDINO-B target, while the
small convolutional fallback is used only for import/CPU smoke. No training
result or final architecture claim is attached to that fallback.

The staged implementation order is U1 foundation/proposal grounding, U2
shared MOT/OVMOT track core, U3 RMOT semantic conditioning, U4 prompts, and U5
joint consolidation. Proposal and expression grounding gates precede any
tracking training.
