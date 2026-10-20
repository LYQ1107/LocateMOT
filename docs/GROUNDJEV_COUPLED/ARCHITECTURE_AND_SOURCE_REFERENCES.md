# GroundJEV single-backbone coupling

This branch is the isolated Phase-A/Phase-B implementation for `LYQ1107/LocateMOT`.
The architecture has one GroundAnything G2 visual backbone.  It produces one
spatial feature map per frame; ROI tokens and (when the DLM hook is enabled)
contextual language hidden states are views of that same forward.  The typed
JEV readout consumes those tensors, the global solver performs one-to-one
assignment, and only final assignments can update the persistent identity
memory.

The current Phase-A bridge loads the protected G2 step-11000 projector into the
official GroundAnything visual implementation.  The language LoRA tensors in
the adapter checkpoint are intentionally not loaded by this visual-only bridge;
the receipt therefore says `language_lora_loaded_in_visual_bridge=false`.
This is an engineering gate, not a claim that the final DLM language path is
complete.  No OneJev, YesTrack, or other VLM weights are instantiated.

## Implemented contracts

- `SharedGroundState` binds frame id, image key, model SHA, boxes, grid and
  region tokens; it rejects non-finite or mismatched tensors.
- `G2FeatureBridge` runs exactly one native visual encoder call per frame and
  uses the official resize/grid geometry for region pooling.
- `G2JEVReadout` exposes dynamic MATCH/REACTIVATE/NEW/NO_MATCH/REJECT slots,
  admission, lifecycle and UPDATE/HOLD probabilities in one typed output.
- `global_one_to_one_assignment` keeps live identities globally unique while
  giving every candidate its own NEW/NO_MATCH/REJECT action column.  Thus two
  legitimate births in one frame remain representable.
- `LifecycleController` downgrades uncertain TERMINATE suggestions to LOST,
  rejects duplicate identity assignments, and keeps state serializable.
- `MemoryWriter` writes only after assignment and never writes REJECT or
  NO_MATCH observations.

## Fixed upstream references

The following public source versions were inspected.  Only small adapter logic
is implemented in this repository; upstream source is not copied into the
branch.

| Reference | Commit | Relevant file/blob | License |
|---|---|---|---|
| GroundAnything | `e054ab329cfe3f1b88ca8cf279bcab3e2f12934e` | `models/dlm/vlm.py` blob `861baa14eeb2e2e9a86c509e2f0d19fa6a827eda`; `models/dlm/hybrid.py` blob `577e21789598203f45066a3b95aad8db17939644` | Apache-2.0 (`LICENSE` blob `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64`) |
| OneJev | `6ce62b3a57a75deb60fde3bcc3dbaa4068c73650` | `qev/schema.py` blob `9c79cf9a86262b3394cfa1aecfdcea8ea8a09f30`; `qev/mm_engine.py` blob `554773a0b2ababbd2a547bcdbba9c05f155669ea` | Apache-2.0 (`LICENSE` blob `d645695673349e3947e8e5ae42332d0ac3164cd7`) |
| YesTrack | `ed42b8315b28c94dc595d83b0d49bf57002bb281` | `refer_llm/infer_utils.py` blob `007df9d9b39c372cc5e1e979cbecbb5122c2f6a3` | MIT (`LICENSE` blob `86361137835142cb033b4f56f8f2ab79fe9f232c`) |
| MOTIP | `ffc0e905ac196a603027eca8d18fb0dff48c8bcc` | `models/motip/id_decoder.py` blob `e79b207470901657531cc7f800daf2173c5f784f` | Apache-2.0 (`LICENSE` blob `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64`) |
| MeMOTR | `main` (MCG-NJU) | lifecycle/memory design reference only | MIT (`LICENSE` blob `8dab812e71086316b759d749f2aa0e42b9d162dc`) |
| SAM2 | `2b90b9f5ceec907a1c18123530e92e794ad901a4` | `sam2/modeling/sam2_base.py` blob `d9f4e515b0d161942bf2bb64560056b3efbe6dac` | Apache-2.0 (`LICENSE` blob `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64`) |

OneJev's shared-prefix and MOTIP's dynamic ID concepts were used as interface
references, not as model weights.  SAM2/MeMOTR memory ideas are represented by
the finite causal `IdentityMemoryStore`, not by importing their tracking
networks.

## Remaining work

The Phase-A bridge does not yet inject identity memory into GroundAnything's
generation prefix or expose full DLM contextual hidden states.  That is the
registered Phase-C change and will be implemented behind an opt-in input while
preserving `memory_enabled=False` behavior.
