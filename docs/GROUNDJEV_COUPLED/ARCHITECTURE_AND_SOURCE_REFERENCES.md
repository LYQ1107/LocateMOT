# GroundJEV coupled architecture and source audit — 2026-10-21

The LocateMOT implementation is an adapter layer. It does not copy an entire
upstream repository and it does not load OneJev or YesTrack online. The formal
G2 Clean optimizer remains in its separate worktree and run directory.

## Current architecture

`G2FeatureBridge` calls one injected GroundAnything feature provider for a
frame and creates a `SharedGroundState` carrying frame id, image key, model
SHA, boxes, region tokens and optional contextual tokens. `G2JEVReadout`
projects those existing G2 region tokens into typed Identity Choice, Admit,
Yes/No, Lifecycle and Memory heads. `DynamicIdentitySlots` builds options from
causal tracker state and reserves explicit `NEW`, `NO_MATCH` and `REJECT`
actions. `CoupledGroundJEVRuntime` performs one-to-one assignment, safe
lifecycle transitions and assignment-conditioned memory writes. There is no
second VLM instance in this path.

The bridge now also has a native clean-DLM provider path.  The bounded Phase-C
smoke loaded a protected G2 Clean adapter, ran one clean visual-plus-language
forward, returned both pooled region tokens and same-frame clean context, and
fed that context to the typed readout without a second VLM.  The opt-in causal
memory input contract is implemented and has baseline-equivalence and
gradient-path tests.  Full SGLang integration and a trained decoder that uses
memory before GroundAnything bbox generation remain explicit next gates; they
are not inferred from the native context smoke.

## Sources actually inspected

| Source | Commit used for inspection | License | Relevant code read | LocateMOT adaptation |
|---|---|---|---|---|
| [GroundAnything](https://github.com/groundingpi/GroundAnything) | `8a3b43a1b42952e117960e85d63ff173bb9fba0a` in the isolated runtime (remote HEAD resolved as `e054ab329cfe3f1b88ca8cf279bcab3e2f12934e`) | Apache-2.0 | `models/dlm/vlm.py` (`GAMQwen3DLM._embed_clean`, `_packed_hybrid_language_forward`); `models/dlm/hybrid.py` (`DLMOutput`, `_embed_clean`, `_build_streams`, `forward`); `models/vlm/modeling_groundinganything.py` vision model | `locatemot/groundjev/native_features.py`, `coupled/g2_feature_bridge.py`, `coupled/shared_ground_state.py`; only an adapter, no upstream source copied |
| [OneJev](https://github.com/OmniJev/OneJev) | `2470c7595688c6a402560cf18b4753feb1fe290a` in the isolated runtime (remote HEAD resolved as `6ce62b3a57a75deb60fde3bcc3dbaa4068c73650`) | Apache-2.0 | `qev/schema.py` typed Noul/Choice/Score questions; `qev/mm_engine.py::_readout_rows`; `train/shared.py::split_prefix`, `branch_masks`, `shared_readout_logits`, `render_group` | `coupled/dynamic_identity_slots.py`, `coupled/g2_jev_readout.py`, `coupled/typed_decisions.py`; concepts adapted, OneJev weights/engine are not loaded |
| [YesTrack](https://github.com/ggbondrighthere24/YesTrack) | remote HEAD `ed42b8315b28c94dc595d83b0d49bf57002bb281` | MIT | Repository identity/Yes-No supervision reference recorded; no YesTrack model is deployed by GroundJEV | Phase B `yesno` head and existing native Yes/No corpus; binary auxiliary supervision is an explicit GroundJEV design |
| [MOTIP](https://github.com/MCG-NJU/MOTIP) | remote HEAD `ffc0e905ac196a603027eca8d18fb0dff48c8bcc` | Apache-2.0 | Dynamic identity-choice problem and causal ID-slot concept | `dynamic_identity_slots.py` and typed identity targets; no MOTIP decoder copied |
| [SAM 2](https://github.com/facebookresearch/sam2) | remote HEAD `2b90b9f5ceec907a1c18123530e92e794ad901a4` | Apache-2.0 | Memory-conditioned prediction is the design reference | `identity_memory.py` and `memory_tokens.py`; no SAM model loaded |
| [MeMOTR](https://github.com/MCG-NJU/MeMOTR) | remote HEAD `eb7a177b9cbcb89742ec69b2545ab3af2ea31a80` | MIT | Persistent track update/lifecycle reference | `lifecycle_controller.py`, `memory_writer.py`; no association Transformer copied |
| [MOTR](https://github.com/megvii-research/MOTR) | remote HEAD `8690da3392159635ca37c31975126acf40220724` | MIT | Track query lifecycle reference | Same lifecycle/memory adapter boundary; no MOTR code copied |

The remote HEAD values were resolved directly on 2026-10-21. The GroundAnything
and OneJev runtime checkouts are intentionally pinned to the commits used by
the existing local G2/diagnostic environment; changing them would invalidate
the current feature provenance.
