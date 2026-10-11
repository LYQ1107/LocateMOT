# LocateMOT-U U2 architecture

U2 replaces the API-only skeleton with `locatemot/unified/models/u2/`:

- `PerceptionBridge` keeps query-independent `id_visual`, semantic visual,
  decoder query, boxes, and scores separate.
- `CandidateEncoder` has independent identity and semantic projections.
- `UnifiedTrackState` stores IDs, identity memory, permanent anchors,
  appearance EMA, semantic memory, boxes, velocity, lifecycle counters,
  confidence, alive flags, and monotonic `next_track_id`.
- `SetInteraction` is four pre-norm TransformerEncoder layers (hidden 256,
  eight heads, FFN 1024, dropout 0.1) over identity tokens only.
- `AssociationHead` emits pair, no-match, and new logits.  `LifecycleHead` and
  `MotionHead` provide causal safety and normalized motion evidence.
- `SharedTrackCore` updates a persistent GRU state and applies Hungarian
  one-to-one matching with zero decision margins.  Lost tracks are retained up
  to `max_age=30` but are not emitted without a current candidate.
- `OVMOTAdapter` consumes semantic memory, candidate semantic features, and
  frozen U1 text features only after raw identity logits.  Its sigmoid gate is
  exactly zero for MOT.
- `U2OnlineTracker` serializes the same state format for both tasks.

The identity GRU, anchor, appearance EMA, and identity association never
receive class text or task embeddings.  The model accepts only precomputed
floating-point U1 language features; it has no random vocabulary embedding.
RMOT, point, box, and mask tracking remain disabled.
