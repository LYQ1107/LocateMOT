# YesTrack-inspired verification architecture for GroundJEV

The proposed branch is an opt-in semantic verifier around the existing
physical tracker:

```text
native perception -> candidate set -> physical OneJev/Compact decision
                                      -> global one-to-one assignment
                                      -> lifecycle proposal
                                      -> selective Yes/No verifier (optional)
                                      -> semantic score / RMOT membership output
                                      -> causal memory update
```

`locatemot/groundjev/yes_verification.py` defines a backend-neutral request
and result. A request contains the current media, task/expression, candidate
box, persistent track ID, and strictly earlier evidence frame IDs. It does not
contain GT labels, evaluation boxes, oracle identities, or future frames. A
backend may return token logits, named Yes/No logits, a probability, or text;
the adapter normalizes all of them to `p_yes` with a tested direction.

`verification_policy.py` is disabled unless the caller enables it. When
enabled, it spends a per-frame budget only on low top-2 margin, high entropy,
reactivation, contradiction, or query ambiguity. It fuses a verified
probability into an existing association score without changing the physical
candidate pool. For a candidate tentatively marked NEW, semantic No yields
IGNORE; it never creates NEW. Global assignment remains one-to-one and is
tested independently on CPU.

`semantic_temporal_cache.py` caches only semantic probability, keyed by
persistent track ID, task, normalized query, verifier checkpoint hash, and
protocol version. It is causal and invalidates on query change, reappearance,
conflict, termination, or TTL expiry. It cannot bypass physical association or
overwrite track memory.

The baseline equivalence contract is simple: with the default configuration
(`enabled=false`) no backend is called, no cache entry is consumed, and the
original association score/lifecycle state is returned byte-for-byte at the
policy boundary. This is a preparation architecture, not a result claim.
