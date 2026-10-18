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
reactivation, contradiction, or query ambiguity. It exposes separate
`verify_identity` and `verify_semantic` request paths while sharing one
backend. It fuses a verified probability into an existing association score
without changing the physical candidate pool. For a candidate tentatively
marked NEW, semantic No yields IGNORE; it never creates NEW. The optional
`assignment.assign_candidates(..., verification_results=...)` hook applies
the fusion to shortlisted Track/Detection pairs before the single Hungarian
solve and returns raw/fused event evidence; it does not write memory or run a
second assignment. Global assignment remains one-to-one and is tested on CPU.

Two registered fusion modes are available: `fixed` uses the configured
weight, while `calibrated` applies a temperature/bias fitted by
`fit_logistic_calibrator` on a declared calibration split before the fixed
weight is applied. No uncalibrated sum is described as a probability.

`semantic_temporal_cache.py` caches only semantic probability, keyed by
persistent track ID, task, normalized query, verifier checkpoint hash, and
protocol version. It is causal and invalidates on query change, reappearance,
conflict, termination, or TTL expiry. It cannot bypass physical association or
overwrite track memory.

The baseline equivalence contract is simple: with the default configuration
(`enabled=false`) no backend is called, no cache entry is consumed, and the
original association score/lifecycle state is returned byte-for-byte at the
policy boundary. This is a preparation architecture, not a result claim.
