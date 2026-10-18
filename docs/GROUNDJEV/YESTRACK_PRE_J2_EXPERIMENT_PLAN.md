# Pre-J2 YesTrack-inspired experiment plan

No large Yes/No training is started in Phase 0. The active J1 and clean G2
owners remain untouched, and J2 is blocked until J1 reaches a valid step-1200
terminal receipt plus the registered diagnostics are complete.

1. **Stage 0 CPU preparation.** Run the three preparers on small native mock or
   existing candidate exports, audit rows, verify no GT/future prompt content,
   and keep the output manifest/checksum. Run probability-direction,
   lifecycle, global-assignment, cache, timeout, and baseline-equivalence
   tests.
2. **Verifier-only smoke.** Inject a deterministic CPU mock backend into the
   adapter. Check that Yes increases membership probability, No cannot create a
   new ID, rejected frames remain represented, and latency/timeout receipts are
   complete. This is a plumbing result only.
3. **J2 data contract.** After J1 final and diagnostics, choose a fixed native
   RMOT/OVMOT scope, freeze the candidate export and source manifest, then run
   the same audit. Do not add Novel training labels or MOT20-05.
4. **Registered ablations.** Compare the existing baseline, verifier-only,
   verifier plus semantic cache, and verifier plus causal temporal cache. Keep
   the same checkpoint, rows, candidate inputs, seeds, frame clock, global
   assignment, and evaluator. Record per-frame calls, timeout rate, p_yes,
   rejected frames, HOTA/IDF1/IDSW/FP/FN and RMOT/OVMOT metrics.
5. **Promotion gate.** Promote only if the fixed-scope result improves the
   registered primary metric without a coverage or protocol change and without
   worsening the required task gates. Otherwise retain the branch as an
   analysis artifact.

The ablation configuration is
`configs/groundjev/yes_verification_ablation_v1.json`. The default Stage 0
configuration keeps every runtime flag disabled. No paper-level improvement
is claimed from the architecture or the CPU smoke.
