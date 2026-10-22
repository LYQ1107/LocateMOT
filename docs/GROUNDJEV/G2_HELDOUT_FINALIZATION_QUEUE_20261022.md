# G2 held-out finalization queue

`scripts/groundjev/queue_g2_heldout_finalize.py` is a one-shot finalizer for the
five registered held-out owners: G0 MOT20-05, two G2 Clean MOT20 owners, and the
DanceTrack/AnimalTrack domain continuations. It waits internally for terminal
owner states, runs the receipt collector once, and writes
`g2_heldout_finalization_receipt.json`.

The finalizer does not start, stop, or modify any GPU inference process. Partial
or failed owners remain visible in the final receipt and are never promoted to
official metrics.
