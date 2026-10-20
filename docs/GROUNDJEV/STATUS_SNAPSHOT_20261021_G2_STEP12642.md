# GroundJEV runtime snapshot — G2 step 12642

Recorded from the authoritative formal event log at `2026-10-21 05:17:02
+08:00`.

- Formal run: `G2_CLEAN_FULL_COVERAGE_V2/formal_clean_coverage_v2_final`.
- Progress: `12642/18227` events, finite loss `0.17345873918384314`, and
  finite gradient norm `10.642035484313965`.
- Terminal state: `final_receipt.json` is absent. The exact G2 completion gate
  remains closed, so no dependent optimizer or evaluation has been launched.
- Resources: the protected two-rank run remains on physical GPUs 3 and 5.
  G1, bundle export, held-out, native TAO and YesTrack owners remain alive in
  their registered waiting states.
- Boundary: this is runtime progress evidence only, not a MOT/RMOT/OVMOT
  metric or a final G2 detector claim.

The previous step-12633 snapshot and all historical G2-PL outputs are retained.
