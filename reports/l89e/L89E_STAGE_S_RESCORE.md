# L89E Stage-S sparse rescore

Stage-S epochs `[2, 4, 6, 8]` were rescored with
`temporal_enabled=false` and `history_contract=zero_history`.  The output has
138 legal dev groups, exactly 1992
records, and [498, 498, 498, 498] records per checkpoint.
The corrected B/R/P rule fits are fit/dev-only; fixed calibration and
validation were not read in this stage.  Candidate rows were retained with no
deletion or truncation, and `zero_training=true`.

The historical T/J epochs 10–40 were not re-forwarded: their original sparse
path already used `build_frame(..., temporal_enabled=True)` and L86's last-four
causal packing.  Their score values were copied unchanged at merge time.
