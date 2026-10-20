# G2 native evaluation identity repair

`tools/groundjev/g2/run_g2_native_eval.py` previously hard-coded the adapted
short-control arm as `G2-PL`. The entrypoint now labels the terminal bundle as
`G2-CLEAN-V2` and records the Clean export receipt, merged model SHA, adapter
step, and Clean TRAIN manifest SHA in its owner state before launching any
service. It uses the dedicated GroundJEV runtime checkout for the service,
inference script and official scorers.

The historical `GroundAnything-DLM-G2-PL` model keeps the exact `G2-PL` label
and bypasses the Clean provenance gate, so old G2-PL result directories remain
addressable and unchanged. The Clean branch cannot run until the protected G2
terminal bundle exists. `py_compile` and a historical-label smoke passed.
