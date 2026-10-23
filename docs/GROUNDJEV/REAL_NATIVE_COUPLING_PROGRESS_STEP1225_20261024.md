# Real native three-task coupling — step 1225

The compact native GroundJEV coupling run remains active on GPUs 3 and 7. It uses the registered 131,618-row native MOT/RMOT/OVMOT manifest (`compact_competition_v1`, SHA-256 `7c92778fb4417ea1e49bf286cab7758ceef1443d683ab059e5e758e5bd823aa3`) with a 32,768-token contract and 8,227 planned optimizer steps.

The latest authoritative event is step 1,225 (19,600 examples): loss `1.6275910484202905`, cross-entropy `1.2094160445517628`, Brier `0.41817499877115094`, slot accuracy `0.615`, finite gradient norm `7.9178149604797365`, and peak event memory `7.7956 GiB`. The latest verified recovery checkpoint is step 1,000. Its full state SHA-256 is `f3b37033f57963dce452b384503009fc587531ac05f6cc16742dd511103e5213`; its adapter SHA-256 is `10a54906af4f001ba466b58ee3cc9c0655721594cc7de8f70f3938256fd505b8`. Both reload successfully on CPU.

This is training-health and resumability evidence only. It is not a MOT/RMOT/OVMOT performance claim; the terminal checkpoint and independent held-out evaluation remain required.
