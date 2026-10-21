# Held-out input preflight

The four registered independent held-out manifests were checked before the G2 terminal bundle exists. All image paths are present and each manifest declares `training_on_this_scope=false`:

| Scope | Task | Frames | Missing images |
|---|---|---:|---:|
| MOT20-05 validation half | MOT | 1,657 | 0 |
| RMOT single | RMOT | 800 | 0 |
| RMOT multi | RMOT | 800 | 0 |
| TAO OVMOT control | OVMOT | 40 | 0 |

This is a CPU-only input check. It does not run a model and does not provide a metric claim.
