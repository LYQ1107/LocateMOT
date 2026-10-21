# Independent held-out scope audit

The registered held-out manifests were checked against the immutable G2 Clean TRAIN rows before the terminal G2 receipt exists. All four scopes are image-disjoint:

| Scope | Frames | Exact TRAIN overlap |
|---|---:|---:|
| MOT20-05 validation half | 1,657 | 0 |
| RMOT single | 800 | 0 |
| RMOT multi | 800 | 0 |
| TAO OVMOT control | 40 | 0 |

The audit is CPU-only and does not run a model. It clears the data-independence gate for later G0/G2 held-out inference, while metrics remain pending the exact G2 terminal bundle.
