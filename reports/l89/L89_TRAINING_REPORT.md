# L89 training report

## Training contract

The fit used only the L49 V1/V2 `fit` units through the frozen L85 Z1 cache,
with seed `20260829`. No calibration, validation, screening, or official-test
labels were used for optimization. The registered optimizer was AdamW,
learning rate `2e-4`, weight decay `1e-2`, betas `(0.9, 0.999)`, gradient clip
`1.0`; BF16 autocast was used with FP32 model parameters. Resource limits
allowed one process on GPU0, with effective frame batch 8. This is recorded as
`world_size=1`, not as a four-GPU reproduction.

Command:

```text
/home/lwr/anaconda3/envs/masaenv_debug/bin/python tools/l89_train_full_rmot.py --epochs 40 --seed 20260829 --z1-cache /data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/outputs/l85/features/fit_dev_eval_full_attempt2 --language-cache outputs/l89/cache/language_tokens_retry1 --out outputs/l89/train/joint40 --effective-frame-batch 8 --bf16
```

The curriculum was fixed as S epochs 1–8 (temporal disabled), T epochs 9–20,
and J epochs 21–40 (causal temporal path enabled). The unchanged L87-A loss
was used, including the all-negative target-bag fallback because verified
same-class metadata is unavailable.

## Results

| item | evidence |
|---|---:|
| epochs / optimizer steps | 40 / 2,640 |
| groups per epoch | 524 |
| trainable parameters | 4,169,829 |
| domain groups | V1 266, V2 258 |
| strata counts | inactive 364, multi-positive 557, positive 545, present-uncovered 198 |
| positive rows | 2,434 |
| masked missing rows | 8,827 |
| negative target bags | 58,007 |
| temporal identity pairs | 346 |
| finite epoch records | 40/40 |
| nonzero-gradient updates | 66/66 recorded updates per epoch |
| final mean loss | 1.1219083 |
| wall time | 22,212.2 s |

Loss decreased from `3.9850474` at epoch 1 to `1.1219083` at epoch 40. All
20 even checkpoints plus the final checkpoint package were written and are
strictly reloadable. The authoritative final package is:

```text
outputs/l89/train/joint40/checkpoint_l89_step40epoch.pt
SHA256 740f5c215bbcc55e5040f4fb3ff08c75881693e9698f336b55d6bb2d559a860a
```

The selected epoch-4 package used later has SHA256
`5ab3cb344b73b320b34de7b4bb41622ce665ecb17c4d90c1999640a318e69aa8`.
It is an L89 model package, not a production checkpoint.

## Interpretation

The finite smoke and full fit show that the QSC-D/data/loss path is executable
and receives gradients. They do not prove that the frozen representation has
usable expression correspondence. The held-out semantic and internal
TrackEval results below determine the scientific decision.
