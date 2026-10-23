# GroundJEV coupling status — 2026-10-23

The protected G2 Clean held-out result is recorded separately. The real native MOT/RMOT/OVMOT OneJev coupling run uses the immutable 131,618-row v2 compact manifest, `compact_competition_v1`, zero over-length rows at max_len 32768, and GPUs [3, 7].

- Queue status: `REAL_NATIVE_THREE_TASK_TRAINING_RUNNING`
- Manifest SHA256: `7c92778fb4417ea1e49bf286cab7758ceef1443d683ab059e5e758e5bd823aa3`
- Planned rows: `131618`; task counts: `{'category_group': 29, 'category_leaf': 29, 'empty_query_presence': 16, 'identity': 131214, 'query_membership': 330}`
- Planned four-rank steps: `8227`; current distributed world size: `2`
- Latest recorded event:

```json
{
  "step": 75,
  "epoch": 0,
  "examples": 1200,
  "loss": 3.3526790034165606,
  "ce": 2.285406290022656,
  "brier": 1.0672727102119826,
  "slot_acc": 0.1825,
  "grad_norm": 9.401009101867675,
  "lr": 6.0728744939271254e-06,
  "rows_per_step": 16.0,
  "examples_per_s": 0.7008054473727477,
  "tokens_per_s": 5510.107358158887,
  "sec_per_step": 22.830872762165964,
  "peak_mem_gb": 7.795474529266357,
  "elapsed_s": 1738.2480546890292
}
```

No JEV checkpoint or tracking metric is claimed until the training receipt is terminal and the resulting checkpoint passes independent evaluation. The dense-context and resource-window attempts remain preserved as superseded evidence.
