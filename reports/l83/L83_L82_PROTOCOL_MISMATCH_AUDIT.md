# L82 protocol mismatch audit

Status: `complete`.  L82 source files are immutable evidence; this audit does not edit them.

| Check | Result | Evidence / L83 correction |
|---|---|---|
| row-level pos/neg, smooth-min and positive floor | `True` | `l82_rank_loss` |
| no target-bag grouping in old primary loss | `True` | `l82_rank_loss` |
| reported AUC is pairwise concordance; L83 uses real ROC-AUC | `True` | `aggregate_group_metrics` |
| old R@5 is row top-5; L83 ranks unique bags | `True` | `group_metrics` |
| candidate/query diagnostics share self.score | `True` | `L82FactorizedRankProbe.forward` |
| candidate visual seed plus reference position vs native query embedding | `True` | `capture_candidate_state plus native pre_decoder` |
| fixed reference has no reg branches and self_attn_mask=None | `True` | `fixed_reference_decoder` |

The mismatch facts are the reason L83 is a new stage; no L82 result is overwritten.
