# J1 DT1203 strict ablation (2026-10-19)

Checkpoint step 1200; merged `model.safetensors` SHA256 `5cc64d1bcae70db4a4d8596c655bc38ce0e8c1d63cb3bd2bf14d30e9848c07a5`. All rows use the same registered 1203-frame DanceTrack scope and the same TrackEval commit. Runtime reads no GT and does not tune thresholds.

| Arm | Parser/candidates | NO_MATCH contract | HOTA | DetA | AssA | IDF1 | IDSW | FP | FN | outputs |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | original parser / original archive | `learned` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 4586 | 0 |
| B | repaired parser archive | `learned` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 4586 | 0 |
| C | legal MOT fallback already in A | `learned` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 4586 | 0 |
| D | original archive | `only_if_omitted` | 0.162326 | 0.202661 | 0.130116 | 0.132145 | 0 | 69 | 3520 | 1135 |
| E | repaired parser archive | `only_if_omitted` | 0.158562 | 0.197692 | 0.127243 | 0.128277 | 0 | 110 | 3560 | 1136 |

## Decision and lifecycle evidence

A and B select NO_MATCH for essentially all candidates; A has 9,755 candidates and B has 13,853 after parser repair, but both produce zero tracks. D changes only the NO_MATCH feasibility contract and produces 1 NEW, 1,133 ACTIVE, 1 REACTIVATE and 1 LOST event with 1,135 output rows. E produces 1 NEW and 1,135 ACTIVE events with 1,136 output rows. Native probability averages, action counts and rejection/fallback counts are in the JSON receipt.

C is explicitly marked non-separable: the immutable original A run already recorded the legal MOT vocabulary fallback on 972 frame records, so rerunning C would duplicate A rather than isolate a new factor.

The repaired E HOTA/IDF1 is 0.158562/0.128277; D is 0.162326/0.132145. This difference is retained as evidence that parser repair changes the candidate set and is not itself an identity gain.
