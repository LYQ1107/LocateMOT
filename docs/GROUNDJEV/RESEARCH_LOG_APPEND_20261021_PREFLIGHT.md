# Research log append — three-task CPU preflight boundary

The registered CPU plan rendered all 131,990 rows in 267 seconds. At the
32,768-token contract, 129,891 rows are trainable and 2,099 high-candidate
MOT identity rows exceed the limit (p50 12,720; p90 25,689; p99 47,023;
maximum 96,734). The queue now records this exclusion explicitly and cannot
report full-source exposure until a compacting or long-row course handles the
2,099 rows.
