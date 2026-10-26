# Published speed is not video FPS

Primary sources checked on 2026-10-26:

- [LocateAnything, section 4.2 and appendix C.2–C.5](https://arxiv.org/html/2605.27365):
  COCO throughput on one H100, batch 1, BF16. Fast/Hybrid/Slow are
  15.3/12.7/4.3 boxes per second. These are not frames per second.
- [GroundAnything, sections 4.3 and 9.1](https://arxiv.org/html/2609.39600):
  the reference operating points are AR 49.6, entropy-guided 127.0 and linear
  self-speculative 223.8 output tokens per second. The 2.56x/4.51x comparisons
  use AR as denominator and are not a matched LocateAnything/G2 video race.

Our current final-G2 run uses the native entropy-guided BF16 eager route.
TAO requests all 1,203 classes in 13 native requests for each registered sparse
annotation frame. Large malformed outputs have a real time cost even when the
frozen parser rejects them. A different query load, output length, hardware or
timer scope invalidates a direct comparison with paper BPS/TPS.

Historical LocateAnything L0-B records remain at
`/data2/user/LocateMOT/outputs/l0_b_token_debug/l0_b_runtime.csv`: nine ordinary
detection requests average 0.7189 seconds and nine single-target requests average
0.5462 seconds. They are small functional samples, not a controlled speed ranking
against the current G2 validation inputs.
