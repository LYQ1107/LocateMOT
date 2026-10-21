# G2 Clean progress snapshot — step 12,938

The protected G2 Clean event stream is valid through step **12,938 / 18,227**.
The latest event at 06:13:50 +0800 has loss `0.1918672598`, gradient norm
`10.0799599`, and peak allocated memory `12.8289 GiB`. No terminal receipt is
present.

The measured rates are 313.316, 312.283, and 311.850 steps/hour over the last
100, 500, and 1,000 steps. With 5,289 updates remaining, the linear estimate
is approximately **23:06–23:11 on 2026-10-21 (Asia/Shanghai)**. This is a
throughput estimate, not a completion guarantee.

The durable recovery pair remains
`checkpoints/adapter-step012500.pt` and
`checkpoints/latest_training_state.pt`. All downstream owners remain gated;
this receipt records training health only and does not claim a detector or
tracking metric.
