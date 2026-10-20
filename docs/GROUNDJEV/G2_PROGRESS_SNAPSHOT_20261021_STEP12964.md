# G2 Clean progress snapshot — step 12,964

The protected G2 Clean event stream is live through step **12,964 / 18,227**.
The latest event at 06:18:52 +0800 has loss `0.3086177218`, gradient norm
`13.7076988`, and peak allocated memory `12.8289 GiB`. Both registered ranks
remain running on physical GPUs 3 and 5; no terminal receipt is present.

The measured rates are 312.500, 312.067, and 311.850 steps/hour over the last
100, 500, and 1,000 steps. With 5,263 updates remaining, the linear estimate
is approximately **23:09–23:11 on 2026-10-21 (Asia/Shanghai)**. This is a
throughput estimate, not a completion guarantee.

The durable recovery pair remains
`checkpoints/adapter-step012500.pt` and
`checkpoints/latest_training_state.pt`. This receipt records training health
only and does not claim a detector or tracking metric.
