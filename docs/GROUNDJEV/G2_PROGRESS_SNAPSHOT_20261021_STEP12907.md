# G2 Clean progress snapshot — step 12,907

The protected G2 Clean event stream is valid through step **12,907 / 18,227**.
The latest event at 06:07:55 +0800 has loss `0.1906903072`, gradient norm
`9.2506351`, and peak allocated memory `12.8288 GiB`. No terminal receipt is
present.

The measured rates are 312.084, 312.309, and 311.484 steps/hour over the last
100, 500, and 1,000 steps. With 5,320 updates remaining, the linear estimate
is approximately **23:10–23:13 on 2026-10-21 (Asia/Shanghai)**. This is a
throughput estimate, not a completion guarantee.

The durable recovery pair remains
`checkpoints/adapter-step012500.pt` and
`checkpoints/latest_training_state.pt`. G1 GPU5, bundle export, held-out GPU7,
native TAO shards, native OVMOT join, native three-task training, and YesTrack
remain correctly gated; none has launched an optimizer update in this snapshot.

This receipt records training health and recoverability only. It does not claim
a G2 tracking metric or open any downstream gate.
