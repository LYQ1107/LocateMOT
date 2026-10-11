# U2 exact fixed native text reuse source

Original scientific commit: `1ad7563d81a0271ad83e5446f0e27216a06149d9`.
Source tree: `e9e2c07b325681bfd6db8691573cb328e44e4b8e`.
Parent scientific commit: `2b775b7132bf07967f8c34cd6b424fa232ef85b5`.

The incremental `u2_source_history_1ad7563.bundle` retains the original
scientific commit and requires that parent. Earlier source-history bundles
retain the chain back to verified U1R2.

This stage adds worker-local fixed native BERT/token-prompt reuse in upcoming
selected ordinary and OVMOT candidate caches. Actual FP32 F0 two-frame
MOT150/full1203-class OVMOT300 outputs and all candidate features are bit-exact.
Native/cached two-frame BERT calls are2/1 and76/38; original image and full
multimodal computations remain unchanged. Actual worker serialization retains
bit-exact FP32/BF16 fields, shared semantic/decoder storage and completed-shard
resume without inference. A warm OVMOT frame took5.256/4.638seconds; these are
shared-GPU two-frame observations, not full-cache throughput measurements.

Running F0 audits and perception/tracking owners were not restarted. Full
selected caches, formal tracking metrics and U3 readiness remain unproven.
No model weights, datasets or private cloud/browser/administration artifacts
are published.
