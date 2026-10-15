"""Explicit training-only FLA0.5.2 phase; native reference inference stays separate."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'.tools/onejev-fla052'), str(ROOT), str(ROOT/'external/OneJev')]
BACKEND = 'fla0.5.2-GDN-chunk-training-only-reference-inference-v1'


def configure_fla_group_reference_sft():
    import fla
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    from tools.groundjev.run_group_reference_sft import configure_group_reference_sft
    assert fla.__version__ == '0.5.2'

    def training_chunk(*values, **kwargs):
        assert not kwargs.get('output_final_state', False), 'Use the native reference entry for cached inference'
        return chunk_gated_delta_rule(*values, **kwargs)

    modeling.torch_chunk_gated_delta_rule = training_chunk
    stateful = configure_group_reference_sft()
    native_protocol = stateful.protocol

    def protocol(args):
        return dict(native_protocol(args), training_compute_backend=BACKEND)

    stateful.protocol = protocol
    return stateful


if __name__ == '__main__':
    raise SystemExit(configure_fla_group_reference_sft().main())
