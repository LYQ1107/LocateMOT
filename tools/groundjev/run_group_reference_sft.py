"""Explicit native SFT entry for shared group references and exact crop recipes."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'external/OneJev')]


def configure_group_reference_sft():
    from tools.groundjev.run_tracking_recipe_sft import configure_tracking_recipe_sft
    from tools.groundjev.group_reference_rows import install, IO_PROTOCOL
    stateful = configure_tracking_recipe_sft()
    install(ROOT)
    native_protocol = stateful.protocol

    def protocol(args):
        return dict(native_protocol(args), input_io_protocol=IO_PROTOCOL)

    stateful.protocol = protocol
    return stateful


if __name__ == '__main__':
    raise SystemExit(configure_group_reference_sft().main())
