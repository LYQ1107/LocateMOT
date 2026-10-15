"""Separate native SFT entry with explicit category and tracking recipe IO."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'external/OneJev')]


def configure_tracking_recipe_sft():
    from tools.groundjev.streamed_category_media import install as install_category
    from tools.groundjev.streamed_tracking_media import install as install_tracking, IO_PROTOCOL
    install_category()
    install_tracking()
    from tools.groundjev import onejev_resumable_sft as stateful
    native_protocol = stateful.protocol

    def protocol(args):
        return dict(native_protocol(args), input_io_protocol=IO_PROTOCOL)

    stateful.protocol = protocol
    return stateful


if __name__ == '__main__':
    raise SystemExit(configure_tracking_recipe_sft().main())
