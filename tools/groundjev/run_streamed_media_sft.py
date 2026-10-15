"""Explicit streamed-input entry point for the existing checkpointed native SFT."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'external/OneJev')]


def configure_streamed_sft():
    from tools.groundjev.streamed_category_media import install,IO_PROTOCOL
    install()
    from tools.groundjev import onejev_resumable_sft as stateful
    native_protocol=stateful.protocol
    def protocol(args):
        return dict(native_protocol(args),input_io_protocol=IO_PROTOCOL)
    stateful.protocol=protocol
    return stateful


def main():
    return configure_streamed_sft().main()


if __name__=='__main__':
    raise SystemExit(main())
