"""Native SFT over immutable source manifests, without copying feature archives."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'external/OneJev')]
PROTOCOL = 'immutable-native-sources-family-filter-full-exposure-v1'
INDEXED_PROTOCOL = 'immutable-native-explicit-row-index-sources-v1'


def install_source_manifest_rows():
    from train import common
    native_load = common.load_rows

    def load(path, limit=None, split=None):
        if not str(path).endswith('.sources.json'):
            return native_load(path, limit=limit, split=split)
        manifest = json.loads(Path(path).read_text())
        assert manifest['protocol'] in (PROTOCOL,INDEXED_PROTOCOL)
        rows = []
        for source in manifest['sources']:
            eligible = [r for r in native_load(ROOT/source['path'], split=split)
                if source['families'] is None or r['task'] in source['families']]
            if 'row_indices' in source:
                assert manifest['protocol']==INDEXED_PROTOCOL and len(eligible)==source['source_rows']
                indices=source['row_indices']
                assert len(set(indices))==len(indices) and all(0<=i<len(eligible) for i in indices)
                eligible=[eligible[i] for i in indices]
            assert len(eligible) == source['rows'], source['path']
            rows.extend(eligible)
        assert len(rows) == manifest['rows']
        return rows if limit is None else rows[:limit]
    common.load_rows = load


def configure_source_manifest_sft(backend='reference'):
    if backend == 'fla':
        from tools.groundjev.run_fla_group_reference_sft import configure_fla_group_reference_sft
        stateful = configure_fla_group_reference_sft()
    else:
        from tools.groundjev.run_group_reference_sft import configure_group_reference_sft
        stateful = configure_group_reference_sft()
    install_source_manifest_rows()
    native_protocol = stateful.protocol
    def protocol(args):
        source_protocol=PROTOCOL
        if str(args.train_file).endswith('.sources.json'):
            source_protocol=json.loads(Path(args.train_file).read_text())['protocol']
        return dict(native_protocol(args), source_manifest_protocol=source_protocol)
    stateful.protocol = protocol
    return stateful


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--source-training-backend', choices=['reference','fla'], default='reference')
    parser.add_argument('--validated-native-lengths', type=Path,
        help='explicit completed native epoch-zero CPU input report; same source/seed/rendering required')
    options, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    stateful = configure_source_manifest_sft(options.source_training_backend)
    if options.validated_native_lengths:
        from tools.groundjev.validated_native_lengths import install
        install(options.validated_native_lengths, stateful, ROOT)
    raise SystemExit(stateful.main())
