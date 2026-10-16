"""Explicit reuse of completed native epoch-zero input measurements."""
import json
from pathlib import Path

PROTOCOL = 'completed-native-epoch0-token-lengths-same-inputs-and-plan-v1'
INPUT_KEYS = ('model', 'train_file', 'split', 'limit', 'media_root', 'multimodal',
    'chat_template', 'seed', 'augment', 'pad_options', 'max_len', 'batch_tokens',
    'max_rows', 'accum_rows', 'epochs')


def measured_lengths(report_path, rows, args, epoch, world, root):
    import numpy as np
    report = json.loads(Path(report_path).read_text())
    assert report['status'] == 'FULL_SOURCE_ALL_NATIVE_MM_ROWS_AND_FOUR_RANK_GLOBAL16_PLAN_VALIDATED'
    assert epoch == report['native_epoch'] == 0 and args.epochs == 1
    assert report['dropped_rows'] == 0 and report['rows'] == len(rows)
    assert report['planned_world_size'] == world
    assert report['global_rows_per_update'] == args.accum_rows
    assert report['paired_index_shuffle'] and not args.share_prefix
    for key in INPUT_KEYS:
        assert getattr(args, key) == report['native_args'][key], key
    manifest = json.loads(Path(args.train_file).read_text())
    assert manifest == report['manifest'] and manifest['rows'] == len(rows)
    assert len(manifest['sources']) == len(report['source_files'])
    for source, saved in zip(manifest['sources'], report['source_files']):
        assert all(source[k] == saved[k] for k in ('path', 'rows', 'families'))
        stat = (Path(root)/source['path']).stat()
        assert stat.st_size == saved['bytes'] and stat.st_mtime_ns == saved['mtime_ns'], source['path']
    lengths = np.load(report['length_array'], allow_pickle=False)
    assert lengths.dtype == np.int32 and lengths.shape == (len(rows),)
    assert int(lengths.min()) > 0 and int(lengths.max()) <= args.max_len
    assert int(lengths.max()) == report['maximum_tokens']
    assert int(lengths.sum(dtype=np.int64)) == report['planned_tokens']
    return lengths.tolist()


def install(report_path, stateful, root):
    from train import sft
    native_parse = sft.parse_args
    native_protocol = stateful.protocol
    validated = None

    def parse(argv=None):
        args = native_parse(argv)
        args.validated_native_lengths = str(Path(report_path).resolve())
        return args

    def token_lengths(bundle, rows, args, epoch, rank=0, world=1):
        nonlocal validated
        assert epoch == 0
        if validated is None:
            validated = measured_lengths(report_path, rows, args, epoch, world, root)
            if rank == 0:
                print(json.dumps(dict(status='EXPLICIT_COMPLETED_NATIVE_TOKEN_LENGTHS_REUSED',
                    protocol=PROTOCOL, report=str(Path(report_path).resolve()),
                    rows=len(rows), epoch=epoch, world_size=world,
                    original_native_batching_and_index_shuffle_unchanged=True)), flush=True)
        assert len(validated) == len(rows)
        return validated

    def protocol(args):
        return dict(native_protocol(args), native_input_planning_protocol=PROTOCOL)

    sft.parse_args = parse
    sft.token_lengths = token_lengths
    stateful.protocol = protocol
