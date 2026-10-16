"""CPU-only parallel native input planning; never allocates model/GPU weights."""
import argparse
from collections import Counter, defaultdict
import datetime
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'external/OneJev')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--probe-only', action='store_true')
    options = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import numpy as np
    import torch
    import torch.distributed as dist
    from tools.groundjev.run_source_manifest_sft import configure_source_manifest_sft
    from tools.groundjev.paired_sft_sampler import index_shuffle_plan
    from train import common, sft, mm

    torch.set_num_threads(4)
    configure_source_manifest_sft('fla')
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    assert world == 4
    dist.init_process_group('gloo', timeout=datetime.timedelta(hours=6))
    started = time.perf_counter()
    args = sft.parse_args(['--config', str(options.config), '--plan-only', '--device', 'cpu'])
    assert args.epochs == 1 and args.max_rows == 1 and args.accum_rows == 16
    assert args.pad_options == 0 and args.limit is None and not args.share_prefix
    manifest = json.loads(Path(args.train_file).read_text())
    rows = common.load_rows(args.train_file, split=args.split)
    assert len(rows) == manifest['rows'] and len(rows) >= 456062
    if args.media_root:
        mm.set_root(args.media_root)
    bundle = common.load_bundle(args.model, device='cpu', dtype='float32', train=True,
        multimodal=args.multimodal, chat_template=args.chat_template, load_model=False)
    assert bundle.model is None and not torch.cuda.is_initialized()
    # Actual multimodal rows from every immutable source, including different families.
    indices = []
    cursor = 0
    for source in manifest['sources']:
        indices.extend([cursor, cursor + source['rows']//2, cursor + source['rows'] - 1])
        cursor += source['rows']
    assert cursor == len(rows)
    indices = sorted(set(indices))
    probe_rows = [rows[i] for i in indices]
    parallel_probe = sft.token_lengths(bundle, probe_rows, args, 0, rank, world)
    probe_ok = [None]
    if rank == 0:
        serial_probe = sft.token_lengths(bundle, probe_rows, args, 0)
        probe_ok[0] = serial_probe == parallel_probe
        options.out.mkdir(parents=True, exist_ok=False)
        probe = dict(status='EXACT_SERIAL_FOUR_CPU_NATIVE_MM_TOKEN_LENGTH_PROBE_VALIDATED'
            if probe_ok[0] else 'SERIAL_FOUR_CPU_NATIVE_MM_TOKEN_LENGTH_PROBE_FAILED',
            source_row_indices=indices, rows=len(indices), native_lengths=parallel_probe,
            all_lengths_exact=probe_ok[0], world_size=world, actual_optimizer_updates=0,
            cuda_initialized=False, model_loaded=False)
        (options.out/'probe.json').write_text(json.dumps(probe, indent=2)+'\n')
        print(json.dumps(probe), flush=True)
    dist.broadcast_object_list(probe_ok, src=0)
    assert probe_ok[0]
    if options.probe_only:
        dist.destroy_process_group()
        return 0
    if rank == 0:
        print('FULL_NATIVE_FOUR_CPU_PLAN_STARTED_ALL_SOURCE_ROWS', len(rows), flush=True)
    batches, dropped, lengths = index_shuffle_plan(sft.plan_epoch)(bundle, rows, args, 0, rank, world)
    assert dropped == 0 and sorted(i for b in batches for i in b) == list(range(len(rows)))
    steps = sft.plan_steps(batches, lengths, world, args.accum_rows)
    assert sum(len(b) for step in steps for per in step for b in per) == len(rows)
    assert not torch.cuda.is_initialized()
    if rank == 0:
        families = Counter()
        images = defaultdict(set)
        cursor = 0
        for source in manifest['sources']:
            for row in rows[cursor:cursor + source['rows']]:
                families[source['bucket']+':'+row['task']] += 1
                if row.get('media'):
                    images[source['bucket']].add(str(Path(row['media'][0]['path']).resolve()))
            cursor += source['rows']
        assert cursor == len(rows)
        length_path = options.out/'epoch0_native_lengths.npy'
        np.save(length_path, np.asarray(lengths, dtype=np.int32), allow_pickle=False)
        sources = [dict(path=s['path'], rows=s['rows'], families=s['families'],
            bytes=(ROOT/s['path']).stat().st_size,
            mtime_ns=(ROOT/s['path']).stat().st_mtime_ns) for s in manifest['sources']]
        result = dict(status='FULL_SOURCE_ALL_NATIVE_MM_ROWS_AND_FOUR_RANK_GLOBAL16_PLAN_VALIDATED',
            rows=len(rows), rows_per_task=manifest['rows_per_task'], dropped_rows=0,
            maximum_tokens=max(lengths), planned_tokens=sum(lengths), planned_updates=len(steps),
            planned_world_size=world, global_rows_per_update=args.accum_rows,
            question_families=dict(families),
            planned_unique_first_media_source_paths={k:len(v) for k,v in images.items()},
            media_path_counts_are_filtered_row_exposure_not_raw_scope_counts=True,
            CPU_plan_elapsed_s=time.perf_counter()-started, CPU_workers=world,
            native_token_lengths_function='train.sft.token_lengths', native_epoch=0,
            paired_index_shuffle=True, exact_serial_parallel_probe_rows=len(indices),
            native_args=vars(args), source_files=sources, manifest=manifest,
            length_array=str(length_path.resolve()), length_array_bytes=length_path.stat().st_size,
            actual_optimizer_updates=0, cuda_initialized=False, model_loaded=False,
            existing_serial_owner_modified=False, full_goal_complete=False)
        (options.out/'CPU_plan_validation.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ('native_args','manifest','source_files')}), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
