"""Export actual final-G2 candidates without initializing any tracker or OneJev."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.groundjev.eval_no_mot20.cache_store import CacheStore
from tools.groundjev.eval_no_mot20.common import G2_SHA, G2_STEP, job_manifest


def bounded_request(adapter, image, manifest, vocabulary, image_path, timeout):
    def expired(_signum, _frame):
        raise TimeoutError(f'native request exceeded frozen {timeout}s limit')
    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return adapter(image, manifest['task'], vocabulary, manifest.get('expression'), image_path=image_path)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def export(job: Path, base_url: str, start: int = 0, stop: int | None = None,
           timeout: float = 600, worker: str = '') -> dict:
    from PIL import Image
    from locatemot.groundjev.perception import GroundAnythingAdapter, GroundingProtocolError

    manifest = job_manifest(job)
    store = CacheStore(job)
    adapter = GroundAnythingAdapter(base_url, parser_policy='official_scoring', decoder_policy='tracking_dense',
                                    allow_vocabulary_fallback=False, request_timeout=timeout)
    vocabulary = manifest['vocabulary']
    chunk_size = manifest.get('vocabulary_chunk_size') or len(vocabulary)
    chunks = [vocabulary[offset:offset + chunk_size] for offset in range(0, len(vocabulary), chunk_size)]
    new_frames = new_requests = 0
    try:
        stop = min(stop if stop is not None else len(manifest['images']), len(manifest['images']))
        for index in range(start, stop):
            frame = manifest['frame_numbers'][index]
            if store.frame(frame) is not None:
                continue
            path = manifest['images'][index]
            io_start = time.perf_counter()
            with Image.open(path) as source:
                image = source.convert('RGB')
            io_seconds = time.perf_counter() - io_start
            candidates, outcomes = [], []
            for chunk, values in enumerate(chunks):
                cached = store.request(frame, chunk)
                if cached is None:
                    began = time.perf_counter()
                    try:
                        current, report = bounded_request(adapter, image, manifest, values, path, timeout)
                        cached = {'status': 'SCORED', 'candidates': current, 'metadata': report}
                    except GroundingProtocolError as error:
                        cached = {'status': 'REJECTED', 'candidates': [], 'metadata': error.metadata}
                    except (TimeoutError, OSError) as error:
                        # Transport failure is unfinished work, never a synthetic
                        # prediction for all remaining frames. The dispatcher can
                        # restart this service/unit with the identical protocol.
                        with (job / 'errors.jsonl').open('a') as handle:
                            handle.write(json.dumps({'frame': frame, 'chunk': chunk, 'worker': worker,
                                'kind': type(error).__name__, 'error': str(error), 'unfinished': True}) + '\n')
                        raise
                    cached.update(seconds=time.perf_counter() - began, vocabulary=values,
                                  physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), worker=worker,
                                  image_path=path, model_sha256=G2_SHA, model_step=G2_STEP)
                    store.put_request(frame, chunk, cached)
                    new_requests += 1
                if cached['model_sha256'] != G2_SHA or cached['image_path'] != path or cached['vocabulary'] != values:
                    raise ValueError('Saved request is incompatible with the current frame/query/model')
                candidates.extend(cached['candidates'])
                outcomes.append({'chunk': chunk, 'status': cached['status'], 'seconds': cached['seconds'],
                                 'raw_response_ref': f'detector.sqlite:requests:{frame}:{chunk}'})
            rejected = any(outcome['status'] != 'SCORED' for outcome in outcomes)
            perception = {'mode': 'final_g2_detector_only', 'model_response_rejected': rejected,
                          'strict_wire_valid': all((store.request(frame, i)['metadata'].get('strict_wire_valid', False))
                                                   for i in range(len(chunks))),
                          'parser_policy': 'official_scoring', 'decoder_policy': 'tracking_dense',
                          'request_count': len(chunks), 'chunk_outcomes': outcomes,
                          'all_vocabulary_classes_requested': True, 'vocabulary_fallback': None,
                          'chunk_failure_policy': 'retain every raw chunk; empty frame if any chunk is rejected',
                          'gt_read_by_runtime': False}
            store.put_frame({'dataset': manifest['dataset'], 'split': manifest['split'],
                             'video_id': manifest.get('video_id', manifest['sequence']),
                             'expression_id': manifest.get('expression_id'), 'query_hash': manifest['query_hash'],
                             'frame': frame, 'image_path': path, 'image_size': list(image.size),
                             'candidates': [] if rejected else candidates, 'perception': perception,
                             'seconds': io_seconds + sum(outcome['seconds'] for outcome in outcomes),
                             'image_io_seconds': io_seconds, 'model_sha256': G2_SHA, 'model_step': G2_STEP,
                             'cache_identity': store.identity})
            new_frames += 1
            if new_frames == 1 or new_frames % 8 == 0:
                print(json.dumps({'event': 'DETECTOR_PROGRESS', 'job': job.name, 'frame': frame,
                                  'new_frames': new_frames, 'new_requests': new_requests,
                                  'rejected': rejected, 'candidates': 0 if rejected else len(candidates)}), flush=True)
        receipt = store.finalize()
        return {'new_frames': new_frames, 'new_requests': new_requests, 'complete': receipt is not None}
    finally:
        store.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--grounding-url', required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--worker', default='')
    args = parser.parse_args()
    print(json.dumps(export(args.job, args.grounding_url, args.start, args.stop, args.timeout, args.worker)), flush=True)


if __name__ == '__main__':
    main()
