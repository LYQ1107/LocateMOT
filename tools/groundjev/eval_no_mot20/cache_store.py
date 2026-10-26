"""Atomic, per-video SQLite request persistence and a single replay JSONL.

Successful vocabulary chunks survive interruption independently. SQLite
transactions commit complete records, so no truncated JSONL prefix is trusted.
The final JSONL is exported atomically only after every registered frame exists.
"""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import sqlite3
from pathlib import Path

from .common import canonical, candidate_payload, fingerprint, job_manifest, manifest_identity, save, sha_file


class CacheStore:
    def __init__(self, job: Path):
        self.job = job
        self.manifest = job_manifest(job)
        self.identity = fingerprint(manifest_identity(self.manifest))
        self.db = sqlite3.connect(job / 'detector.sqlite', timeout=120)
        # Distinct frame units can first open this video's database together.
        # WAL activation itself does not reliably wait via SQLite busy_timeout.
        with (job / '.initialize.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.db.execute('PRAGMA journal_mode').fetchone()[0] != 'wal':
                self.db.execute('PRAGMA journal_mode=WAL')
            self.db.execute('PRAGMA synchronous=FULL')
            self.db.executescript('''
                CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS requests(frame INTEGER, chunk INTEGER, data BLOB NOT NULL,
                    PRIMARY KEY(frame,chunk));
                CREATE TABLE IF NOT EXISTS frames(frame INTEGER PRIMARY KEY, data BLOB NOT NULL);
            ''')
            with self.db:
                self.db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)', ('identity', self.identity))
        actual = self.db.execute('SELECT value FROM metadata WHERE key=?', ('identity',)).fetchone()[0]
        if actual != self.identity:
            raise ValueError('Cache model, images, query or decoder differs from immutable manifest')

    def close(self):
        self.db.close()

    def _read(self, table, where, values):
        row = self.db.execute(f'SELECT data FROM {table} WHERE {where}', values).fetchone()
        return json.loads(gzip.decompress(row[0])) if row else None

    def request(self, frame, chunk):
        return self._read('requests', 'frame=? AND chunk=?', (frame, chunk))

    def put_request(self, frame, chunk, data):
        blob = gzip.compress(canonical(data), mtime=0)
        with self.db:
            self.db.execute('INSERT INTO requests VALUES (?,?,?)', (frame, chunk, blob))

    def frame(self, frame):
        return self._read('frames', 'frame=?', (frame,))

    def put_frame(self, row):
        with self.db:
            self.db.execute('INSERT INTO frames VALUES (?,?)',
                            (row['frame'], gzip.compress(canonical(row), mtime=0)))

    def completed(self):
        return self.db.execute('SELECT COUNT(*) FROM frames').fetchone()[0]

    def finalize(self):
        if self.completed() != len(self.manifest['images']):
            return None
        with (self.job / '.finalize.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            receipt_path = self.job / 'cache_receipt.json'
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if receipt['cache_identity'] != self.identity:
                    raise ValueError('Finalized cache identity changed')
                return receipt
            output = self.job / 'detector_candidates.jsonl'
            temporary = output.with_suffix('.jsonl.tmp')
            rejected = candidates = requests = reused = 0
            seconds = io_seconds = 0.0
            candidate_hash = hashlib.sha256()
            with temporary.open('wb') as handle:
                for frame, image in zip(self.manifest['frame_numbers'], self.manifest['images']):
                    row = self.frame(frame)
                    if row is None or row['image_path'] != image:
                        raise ValueError('Complete ordered scope is missing or changed')
                    rejected += bool(row['perception']['model_response_rejected'])
                    candidates += len(row['candidates'])
                    requests += row['perception']['request_count']
                    reused += row.get('reused_historical_requests', 0)
                    seconds += row['seconds']
                    io_seconds += row['image_io_seconds']
                    candidate_hash.update(canonical(candidate_payload(row)) + b'\n')
                    handle.write(canonical(row) + b'\n')
            temporary.replace(output)
            receipt = {'status': 'DETECTION_CACHE_COMPLETE', 'cache_identity': self.identity,
                       'model_sha256': self.manifest['model_sha256'], 'model_step': self.manifest['model_step'],
                       'frames': len(self.manifest['images']), 'candidates': candidates, 'rejected': rejected,
                       'requests': requests, 'reused_historical_requests': reused,
                       'detector_seconds': seconds, 'image_io_seconds': io_seconds,
                       'detector_only_fps': len(self.manifest['images']) / seconds if seconds else None,
                       'candidate_sha256': candidate_hash.hexdigest(), 'jsonl_sha256': sha_file(output),
                       'raw_responses': 'detector.sqlite requests table: gzip JSON by physical frame/chunk',
                       'gt_read_by_detector': False, 'prompt_fallback': False,
                       'physical_video_fps_claim': self.manifest.get('physical_frames_complete', False),
                       'timing_scope': 'sum of actual per-frame image IO and complete native requests; not four-worker wall time'}
            save(receipt_path, receipt)
            return receipt


def audit(job: Path) -> dict:
    store = CacheStore(job)
    try:
        receipt = store.finalize()
        if receipt is None:
            return {'status': 'INCOMPLETE', 'frames': store.completed()}
        output = job / 'detector_candidates.jsonl'
        if sha_file(output) != receipt['jsonl_sha256']:
            raise ValueError('Final JSONL checksum mismatch')
        return receipt
    finally:
        store.close()
