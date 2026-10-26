from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
GROUND = Path('/data2/user/LocateMOT_GroundJEV')
RUN = Path('/data3/user/LocateMOT-runs/GroundJEV_Eval_NoMOT20_4GPU/request_local_seed_v2')
G2_STEP = 18227
G2_SHA = '7c4a81349e333a304e32f42cf1999db289802d8c31eba415b85827038d14801f'
TRAIN_SHA = 'f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71'
TRAIN_ROOT = Path('/data3/user/LocateMOT-runs/GroundAnythingG2/G2_CLEAN_FULL_COVERAGE_V2')


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as handle:
        handle.write(canonical(value) + b'\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def arm_directory(arm: str, config=None) -> str:
    if arm!='j1':return arm
    if config is None:config=load(ROOT/'configs/groundjev/eval_no_mot20_4gpu_v1.json')
    return config.get('j1_artifact_dir','j1')


def assert_no_mot20(manifest: dict) -> None:
    # Explicit user-requested dataset boundary, also catches aliased manifests.
    values = [manifest.get(k, '') for k in ('dataset', 'sequence', 'video_id', 'images')]
    if 'mot20' in json.dumps(values).lower():
        raise ValueError('MOT20 is excluded from this entire evaluation')


def candidate_payload(row: dict) -> dict:
    return {'frame': row['frame'], 'image_path': row['image_path'],
            'candidates': [{k: c.get(k) for k in ('box', 'label', 'native_score', 'score_kind')}
                           for c in row['candidates']],
            'rejected': bool(row['perception'].get('model_response_rejected', False))}


def manifest_identity(manifest: dict) -> dict:
    assert_no_mot20(manifest)
    return {k: manifest.get(k) for k in ('dataset', 'split', 'sequence', 'video_id', 'expression_id',
            'expression', 'vocabulary', 'images', 'frame_numbers', 'model_sha256', 'model_step',
            'parser', 'decoder', 'vocabulary_chunk_size', 'prompt_fallback',
            'sampling_seed', 'sampling_seed_scope')}


def job_manifest(job: Path) -> dict:
    manifest = load(job / 'manifest.json')
    assert_no_mot20(manifest)
    if manifest['model_sha256'] != G2_SHA or manifest['model_step'] != G2_STEP:
        raise ValueError('Only the registered final G2 Clean model is permitted')
    return manifest


def merged_eval_manifest(job: Path) -> dict:
    return {**job_manifest(job), **load(job / 'evaluation.json')}
