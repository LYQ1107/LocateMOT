"""Paths for the migrated LocateMOT checkout."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
RESTORE_ROOT = OUTPUT_ROOT / "restore_20261001"
R1_ROOT = OUTPUT_ROOT / "r1_rebuild"
IMAGE_ROOT = DATA_ROOT / "kitti_tracking_training" / "image_02"
V1_ROOT = DATA_ROOT / "refer_kitti_v1"
V2_ROOT = DATA_ROOT / "refer_kitti_v2"
R1_L49_DATA = R1_ROOT / "l49" / "data"
R1_L48_TEXT = R1_ROOT / "l48" / "data" / "text_cache.pt"
R1_L62_ROWS = R1_ROOT / "l62" / "score_records.jsonl"
R1_SPLIT = R1_ROOT / "protocol" / "fit_video_train_dev_split.json"
R1_L69_ROOT = R1_ROOT / "l69" / "budget40_features" / "kitti"
R1_LANGUAGE_ROOT = R1_ROOT / "language_tokens"
R1_VISUAL_ROOT = R1_ROOT / "visual_tokens"
R1_MANIFEST = R1_ROOT / "protocol" / "kitti_fast_eval_manifest.json"
R1_FIT_INDEX_ROOT = R1_ROOT / "data" / "fit_indexes"
R1_DEV_INDEX_ROOT = R1_ROOT / "data" / "dev_indexes"
R1_SAFE_TARGET_ROOT = R1_ROOT / "data" / "safe_targets"
R1_ANCHOR = R1_ROOT / "anchor" / "checkpoint.pt"
R1_RULE = R1_ROOT / "anchor" / "rule.json"


def require_file(path: Path, description: str) -> Path:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    return path
