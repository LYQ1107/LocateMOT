#!/usr/bin/env python3
"""Pin unchanged official TETA and prepare full original TAO validation view."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from locatemot.unified.runtime.u2_teta_evaluation import TETA_ROOT, digest, prepare_tao_validation_gt

COMMIT = "b498aa87b252bfb75d7cc0d8d6435c3e1e74e4de"


def main() -> int:
    if not TETA_ROOT.exists():
        subprocess.run(["git", "-c", "http.proxy=", "-c", "https.proxy=", "clone",
                        "https://github.com/siyuanliii/TETA", str(TETA_ROOT)], check=True)
        subprocess.run(["git", "-C", str(TETA_ROOT), "checkout", COMMIT], check=True)
    revision = subprocess.check_output(["git", "-C", str(TETA_ROOT), "rev-parse", "HEAD"], text=True).strip()
    modified = subprocess.check_output(["git", "-C", str(TETA_ROOT), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if revision != COMMIT or modified:
        raise ValueError("Official TETA must use the unchanged frozen upstream revision")
    files = ["teta/eval.py", "teta/config.py", "teta/metrics/teta.py", "teta/metrics/_base_metric.py",
             "teta/datasets/tao.py", "teta/datasets/_base_dataset.py", "scripts/run_ovmot.py"]
    manifest = {"format": "locatemot-u2-official-teta-source-v1", "repository": "https://github.com/siyuanliii/TETA",
                "commit": COMMIT, "unmodified": True, "root": str(TETA_ROOT),
                "acquisition": "direct git clone without proxy", "files": {name: digest(TETA_ROOT / name) for name in files},
                "exhaustive": False, "cluster_threshold": 50, "max_predictions_per_image": 300}
    (ROOT / "outputs/unified/u2/protocol/teta_source.json").write_text(json.dumps(manifest, indent=2) + "\n")
    view = prepare_tao_validation_gt(ROOT / "outputs/unified/u2/teta/gt")
    print(json.dumps({key: view[key] for key in ("status", "view_images", "view_videos", "source_gt_units", "view_gt_units", "unmapped_gt_units", "source_tracks", "view_tracks")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
