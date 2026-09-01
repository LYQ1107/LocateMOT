"""Choose one pooled validation calibration for KITTI and Dance."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from tools.eval_l19_ablation import (  # noqa: E402
    MODES, derive_norm_stats, derive_query_stats, load_caches, threshold_grid,
)
from tools.eval_l18_carr import trainval_queries  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti-cache-root", required=True)
    parser.add_argument("--dance-cache-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    all_caches = []
    counts = {}
    for dataset, root in (("trainval_kitti", args.kitti_cache_root),
                          ("trainval_dance", args.dance_cache_root)):
        queries, _gt, _seqmap, _sequences, _kind = trainval_queries(dataset)
        caches = load_caches((ROOT / root).resolve(), dataset, queries)
        counts[dataset] = {"queries": len(queries),
                           "candidate_rows": int(sum(len(data["raw"])
                                                      for _v, _e, _p, data in caches))}
        all_caches.extend(caches)
    norm_stats = derive_norm_stats(all_caches)
    query_stats = derive_query_stats(all_caches)
    thresholds = threshold_grid(all_caches, MODES, norm_stats, query_stats)
    payload = {
        "protocol": "pooled train_val KITTI + Dance; one threshold per score calibration",
        "datasets": counts, "thresholds": thresholds,
        "source_norm_stats": norm_stats,
        "query_norm_stats": query_stats,
    }
    output = (ROOT / args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"datasets": counts, "thresholds": thresholds}, indent=2))
    print(f"output={output}")


if __name__ == "__main__":
    main()
