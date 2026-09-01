"""Materialize the L18 four-state counts on the fixed train-val split."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from tools.eval_l18_carr import metadata, trainval_queries
from tools.train_l18_carr import BankStore


STATE_NAMES = ("ABSENT", "MAIN_COVERED", "RESERVE_COVERED",
               "PRESENT_UNCOVERED")


def bank_frame_sets(bank: dict):
    tensors = bank["tensors"]
    candidate_gt = bank["candidate_gt"]
    pool = tensors.get("pool_id")
    frame_sets = []
    for fi, frame in enumerate(tensors["frame_ids"].tolist()):
        begin = int(tensors["frame_ptr"][fi])
        end = int(tensors["frame_ptr"][fi + 1])
        main, reserve = set(), set()
        for index in range(begin, end):
            value = candidate_gt[index]
            if value is None:
                continue
            target = main if pool is None or int(pool[index]) == 0 else reserve
            target.add(str(value))
        frame_sets.append((int(frame), main, reserve))
    return frame_sets


def main():
    store = BankStore(ROOT / "outputs/l18/dual_banks", cache_size=1)
    total = Counter()
    by_domain = defaultdict(Counter)
    query_counts = {}
    for kind, domain in (("trainval_kitti", "kitti"),
                         ("trainval_dance", "dance")):
        queries, _gt, _seqmap, _sequences, _protocol = trainval_queries(kind)
        grouped = defaultdict(list)
        for video, expression, spec in queries:
            grouped[video].append((expression, spec))
        query_counts[domain] = len(queries)
        for video, entries in sorted(grouped.items()):
            bank = store.get("kitti" if domain == "kitti" else "dance_eval",
                             video)
            frame_sets = bank_frame_sets(bank)
            lookup = metadata("kitti_v2" if domain == "kitti" else "dance")
            for expression, _spec in entries:
                entry = lookup[(video, expression)]
                labels = entry.get("label", {})
                for frame, main, reserve in frame_sets:
                    values = labels.get(str(frame), labels.get(frame, []))
                    target = {str(value) for value in values}
                    if not target:
                        state = 0
                    elif target.intersection(main):
                        state = 1
                    elif target.intersection(reserve):
                        state = 2
                    else:
                        state = 3
                    total[STATE_NAMES[state]] += 1
                    by_domain[domain][STATE_NAMES[state]] += 1
    result = {
        "protocol": "train_val only; no official evaluation queries",
        "query_counts": query_counts,
        "states": dict(total),
        "by_domain": {key: dict(value) for key, value in by_domain.items()},
        "state_order": list(STATE_NAMES),
    }
    output = ROOT / "outputs/l18/audit/four_state_trainval.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
