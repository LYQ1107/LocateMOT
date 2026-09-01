"""Stage L4: rewrite the TAO manifest with explicit cache keys.

The original `tao_amodal_train.jsonl` uses the generic key
`tao_amodal/<video_id>/<frame>/pilot`, but the cached files live under
`cache_dla/tao_amodal/train/<source>/<video_id>/<frame>/pilot.*`
(source in {BDD, AVA, YFCC100M, HACS, LaSOT}).  This tool detects each
video's source once and writes a copy of the manifest with a `cache_key`
field, without touching the shared cache.

Usage:
  python tools/fix_tao_manifest.py
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(
    ROOT, "outputs/l1_c/fixed_candidate_manifest/tao_amodal_train.jsonl")
DST = os.path.join(ROOT, "outputs/l4/manifests/tao_amodal_train_l4.jsonl")
CACHE = ("/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L1B/"
         "cache_dla/tao_amodal/train")


def source_of(video_id):
    for s in sorted(os.listdir(CACHE)):
        if os.path.isdir(os.path.join(CACHE, s, video_id)):
            return s
    return None


def main():
    cache = {}
    n = 0
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    with open(SRC) as f, open(DST, "w") as g:
        for line in f:
            e = json.loads(line)
            vid = e["video_id"]
            if vid not in cache:
                cache[vid] = source_of(vid)
            src = cache[vid]
            if src is None:
                raise RuntimeError(f"no cache source for {vid}")
            e["cache_key"] = (
                f"tao_amodal/train/{src}/{vid}/"
                f"{int(e['frame']):05d}/{e['protocol']}")
            g.write(json.dumps(e, ensure_ascii=False) + "\n")
            n += 1
    print(f"[tao] {n} entries -> {DST}")


if __name__ == "__main__":
    main()
