"""Stage L1-B pilot training: Universal Identity Adapter (InfoNCE).

Reads the pilot ObjectToken cache, builds identity units with
same-category hard negatives, trains one shared adapter (seed 20260806),
saves outputs/l1_b/checkpoints/identity_adapter.pt.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.data.token_cache import read_frame_cache  # noqa: E402
from locatemot.models.identity.identity_adapter import (  # noqa: E402
    IdentityAdapter,
    infonce_loss,
)

CACHE = Path(
    "/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L1B/cache_dla")
SEED = 20260806
REQUIRED = ("pbd_box_end_last", "pbd_coord_mean_last", "geometry",
            "gen_score")
REGION_DIM = 4608


def collect(cache_root):
    obs = []
    for ds_dir in sorted(cache_root.iterdir()):
        if not ds_dir.is_dir():
            continue
        ds = ds_dir.name
        for meta_path in ds_dir.rglob("*.meta.json"):
            rel = meta_path.relative_to(cache_root)
            key = str(rel).rsplit(".meta.json", 1)[0]
            data = read_frame_cache(str(cache_root), key)
            if data is None:
                continue
            feats = data["features"]
            meta = data["meta"]
            for gt_id, m in meta.get("matched_candidates", {}).items():
                ci = int(m["candidate"])
                vec = {}
                ok = True
                for name in REQUIRED:
                    arr = feats.get(name)
                    if arr is None:
                        ok = False
                        break
                    arr = arr.numpy() if hasattr(arr, "numpy") \
                        else np.asarray(arr)
                    if ci >= len(arr):
                        ok = False
                        break
                    val = np.asarray(arr[ci], dtype=np.float32)
                    vec[name] = val.reshape(1) if val.ndim == 0 else val
                if not ok:
                    continue
                region = feats.get("region")
                if region is not None:
                    region = region.numpy() if hasattr(region, "numpy") \
                        else np.asarray(region)
                    if ci < len(region):
                        vec["region"] = np.asarray(region[ci],
                                                   dtype=np.float32)
                else:
                    vec["region"] = np.zeros(REGION_DIM, dtype=np.float32)
                obs.append({
                    "ds": ds, "video": str(meta.get("video_id")),
                    "gt_id": gt_id, "frame": meta.get("frame"),
                    "idx": ci, **vec,
                })
    return obs


def group_identities(obs):
    groups = defaultdict(list)
    for o in obs:
        groups[(o["ds"], o["video"], o["gt_id"])].append(o)
    return {k: sorted(v, key=lambda x: x["frame"]) for k, v in
            groups.items() if len(v) >= 2}


def to_tensor(v):
    return torch.from_numpy(np.asarray(v, dtype=np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", type=Path, default=CACHE)
    ap.add_argument("--gpu", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-identities", type=int, default=32)
    ap.add_argument("--hard-negatives", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--per-dataset-cap", type=int, default=60)
    ap.add_argument("--input-mode", choices=["full", "pbd"], default="full")
    ap.add_argument("--datasets", default="",
                    help="comma list to restrict training datasets "
                         "(default: all)")
    ap.add_argument("--tag", default="identity_adapter",
                    help="checkpoint name prefix")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "outputs/l1_b/checkpoints")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    obs = collect(args.cache_root)
    groups = group_identities(obs)
    if args.datasets:
        ds_set = set(args.datasets.split(","))
        groups = {k: v for k, v in groups.items() if k[0] in ds_set}
    print("observations:", len(obs), "identities:", len(groups),
          flush=True)
    by_ds = Counter(k[0] for k in groups)
    print("identities by dataset:", dict(by_ds), flush=True)
    device = torch.device(f"cuda:{args.gpu}")
    model = IdentityAdapter(input_mode=args.input_mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    args.out.mkdir(parents=True, exist_ok=True)
    keys = list(groups)
    rng = random.Random(SEED)
    history = []
    for epoch in range(1, args.epochs + 1):
        # dataset-balanced identity sampling
        per_ds = defaultdict(list)
        for k in keys:
            per_ds[k[0]].append(k)
        chosen = []
        for ds, ks in per_ds.items():
            rng.shuffle(ks)
            chosen.extend(ks[:args.per_dataset_cap])
        rng.shuffle(chosen)
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for s in range(0, len(chosen), args.batch_identities):
            batch_keys = chosen[s:s + args.batch_identities]
            anchors, poss, negs = [], [], []
            for k in batch_keys:
                g = groups[k]
                if len(g) < 2:
                    continue
                a = g[0]
                p = g[1]
                # hard negatives: same video first, then others
                neg_obs = []
                for k2 in keys:
                    if k2 == k:
                        continue
                    if k2[0] == k[0] and k2[1] == k[1]:
                        neg_obs.append(groups[k2][0])
                    if len(neg_obs) >= args.hard_negatives:
                        break
                for k2 in keys:
                    if len(neg_obs) >= args.hard_negatives:
                        break
                    if k2 != k and (k2[0] != k[0] or k2[1] != k[1]):
                        neg_obs.append(groups[k2][0])
                neg_obs = neg_obs[:args.hard_negatives]
                if not neg_obs:
                    continue
                anchors.append(a)
                poss.append(p)
                negs.append(neg_obs)
            if not anchors:
                continue

            def stack(items, key):
                return torch.stack([to_tensor(x[key]) for x in items]).to(
                    device)

            A = model(stack(anchors, "pbd_box_end_last"),
                      stack(anchors, "pbd_coord_mean_last"),
                      stack(anchors, "region"),
                      stack(anchors, "geometry"),
                      stack(anchors, "gen_score"))
            P = model(stack(poss, "pbd_box_end_last"),
                      stack(poss, "pbd_coord_mean_last"),
                      stack(poss, "region"),
                      stack(poss, "geometry"),
                      stack(poss, "gen_score"))
            N = []
            for i in range(args.hard_negatives):
                items = [n[i] for n in negs]
                N.append(model(stack(items, "pbd_box_end_last"),
                               stack(items, "pbd_coord_mean_last"),
                               stack(items, "region"),
                               stack(items, "geometry"),
                               stack(items, "gen_score")))
            N = torch.stack(N, dim=1)
            loss = infonce_loss(A, P, N, temperature=args.temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1
        avg = total_loss / max(n_batches, 1)
        history.append({"epoch": epoch, "loss": round(avg, 5),
                        "identities": len(chosen),
                        "seconds": round(time.time() - t0, 1)})
        print(f"epoch {epoch} loss {avg:.5f} identities {len(chosen)} "
              f"{time.time()-t0:.0f}s", flush=True)
    mode = "" if args.input_mode == "full" else "_pbd"
    ckpt = args.out / f"{args.tag}{mode}.pt"
    torch.save({"state_dict": model.state_dict(), "history": history,
                "seed": SEED, "d_model": model.d_model,
                "input_mode": args.input_mode}, ckpt)
    (args.out / "history.json").write_text(json.dumps(history, indent=1))
    print("L1B_ADAPTER_TRAIN_DONE", ckpt)


if __name__ == "__main__":
    main()
