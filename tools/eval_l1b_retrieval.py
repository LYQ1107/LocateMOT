"""Stage L1-B raw ObjectToken retrieval baselines (R0-R3).

Uses the pilot cache (safetensors) and matched candidate->GT identity
labels.  Metrics: Same-Category R@1/5/10, mAP, ROC-AUC, PR-AUC.
R4 (IdentityToken) is evaluated after the adapter is trained.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")

from locatemot.data.token_cache import read_frame_cache  # noqa: E402

CACHE_ROOT = Path(
    "/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L1B/cache_dla")
FEATURES = {
    "R0_pbd_box_end": "pbd_box_end_last",
    "R1_pbd_coord": "pbd_coord_mean_last",
    "R2_region": "region",
    "R3_fused": "fused",
}


def _l2(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-9)


def _map_at_r(scores, labels, ks=(1, 5, 10)):
    order = np.argsort(scores)[::-1]
    labs = labels[order]
    n_pos = int(labs.sum())
    out = {"R@1": 0, "R@5": 0, "R@10": 0, "mAP": 0.0}
    if n_pos == 0:
        return out
    out["R@1"] = int(labs[0] == 1)
    for k in (5, 10):
        out[f"R@{k}"] = int(labs[:k].sum() > 0)
    hits = 0
    prec = []
    for i, lab in enumerate(labs, 1):
        if lab == 1:
            hits += 1
            prec.append(hits / i)
    out["mAP"] = float(np.mean(prec))
    return out


def _roc_pr(scores, labels):
    from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
    if len(np.unique(labels)) < 2:
        return 0.0, 0.0
    try:
        roc = roc_auc_score(labels, scores)
    except ValueError:
        roc = 0.0
    p, r, _ = precision_recall_curve(labels, scores)
    pr = auc(r, p)
    return float(roc), float(pr)


def collect(cache_root, datasets, features_map=None, token_model=None):
    """group identity observations -> (dataset, video, gt_id, category,
    frame, feature dict)."""
    groups = defaultdict(list)
    meta_cat = {}
    for ds in datasets:
        base = cache_root / ds
        if not base.exists():
            continue
        for meta_path in base.rglob("*.meta.json"):
            rel = meta_path.relative_to(cache_root)
            key = str(rel).rsplit(".meta.json", 1)[0]
            data = read_frame_cache(str(cache_root), key)
            if data is None:
                continue
            feats = data["features"]
            meta = data["meta"]
            matched = meta.get("matched_candidates", {})
            if not matched:
                continue
            vecs = {}
            for name, fk in (features_map or FEATURES).items():
                if name == "R4_identity":
                    continue
                arr = feats.get(fk)
                if arr is not None and len(arr) > 0:
                    arr = arr.numpy() if hasattr(arr, "numpy") \
                        else np.asarray(arr)
                    vecs[name] = arr
            if token_model is not None and vecs:
                for extra in ("geometry", "gen_score"):
                    arr = feats.get(extra)
                    if arr is not None:
                        arr = arr.numpy() if hasattr(arr, "numpy") \
                            else np.asarray(arr)
                        if extra == "gen_score" and arr.ndim == 1:
                            arr = arr.reshape(-1, 1)
                        vecs[extra] = arr
                import torch
                ok = all(k in vecs for k in (
                    "R0_pbd_box_end", "R1_pbd_coord", "R2_region"))
                if ok:
                    dev = next(token_model.parameters()).device
                    def t(x):
                        return torch.from_numpy(
                            np.asarray(x, dtype=np.float32)).to(dev)
                    toks = []
                    for i in range(len(vecs["R0_pbd_box_end"])):
                        tok = token_model(
                            t(vecs["R0_pbd_box_end"][i]).unsqueeze(0),
                            t(vecs["R1_pbd_coord"][i]).unsqueeze(0),
                            t(vecs["R2_region"][i]).unsqueeze(0),
                            t(vecs["geometry"][i]).unsqueeze(0),
                            t(vecs["gen_score"][i]).unsqueeze(0))
                        toks.append(tok[0].detach().cpu().numpy())
                    vecs["R4_identity"] = np.stack(toks)
            if not vecs:
                continue
            vid = meta.get("video_id")
            for gt_id, m in matched.items():
                ci = int(m["candidate"])
                if ci >= len(feats.get("boxes", [])):
                    continue
                groups[(ds, str(vid), gt_id)].append({
                    "frame": meta.get("frame"), "idx": ci,
                    "feats": {k: v[ci] for k, v in vecs.items()},
                    "category": meta.get("query", ""),
                })
    return groups


def category_map(cache_root, datasets):
    """gt identity -> category for multi-class datasets."""
    out = {}
    if "tao_amodal" in datasets:
        d = json.loads(
            (Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/"
                  "TAO-Amodal/annotations/train.json")).read_text())
        cat_name = {c["id"]: c["name"] for c in d["categories"]}
        for a in d["annotations"]:
            im = next((x for x in d["images"] if x["id"] == a["image_id"]),
                      None)
            if im is not None:
                out[("tao_amodal", im["video"], str(a["track_id"]))] = \
                    cat_name.get(a["category_id"], "object")
    if "ytvos" in datasets:
        m = json.loads(
            (Path("/data3/testdata/vranlee/.MOTSynth.partial/"
                  "YouTube-VOS-2019/train/meta.json")).read_text())
        for vid, info in m["videos"].items():
            for oid, o in info.get("objects", {}).items():
                out[("ytvos", vid, str(oid))] = o.get("category", "object")
    if "bdd100k" in datasets:
        lab_root = Path(
            "/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/bdd/"
            "annotations/box_track_20/train")
        for p in lab_root.glob("*.json"):
            for fr in json.loads(p.read_text()):
                for lab in fr.get("labels", []):
                    if lab.get("box2d") is not None:
                        out[("bdd100k", p.stem, str(lab["id"]))] = \
                            lab.get("category", "object")
    return out


def eval_same_category(groups, catmap, datasets, features_map=None,
                       max_queries=300):
    """Same-category gallery retrieval per dataset + cross-category."""
    rows = []
    for ds in datasets:
        sub = {k: v for k, v in groups.items() if k[0] == ds}
        keys = list(sub)
        if not keys:
            continue
        rng = np.random.RandomState(20260806)
        rng.shuffle(keys)
        keys = keys[:max_queries]
        for feat_name in (features_map or FEATURES):
            sub_obs = {k: [o for o in v if feat_name in o["feats"]]
                       for k, v in sub.items()}
            agg_same = []
            agg_cross = []
            for key in keys:
                obs = sub_obs[key]
                if len(obs) < 2:
                    continue
                cat = catmap.get(key)
                q = obs[0]
                qv = q["feats"][feat_name].astype(np.float32)
                qn = qv / max(float(np.linalg.norm(qv)), 1e-9)
                gallery_obs = obs[1:]
                gv = np.stack([o["feats"][feat_name].astype(np.float32)
                               for o in gallery_obs])
                gn = _l2(gv)
                scores = gn @ qn
                labels = np.ones(len(gallery_obs), dtype=np.int64)
                # always add hard (same-video) + easy (cross-video) negatives
                hard = []
                easy = []
                for k2 in keys:
                    if k2 == key or not sub_obs[k2]:
                        continue
                    if k2[1] == key[1]:
                        hard.append(sub_obs[k2][0])
                    else:
                        easy.append(sub_obs[k2][0])
                negs = []
                for o2 in hard[:4] + easy[:4]:
                    v = o2["feats"][feat_name].astype(np.float32)
                    negs.append(v / max(float(np.linalg.norm(v)), 1e-9))
                if negs:
                    scores = np.concatenate([
                        scores, np.asarray([float(v @ qn) for v in negs],
                                           dtype=np.float64)])
                    labels = np.concatenate([
                        labels, np.zeros(len(negs), dtype=np.int64)])
                if cat is not None:
                    same = []
                    cross = []
                    for k2 in keys:
                        if k2 == key or not sub_obs[k2]:
                            continue
                        c2 = catmap.get(k2)
                        if c2 == cat:
                            same.append(sub_obs[k2][0])
                        else:
                            cross.append(sub_obs[k2][0])
                    same_vecs = []
                    for o2 in same[:6]:
                        v = o2["feats"][feat_name].astype(np.float32)
                        same_vecs.append(
                            v / max(float(np.linalg.norm(v)), 1e-9))
                    sc_same = np.concatenate([
                        gn @ qn,
                        np.asarray([float(v @ qn) for v in same_vecs],
                                   dtype=np.float64)])
                    lc_same = np.concatenate([
                        np.ones(len(gallery_obs), dtype=np.int64),
                        np.zeros(len(same_vecs), dtype=np.int64)])
                    m = _map_at_r(sc_same, lc_same)
                    roc, pr = _roc_pr(sc_same, lc_same)
                    agg_same.append({**m, "roc": roc, "pr": pr})
                    # cross-category: only cross negatives + positives
                    cross_vecs = []
                    for o2 in cross[:6]:
                        v = o2["feats"][feat_name].astype(np.float32)
                        cross_vecs.append(
                            v / max(float(np.linalg.norm(v)), 1e-9))
                    if cross_vecs:
                        sc = np.concatenate([
                            gn @ qn,
                            np.asarray([float(v @ qn) for v in cross_vecs],
                                       dtype=np.float64)])
                        lc = np.concatenate([
                            np.ones(len(gallery_obs), dtype=np.int64),
                            np.zeros(len(cross_vecs), dtype=np.int64)])
                        mc = _map_at_r(sc, lc)
                        roc_c, pr_c = _roc_pr(sc, lc)
                        agg_cross.append({**mc, "roc": roc_c, "pr": pr_c})
                else:
                    # single-category datasets: mixed == same-category
                    m = _map_at_r(scores, labels)
                    roc, pr = _roc_pr(scores, labels)
                    agg_same.append({**m, "roc": roc, "pr": pr})
                    agg_cross.append({**m, "roc": roc, "pr": pr})
            for tag, agg in (("same_cat", agg_same),
                             ("cross_cat", agg_cross)):
                if not agg:
                    continue
                macro = {k: float(np.mean([a[k] for a in agg]))
                         for k in ("R@1", "R@5", "R@10", "mAP", "roc",
                                   "pr")}
                rows.append({"dataset": ds, "feature": feat_name,
                             "gallery": tag, "queries": len(agg),
                             **macro})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    ap.add_argument("--datasets", default="dancetrack,mot17,mot20,"
                                          "tao_amodal,bdd100k,ytvos,mose")
    ap.add_argument("--max-queries-per-dataset", type=int, default=400)
    ap.add_argument("--identity-ckpt", type=str, default="")
    args = ap.parse_args()
    datasets = args.datasets.split(",")
    token_model = None
    features_map = FEATURES
    if args.identity_ckpt:
        import torch
        from locatemot.models.identity.identity_adapter import (
            IdentityAdapter)
        ck = torch.load(args.identity_ckpt, map_location="cpu")
        token_model = IdentityAdapter(
            input_mode=ck.get("input_mode", "full"))
        token_model.load_state_dict(ck["state_dict"])
        token_model.eval()
        dev = torch.device("cuda:0" if torch.cuda.is_available()
                           else "cpu")
        token_model.to(dev)
        features_map = {**FEATURES, "R4_identity": "R4_identity"}
    groups = collect(args.cache_root, datasets, features_map, token_model)
    print("identity groups:", len(groups), flush=True)
    rows = []
    for ds in datasets:
        sub = {k: v for k, v in groups.items() if k[0] == ds}
        keys = list(sub)
        if not keys:
            print(ds, "no groups")
            continue
        rng = np.random.RandomState(20260806)
        rng.shuffle(keys)
        keys = keys[:args.max_queries_per_dataset]
        for feat_name in features_map:
            all_s = []
            all_l = []
            agg = []
            sub_obs = {k: [o for o in v if feat_name in o["feats"]]
                       for k, v in sub.items()}
            for key in keys:
                obs = sub_obs[key]
                if len(obs) < 2:
                    continue
                q = obs[0]
                gallery = obs[1:]
                qv = q["feats"][feat_name].astype(np.float32)
                qn = qv / max(float(np.linalg.norm(qv)), 1e-9)
                gv = np.stack([o["feats"][feat_name].astype(np.float32)
                               for o in gallery])
                gn = _l2(gv)
                scores = gn @ qn
                labels = np.ones(len(gallery), dtype=np.int64)
                # hard negative: other identities in same video
                hard = []
                for k2 in keys:
                    if k2[1] == key[1] and k2[2] != key[2] and \
                            sub_obs[k2]:
                        hard.append(sub_obs[k2][0])
                for o2 in hard[:4]:
                    v = o2["feats"][feat_name].astype(np.float32)
                    v = v / max(float(np.linalg.norm(v)), 1e-9)
                    scores = np.concatenate([scores, [float(v @ qn)]])
                    labels = np.concatenate([labels, [0]])
                # easy negative: other videos
                easy = []
                for k2 in keys:
                    if k2[1] != key[1] and sub_obs[k2]:
                        easy.append(sub_obs[k2][0])
                for o2 in easy[:4]:
                    v = o2["feats"][feat_name].astype(np.float32)
                    v = v / max(float(np.linalg.norm(v)), 1e-9)
                    scores = np.concatenate([scores, [float(v @ qn)]])
                    labels = np.concatenate([labels, [0]])
                m = _map_at_r(scores, labels)
                roc, pr = _roc_pr(scores, labels)
                agg.append({**m, "roc": roc, "pr": pr})
                all_s.extend(scores.tolist())
                all_l.extend(labels.tolist())
            if not agg:
                continue
            macro = {k: float(np.mean([a[k] for a in agg]))
                     for k in ("R@1", "R@5", "R@10", "mAP", "roc", "pr")}
            rows.append({"dataset": ds, "feature": feat_name,
                         "queries": len(agg), **macro})
            print(ds, feat_name, {k: round(v, 4) for k, v in macro.items()},
                  flush=True)
    if rows:
        import csv
        out = ROOT / "outputs" / "l1_b" / "raw_token_retrieval.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("RETRIEVAL_DONE", out)
    catmap = category_map(args.cache_root, datasets)
    rows2 = eval_same_category(groups, catmap, datasets, features_map)
    if rows2:
        import csv
        out2 = ROOT / "outputs" / "l1_b" / "same_category_retrieval.csv"
        with open(out2, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows2[0]))
            w.writeheader()
            w.writerows(rows2)
        print("SAME_CATEGORY_DONE", out2)


if __name__ == "__main__":
    main()
