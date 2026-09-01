#!/usr/bin/env python3
"""Train-only L42 current-frame grounding smoke.

The CLIP crop tensors are transient.  A small deterministic sample is kept in
RAM for the 100-step smoke, never serialized as a feature bank.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from locatemot.models.l29_frame_membership_set_decoder import L29FrameMembershipSetDecoder
from locatemot.models.l42_current_frame_grounding import L42CurrentFrameGrounding
from tools.audit_l28_identity_bank import load_labels
from tools.l40_raw_data import RAW_ROOT, WEIGHTS, crop_box, image_path
from tools.train_l26_crossmodal_adapter import EXP, SPLIT, V5, load_expressions
from tools.train_l28_track_set_decoder import state_at

L19 = ROOT / "outputs/l19/dual_banks_features/kitti"
L29 = ROOT / "outputs/l29/train/frame_membership_step1000/checkpoint_frame_membership_step1000.pt"
L28 = ROOT / "outputs/l28/track_sequence_bank_final"
FAST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
TRAIN_VIDEOS = ("0000", "0001", "0002", "0003", "0006", "0007", "0008", "0009", "0010", "0012", "0014", "0015", "0016", "0017", "0020")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def load_bank(video):
    path = L19 / f"{video}.pt"; d = torch.load(path, map_location="cpu", weights_only=False); t = d["tensors"]
    labels, label_path = load_labels(path, int(t["track_id"].numel()), tensors=t)
    return {"box": t["box"].float(), "frame": t["frame"].long(), "track": t["track_id"].long(),
            "geometry": t["geometry"].float(), "motion": t["motion"].float(), "lifecycle": t["lifecycle"].float(),
            "context": t["context"].float(), "objectness": t["objectness"].float(), "history_clip": t["history_clip"].float(),
            "frame_ids": t["frame_ids"].long(), "ptr": t["frame_ptr"].long(), "labels": labels, "label_path": str(label_path)}


def numeric_for(bank, rows):
    h = bank["history_clip"][rows]
    hs = torch.stack((h.mean(1), h.std(1), h.norm(dim=1) / 100.0, h.abs().mean(1)), 1)
    return torch.cat((bank["geometry"][rows], bank["motion"][rows], bank["lifecycle"][rows],
                      bank["context"][rows], bank["objectness"][rows, None], hs), 1)


class StreamingCropPatchEncoder:
    def __init__(self, device, batch_size=32):
        import clip
        self.device = torch.device(device); self.batch_size = int(batch_size)
        self.model, self.preprocess = clip.load(str(WEIGHTS), device=self.device)
        self.model.eval()
        for p in self.model.parameters(): p.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, video, bank, rows):
        pixels = []
        for row in rows:
            path = image_path(video, int(bank["frame"][row]))
            with Image.open(path) as im:
                im = im.convert("RGB"); box = crop_box(bank["box"][row].tolist(), im.width, im.height)
                pixels.append(self.preprocess(im.crop(box)))
        outputs = []
        for start in range(0, len(pixels), self.batch_size):
            x = torch.stack(pixels[start:start + self.batch_size]).to(self.device)
            visual = self.model.visual; x = x.to(dtype=visual.conv1.weight.dtype)
            x = visual.conv1(x).reshape(x.shape[0], visual.conv1.out_channels, -1).permute(0, 2, 1)
            cls = visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
            x = torch.cat((cls, x), 1) + visual.positional_embedding.to(x.dtype)
            x = visual.ln_pre(x).permute(1, 0, 2); x = visual.transformer(x).permute(1, 0, 2)[:, 1:]
            side = int(round(x.shape[1] ** .5)); x = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], side, side)
            outputs.append(F.adaptive_avg_pool2d(x.float(), (2, 2)).flatten(2).transpose(1, 2).cpu().half())
            del x
        return torch.cat(outputs, 0)


def load_queries():
    train = {str(x) for x in json.loads(SPLIT.read_text())["kitti_v2"]["train"]}
    tm = json.loads((V5 / "text_manifest.json").read_text())["expressions"]
    index = {(str(x["video"]), str(x["expression"])): int(x["query_index"]) for x in tm}
    out = []
    for x in load_expressions():
        key = (str(x["video"]), str(x["expression"]))
        if key[0] in train and key in index:
            out.append({"video": key[0], "expression": key[1], "text_index": index[key],
                        "target": {int(k): {str(v) for v in vals} for k, vals in x.get("label", {}).items()}})
    if len(out) != 7757: raise AssertionError(f"train expression count {len(out)} != 7757")
    return out


def make_units(queries, banks, limit=32):
    buckets = {x: [] for x in ("multi_positive", "positive", "inactive", "other")}; seen = set()
    for q in queries:
        b = banks[q["video"]]
        for fi, frame in enumerate(b["frame_ids"].tolist()):
            key = (q["video"], q["expression"], int(frame))
            if key in seen: continue
            begin, end = int(b["ptr"][fi]), int(b["ptr"][fi + 1]); target = q["target"].get(int(frame), set())
            y = np.asarray([b["labels"][r] is not None and str(b["labels"][r]) in target for r in range(begin, end)], bool)
            bucket = "multi_positive" if y.sum() > 1 else "positive" if y.any() else "inactive" if not target else "other"
            buckets[bucket].append((q, fi, y)); seen.add(key)
            if all(len(v) >= limit for v in buckets.values()): break
        if all(len(v) >= limit for v in buckets.values()): break
    chosen = []
    for name in ("multi_positive", "positive", "inactive", "other"):
        chosen.extend(buckets[name][:max(1, limit // 4)])
    if len(chosen) < limit:
        for name in buckets:
            chosen.extend(buckets[name][len(chosen):limit])
            if len(chosen) >= limit: break
    if len(chosen) < 8: raise RuntimeError("not enough train smoke units")
    return chosen[:limit]


def teacher_for(l29, cache, q, frame, bank, rows, hidden, mask, device):
    obs, om, ot, _, _ = state_at(cache, int(frame), history=8)
    with torch.inference_mode():
        encoded = l29.encode_observations(obs.to(device), om.to(device), ot.to(device))
        z = l29.forward_encoded(encoded, encoded[1], hidden[q["text_index"]].to(device), mask[q["text_index"]].to(device))
    values = {int(t): float(s) for t, s in zip(cache["track_ids"].tolist(), z["current_membership_logits"].float().cpu().tolist())}
    return torch.tensor([values.get(int(bank["track"][r]), -20.0) for r in rows], dtype=torch.float32)


def balanced_bce(s, y):
    if not len(s): return s.new_zeros(())
    p, n = y.bool(), ~y.bool(); terms = []
    if p.any(): terms.append(F.binary_cross_entropy_with_logits(s[p], y[p].float()))
    if n.any(): terms.append(F.binary_cross_entropy_with_logits(s[n], y[n].float()))
    return torch.stack(terms).mean()


def unit_loss(model, unit, hidden, mask, device):
    patch = unit["patch"].unsqueeze(0).to(device).float(); numeric = unit["numeric"].unsqueeze(0).to(device).float()
    cm = torch.ones((1, patch.shape[1]), dtype=torch.bool, device=device)
    qh = hidden[unit["query"]["text_index"]].unsqueeze(0).to(device); qm = mask[unit["query"]["text_index"]].unsqueeze(0).to(device)
    teacher = unit["teacher"].unsqueeze(0).to(device); out = model(patch, qh, numeric, cm, qm, teacher); s = out["s_expr"][0]; s.retain_grad()
    y = torch.as_tensor(unit["y"], dtype=torch.bool, device=device); pos = torch.nonzero(y).flatten(); neg = torch.nonzero(~y).flatten(); zero = s.new_zeros(())
    pre = neg[torch.argsort(unit["objectness"].to(device)[neg], descending=True)[:min(96, len(neg))]] if len(neg) else neg
    with torch.no_grad(): hard = pre[torch.argsort(s.detach()[pre], descending=True)[:min(24, len(pre))]] if len(pre) else pre
    bce = balanced_bce(s, y); hb = F.binary_cross_entropy_with_logits(s[hard], torch.zeros_like(s[hard])) if len(hard) else zero
    pair = F.softplus(.2 + s[hard][None, :] - s[pos][:, None]).mean() if len(pos) and len(hard) else zero
    listwise = torch.logsumexp(s, 0) - torch.logsumexp(s[pos], 0) if len(pos) else zero
    minpos = F.binary_cross_entropy_with_logits(s[pos], torch.ones_like(s[pos])) if len(pos) else zero
    distill = F.huber_loss(s, teacher[0], delta=1.0); inactive = balanced_bce(s, torch.zeros_like(y)) if not len(pos) else zero
    brier = ((torch.sigmoid(s) - y.float()) ** 2).mean(); quality = F.binary_cross_entropy_with_logits(out["q_conf"][0], y.float())
    total = bce + hb + pair + .5 * listwise + .5 * minpos + .25 * distill + inactive + .05 * brier + .1 * quality
    part = {"total": float(total.detach()), "membership_bce": float(bce.detach()), "hard_bce": float(hb.detach()), "pairwise": float(pair.detach()), "listwise": float(listwise.detach()), "min_positive": float(minpos.detach()), "teacher_distillation": float(distill.detach()), "inactive": float(inactive.detach()), "brier": float(brier.detach()), "quality": float(quality.detach()), "positive_count": int(y.sum()), "hard_count": int(len(hard))}
    return total, part, s, y, hard


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-root", required=True); ap.add_argument("--steps", type=int, default=100); ap.add_argument("--seed", type=int, default=20260829); ap.add_argument("--device", default="cuda:0"); args = ap.parse_args()
    assert Path.cwd().resolve() == ROOT
    out = Path(args.out_root); out = out if out.is_absolute() else ROOT / out
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device); queries = load_queries(); banks = {v: load_bank(v) for v in TRAIN_VIDEOS}
    text = torch.load(V5 / "text_tokens.pt", map_location="cpu", weights_only=False); hidden, mask = text["token_hidden"].float(), text["attention_mask"].bool(); del text
    caches = {v: torch.load(L28 / f"{v}.pt", map_location="cpu", weights_only=False) for v in TRAIN_VIDEOS}
    l29 = L29FrameMembershipSetDecoder().to(device); l29.load_state_dict(torch.load(L29, map_location=device, weights_only=False)["model"]); l29.eval()
    meta_units = make_units(queries, banks, 32); enc = StreamingCropPatchEncoder(device); units=[]
    for q, fi, y in meta_units:
        b = banks[q["video"]]; begin, end = int(b["ptr"][fi]), int(b["ptr"][fi + 1]); rows = list(range(begin, end))
        units.append({"query":q, "frame":int(b["frame_ids"][fi]), "y":y, "objectness":b["objectness"][rows].cpu(), "numeric":numeric_for(b, rows).cpu(), "patch":enc.encode(q["video"], b, rows), "teacher":teacher_for(l29, caches[q["video"]], q, int(b["frame_ids"][fi]), b, rows, hidden, mask, device)})
    del enc, l29, caches, banks
    model = L42CurrentFrameGrounding(hidden=128, heads=4, layers=2).to(device); opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4); rng=np.random.default_rng(args.seed); trace=[]; grads=[]; pgr=[]; hgr=[]; start=time.time(); model.train()
    for _ in range(args.steps):
        u=units[int(rng.integers(len(units)))]; loss, part, s, y, hard=unit_loss(model,u,hidden,mask,device); opt.zero_grad(set_to_none=True); loss.backward(); grads.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.0))); sg=s.grad.detach().abs(); pi=torch.nonzero(y).flatten(); pgr.append(float((sg[pi]>1e-10).float().mean()) if len(pi) else 0.0); hgr.append(float((sg[hard]>1e-10).float().mean()) if len(hard) else 0.0); opt.step(); trace.append({k:part[k] for k in ("total","membership_bce","hard_bce","pairwise","listwise","min_positive","teacher_distillation","inactive","brier","quality")})
    ck=out/f"checkpoint_l42_current_frame_step{args.steps}.pt"; payload={"format":"locatemot-l42-current-frame-grounding-v1","stage":"train-only-smoke","seed":args.seed,"steps":args.steps,"device":str(device),"train_video_count":len(TRAIN_VIDEOS),"train_query_count":len(queries),"sampled_unit_count":len(units),"sample_categories":{"multi_positive":sum(int(x[2].sum()>1) for x in meta_units),"positive":sum(int(x[2].sum()==1) for x in meta_units),"inactive":sum(int(not x[2].any()) for x in meta_units)},"screening_gt_used_for_fit":False,"fixed_fast_manifest_used_for_training":False,"fast_manifest_sha256":sha(FAST),"semantic_inputs_excluded":["source_id","pool_id","group_id","state_key"],"token_level_alignment_verified":False,"motion_language_decomposition":"not claimed; no verified motion-language mask","model_config":model.config,"crop_contract":{"encoder":"frozen CLIP ViT-B/16","weights":str(WEIGHTS),"weights_sha256":sha(WEIGHTS),"patch_tokens_per_candidate":4,"crop_rule":"10 percent padding, clipped, transient pixels only"},"teacher":{"checkpoint":str(L29.resolve()),"sha256":sha(L29),"weight":0.25,"role":"distillation/control only"},"loss_contract":{"frame_balanced_bce":True,"online_hard_prefilter":96,"online_hard_model_topk":24,"pairwise_margin":0.2,"multi_positive_all_gradients":True,"teacher_huber_weight":0.25,"inactive_no_target":True,"brier_regularizer":True},"loss_mean":{k:float(np.mean([x[k] for x in trace])) for k in trace[0]},"gradient_norm":{"mean":float(np.mean(grads)),"max":float(np.max(grads)),"nonzero_steps":int(np.count_nonzero(np.asarray(grads)>0))},"gradient_audit":{"positive_nonzero_fraction_mean":float(np.mean(pgr)),"hard_nonzero_fraction_mean":float(np.mean(hgr))},"elapsed_sec":time.time()-start}
    torch.save({"model":model.state_dict(),"config":payload},ck); payload["checkpoint"]=str(ck.resolve()); reload=L42CurrentFrameGrounding(**model.config); reload.load_state_dict(torch.load(ck,map_location="cpu",weights_only=False)["model"]); payload["checkpoint_reload"]=True; (out/f"metrics_l42_smoke{args.steps}.json").write_text(json.dumps(payload,indent=2)+"\n"); (out/"loss_trace.json").write_text(json.dumps(trace,indent=2)+"\n"); (out/"config.json").write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(payload,indent=2),flush=True)


if __name__ == "__main__": main()
