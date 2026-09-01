#!/usr/bin/env python3
"""Validate L26 v5 row/frame alignment and finite cross-modal features."""
from __future__ import annotations
import json
from pathlib import Path
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
V5 = ROOT / "outputs/l26/candidate_bank_v5_crossmodal"
OLD = ROOT / "outputs/l19/dual_banks_features/kitti"
OUT = ROOT / "outputs/l26/audit/candidate_bank_v5_validation"


def main():
    summary = json.loads((V5 / "build_summary.json").read_text())
    videos = sorted(summary["videos"])
    checks = {"video_count": len(videos) == 21, "query_count": summary["query_count"] == 9778,
              "train_queries": summary["train_query_count"] == 7757,
              "text_present": (V5 / "text_tokens.pt").exists() and (V5 / "text_manifest.json").exists(),
              "rows": True, "frame_ptr": True, "frame_ids": True, "finite": True,
              "maps": True, "complete_markers": True}
    videos_report = {}
    total_rows = total_frames = total_maps = 0
    for video in videos:
        p = V5 / "kitti" / f"{video}.pt"
        oldp = OLD / f"{video}.pt"
        d = torch.load(p, map_location="cpu", weights_only=False)
        old = torch.load(oldp, map_location="cpu", weights_only=False)
        t, ot = d["tensors"], old["tensors"]
        row_count = int(t["box"].shape[0]); frame_count = int(t["frame_ids"].shape[0])
        row_ok = row_count == int(ot["box"].shape[0])
        ptr_ok = torch.equal(t["frame_ptr"], ot["frame_ptr"])
        ids_ok = torch.equal(t["frame_ids"], ot["frame_ids"])
        finite = all((not torch.is_floating_point(v)) or bool(torch.isfinite(v.float()).all()) for k, v in t.items() if k.startswith("dino_") or k.endswith("_v5"))
        maps = sorted((V5 / "dense_maps").glob(f"{video}_*.pt"))
        map_ok = len(maps) == frame_count and all(tuple(torch.load(m, map_location="cpu", weights_only=False)["feature_map"].shape) == (1,768,16,16) for m in maps)
        complete = (V5 / "kitti" / f"{video}.complete").exists()
        checks["rows"] &= row_ok; checks["frame_ptr"] &= ptr_ok; checks["frame_ids"] &= ids_ok; checks["finite"] &= finite; checks["maps"] &= map_ok; checks["complete_markers"] &= complete
        videos_report[video] = {"rows": row_count, "frames": frame_count, "maps": len(maps), "row_alignment": row_ok, "frame_ptr": ptr_ok, "frame_ids": ids_ok, "finite": finite, "map_shape": map_ok, "complete": complete}
        total_rows += row_count; total_frames += frame_count; total_maps += len(maps)
    text = torch.load(V5 / "text_tokens.pt", map_location="cpu", weights_only=False)
    text_ok = tuple(text["token_hidden"].shape) == (9778,64,768) and tuple(text["attention_mask"].shape) == (9778,64) and bool(torch.isfinite(text["token_hidden"].float()).all())
    checks["text_shape_finite"] = text_ok
    payload = {"format": "locatemot-l26-crossmodal-bank-v5-validation-v1", "v5_root": str(V5), "videos": videos_report, "totals": {"rows": total_rows, "frames": total_frames, "maps": total_maps}, "text": {"shape": list(text["token_hidden"].shape), "attention_shape": list(text["attention_mask"].shape), "finite": text_ok}, "checks": checks, "passed": all(checks.values())}
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "validation.json").write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / "validation.md").write_text("# L26 v5 validation\n\n" + json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"out": str(OUT), "passed": payload["passed"], "totals": payload["totals"], "checks": checks}, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__": main()
