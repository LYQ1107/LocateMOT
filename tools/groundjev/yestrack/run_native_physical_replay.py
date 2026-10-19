#!/usr/bin/env python3
"""Strict candidate-aligned YesTrack replay for a physical tracking trace.

The source trace supplies the real candidates and causal decision rows.  The
Yes/No model is queried only for candidate/track pairs that are already in the
physical association matrix, and the result is fused in the same global solve
through ``OnlineTracker``.  This tool is intentionally a short integration
receipt unless the source is a complete legal video; it never turns a short
trace into a benchmark metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ONEJEV = Path(os.environ.get("ONEJEV_ROOT", "/data2/user/LocateMOT_GroundJEV/external/OneJev"))
sys.path[:0] = [str(ROOT), str(ONEJEV)]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class RecordedDecision:
    def __init__(self, row: dict):
        self.row = row

    def decide(self, image, candidates, tracks, timestamp, specification):
        decisions = [None] * len(candidates)
        lifecycle = {}
        for event in self.row["state"]["events"]:
            if "candidate" in event:
                decisions[int(event["candidate"])] = event["decision"]
            if event.get("lifecycle_decision") is not None:
                lifecycle[int(event["track_id"])] = event["lifecycle_decision"]
        if any(value is None for value in decisions):
            raise ValueError(f"incomplete recorded decisions at frame {self.row['frame']}")
        return decisions, lifecycle, dict(self.row["state"]["decision_metadata"])


class NativeYesNoBackend:
    def __init__(self, model: Path, device: str):
        import torch
        from train import common, mm

        self.torch = torch
        self.common = common
        self.mm = mm
        self.device = torch.device(device)
        self.model_path = model
        self.bundle = common.load_bundle(str(model), device=str(self.device), dtype="bfloat16",
                                         train=False, multimodal=True, chat_template=None,
                                         load_model=True)
        self.bundle.model.eval()
        self.mm.set_root("/")
        self.sha256 = digest(model / "model.safetensors")

    def __call__(self, *, image, candidate, track, timestamp, specification, decision, reason):
        from locatemot.groundjev.yes_verification import YesNoResult

        source = Path(str(specification["image_path"]))
        history = [{
            "id": int(track.track_id), "box": list(track.box),
            "predicted_box": list(track.predicted_box(timestamp)), "age": int(track.age),
            "lost_age": float(track.lost_age), "state": track.state,
        }]
        state = {
            "current_frame": "<image:1>",
            "timestamp": float(timestamp),
            "task": specification.get("task", "mot"),
            "candidates": [{"index": 0, "box": list(candidate["box"]),
                            "label": candidate.get("label", "person")}],
            "historical_tracks": history,
            "yestrack_proposal": {"kind": "identity", "candidate_index": 0,
                                  "proposed_identity": str(track.track_id)},
        }
        row = {
            "id": f"runtime_{source.stem}_candidate_{id(candidate)}_track_{track.track_id}",
            "source": str(source), "task": "identity_pair", "state": state,
            "question": {"type": "noul", "instructions":
                f"Does proposed persistent identity {track.track_id} own the current candidate? "
                "Use only the current image, candidate geometry and causal track history. "
                "Answer yes or no."},
            # The target is used only to make the prompt renderer happy; it is
            # never included in the prompt and is not read by inference.
            "target": {"yes": 1.0},
            "media": [{"type": "image", "path": str(source)}],
        }
        rendered = self.common.render_example(row, self.bundle.template_for(row), 0,
                                              seed=None, augment=False)
        started = time.perf_counter()
        with self.torch.inference_mode():
            inputs, ends = self.mm.mm_inputs(self.bundle.processor, [rendered.prompt],
                                             [row["media"]], self.device)
            with self.torch.autocast("cuda", dtype=self.torch.bfloat16):
                logits = self.mm.mm_readout_logits(self.bundle.model, inputs, ends)
            selected = logits[:, self.bundle.slot_ids[:2]].float()
            probs = self.torch.softmax(selected, dim=-1)[0].detach().cpu().tolist()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        p_yes = float(probs[0])
        return YesNoResult(
            status="VERIFIED", used=True, p_yes=p_yes,
            answer="Yes" if p_yes >= 0.5 else "No", question_type="identity",
            evidence_frame_ids=(int(round(timestamp * 30)),),
            checkpoint_hash=self.sha256, latency_ms=elapsed_ms,
        )


def run_arm(rows: list[dict], *, verifier=None, policy=None, fps: float) -> tuple[list[dict], dict]:
    from PIL import Image
    from locatemot.groundjev.tracker import OnlineTracker

    tracker = OnlineTracker(
        RecordedDecision(rows[0]),
        memory_policy="gated", lifecycle_termination_policy="learned_or_timeout",
        no_match_policy="learned", verification_backend=verifier,
        verification_policy=policy,
    )
    outputs = []
    calls = 0
    for row in rows:
        tracker.decision.row = row
        image_path = Path(row["image_path"])
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        specification = {"task": "mot", "vocabulary": ["person"],
                         "image_path": str(image_path)}
        frame_outputs, state = tracker.step(image, row["candidates"], row["frame"] / fps,
                                            specification)
        if verifier is not None:
            calls += int((state["decision_metadata"].get("yestrack_verification") or {}).get("calls", 0))
        outputs.append({"frame": row["frame"], "outputs": frame_outputs, "state": state})
    return outputs, {"frames": len(rows), "verifier_calls": calls, "next_id": tracker.next_id}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--max-verifications-per-frame", type=int, default=2)
    args = p.parse_args()
    rows = [json.loads(line) for line in args.source.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty source trace")
    if any("native_region_feature" not in candidate for row in rows for candidate in row["candidates"]):
        raise ValueError("source must retain exact native candidate features for a physical replay")

    baseline, baseline_meta = run_arm(rows, fps=args.fps)
    if any(b["outputs"] != row["outputs"] for b, row in zip(baseline, rows)):
        raise RuntimeError("verifier-disabled replay is not identical to the source physical trace")

    from locatemot.groundjev.verification_policy import SelectiveVerificationPolicy, VerificationPolicyConfig

    backend = NativeYesNoBackend(args.model, args.device)
    policy = SelectiveVerificationPolicy(VerificationPolicyConfig(
        enabled=True, top2_margin_threshold=100.0,
        max_verifications_per_frame=args.max_verifications_per_frame,
        semantic_weight=0.25, yes_threshold=0.5,
    ))
    started = time.perf_counter()
    verified, verified_meta = run_arm(rows, verifier=backend, policy=policy, fps=args.fps)
    elapsed = time.perf_counter() - started
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "REAL_ONEJEV_YESTRACK_PHYSICAL_REPLAY_COMPLETE_SHORT_SCOPE",
        "scientific_result": False,
        "source": str(args.source.resolve()), "source_sha256": digest(args.source),
        "frames": len(rows), "model": str(args.model.resolve()),
        "model_sha256": backend.sha256, "baseline": baseline_meta,
        "verified": {**verified_meta, "elapsed_seconds": elapsed,
                     "end_to_end_fps": len(rows) / max(elapsed, 1e-9)},
        "policy": {"top2_margin_threshold": 100.0,
                   "max_verifications_per_frame": args.max_verifications_per_frame,
                   "semantic_weight": 0.25, "yes_threshold": 0.5},
        "metric_status": "NOT_RUN_SHORT_SCOPE",
        "reason": "The source is a four-frame functional control, not a complete legal video; no HOTA/IDF1/IDSW claim is permitted.",
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "baseline_outputs.json").write_text(json.dumps(baseline, indent=2) + "\n")
    (args.out / "verified_outputs.json").write_text(json.dumps(verified, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "frames", "model_sha256", "baseline", "verified", "metric_status")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
