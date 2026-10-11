#!/usr/bin/env python3
"""Causal/isolation/state audit without altering actual selected core weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def core_contract(core, clip: list[dict], class_text, directory: Path) -> dict:
    import torch
    from tools.unified.u2_causal_contract import state_equal
    from locatemot.unified.runtime.u2_online_tracker import U2OnlineTracker

    core.eval()
    directory.mkdir(parents=True, exist_ok=True)
    c0, c1 = clip[0]["candidates"], clip[1]["candidates"]
    start = clip[0]["frame_index"]
    with torch.inference_mode():
        first = core(c0, task_type="mot", frame_index=start)
        assert first["state"].num_tracks > 0, "Registered positive MOT probe produced no birth"
        repeated = core(c0, first["state"].clone(), task_type="mot", frame_index=start + 1)
        assert repeated["matched_pairs"], "No persistent match on repeated actual candidates"
        current = dict(zip(repeated["candidate_indices"].tolist(), repeated["track_ids"].tolist()))
        assert all(current[col] == int(first["state"].track_ids[row]) for row, col in repeated["matched_pairs"])
        empty = {name: value[:0] for name, value in c0.items() if isinstance(value, torch.Tensor)}
        hidden = core(empty, repeated["state"].clone(), task_type="mot", frame_index=start + 2)
        assert torch.equal(hidden["state"].track_ids, repeated["state"].track_ids)
        assert (hidden["state"].lost_age == 1).all() and not hidden["track_ids"].numel()
        recovered = core(c0, hidden["state"].clone(), task_type="mot", frame_index=start + 3)
        assert set(repeated["track_ids"].tolist()).intersection(recovered["track_ids"].tolist()), "One-frame occlusion recycled every identity"
        old_ids = set(hidden["state"].track_ids.tolist())
        assert all(identity in old_ids or identity >= hidden["state"].next_track_id for identity in recovered["track_ids"].tolist())

        preserved = first["state"].clone()
        altered = {name: value.clone() if isinstance(value, torch.Tensor) else value for name, value in c1.items()}
        altered["boxes"] = altered["boxes"] + 1000.
        altered["id_visual"] = -altered["id_visual"] * 3.
        core(c1, first["state"], task_type="mot", frame_index=start + 1)
        core(altered, first["state"], task_type="mot", frame_index=start + 1)
        assert state_equal(first["state"], preserved), "Future computation mutated prefix state"
        generator = torch.Generator(device=class_text.device).manual_seed(20261012)
        unrelated = torch.randn(class_text.shape, generator=generator, device=class_text.device)
        mot_a = core(c1, first["state"].clone(), task_type="mot", frame_index=start + 1, class_text=class_text)
        mot_b = core(c1, first["state"].clone(), task_type="mot", frame_index=start + 1, class_text=unrelated)
        outputs = ("identity_pair_logits", "pair_logits", "new_logits", "no_match_logits", "track_ids", "candidate_indices")
        assert all(torch.equal(mot_a[name], mot_b[name]) for name in outputs)
        assert state_equal(mot_a["state"], mot_b["state"])
        ov_a = core(c1, first["state"].clone(), task_type="ovmot", frame_index=start + 1, class_text=class_text)
        ov_b = core(c1, first["state"].clone(), task_type="ovmot", frame_index=start + 1, class_text=unrelated)
        raw_fields = ("identity_pair_logits", "new_logits", "no_match_logits", "candidate_id_tokens", "track_tokens")
        assert all(torch.equal(ov_a[name], ov_b[name]) for name in raw_fields)
        # Compare the pre-semantic identity update for identical fixed pairs,
        # without forcing or changing either online assignment decision.
        count = min(first["state"].num_tracks, c1["boxes"].shape[0])
        assert count > 0
        def raw_update(result):
            inputs = torch.cat([result["candidate_id_tokens"][:count], core.frame_context(result["track_tokens"][:count])], dim=-1)
            return core.identity_gru(inputs, first["state"].id_memory[:count])
        assert torch.equal(raw_update(ov_a), raw_update(ov_b))
        serialized = []
        for task in ("mot", "ovmot"):
            kwargs = {"task_type": task, "class_text": class_text if task == "ovmot" else None}
            continuous = U2OnlineTracker(core, device=class_text.device)
            uninterrupted = [continuous.step(item["candidates"], frame_index=item["frame_index"], **kwargs) for item in clip]
            split = U2OnlineTracker(core, device=class_text.device)
            for item in clip[:2]:
                split.step(item["candidates"], frame_index=item["frame_index"], **kwargs)
            state_path = directory / (task + "_online_state.pt")
            split.save_state(state_path)
            restored = U2OnlineTracker(core, device=class_text.device)
            restored.load_state(state_path, map_location=class_text.device)
            resumed = [restored.step(item["candidates"], frame_index=item["frame_index"], **kwargs) for item in clip[2:]]
            for a, b in zip(uninterrupted[2:], resumed):
                assert state_equal(a["state"], b["state"])
                assert all(torch.equal(a[name], b[name]) for name in outputs)
                assert not a["teacher_forced"] and not b["teacher_forced"]
            assert continuous.core is split.core is restored.core is core
            serialized.append(task)
    return {"birth_persistence": True, "observed_births": first["state"].num_tracks,
            "observed_repeat_matches": len(repeated["matched_pairs"]), "occlusion_retention": True,
            "no_future_dependency": True, "mot_specification_invariance": True,
            "ovmot_raw_identity_and_fixed_pair_gru_isolation": True,
            "semantic_scores_may_change": not torch.equal(ov_a["pair_logits"], ov_b["pair_logits"]),
            "online_state_save_reload": "BIT_EXACT", "serialized_tasks": serialized,
            "both_tasks_same_core_object": True, "selected_weights_modified": False, "teacher_forcing_used": False,
            "scope": "four original candidate frames plus repeated-observation/one-frame-occlusion causal probes"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    import torch
    from locatemot.unified.runtime.u2_checkpoint import load_u2_checkpoint
    from locatemot.unified.data.u2_mot_clips import CachedMOTClips, source_frame_index
    from tools.unified.build_u2_candidate_cache import digest, write_json
    torch.set_num_threads(2)
    packed = load_u2_checkpoint(args.checkpoint, device=args.device)
    if packed.reference_only:
        raise ValueError("Selected-core audit cannot accept a reference-only/random checkpoint")
    if digest(args.cache_manifest) != packed.metadata["ordinary_manifest"]["sha256"]:
        raise ValueError("Causal candidates differ from the selected ordinary feature inventory")
    cache = json.loads(args.cache_manifest.read_text())
    sampler = CachedMOTClips(cache, {})  # Candidate-only path: no GT label file is read.
    group = min((item["group"] for item in cache["groups"] if item["group"]["dataset"] == "mot17" and item["group"]["role"] == "dev"), key=lambda g: g["key"])
    clip = [{"candidates": sampler.load_candidates(group, position, args.device),
             "frame_index": source_frame_index(group, position), "frame": group["frames"][position]} for position in range(4)]
    report = {"format": "locatemot-u2-selected-core-causal-contract-v1", "reference_only": False,
              "checkpoint_sha256": digest(args.checkpoint), "candidate_cache_sha256": digest(args.cache_manifest),
              "foundation_sha256": packed.metadata["foundation_sha256"], "video": group["video"],
              "frames": [item["frame"] for item in clip], "official_test_labels_read": False}
    try:
        report.update(core_contract(packed.core, clip, packed.class_text, args.out.parent / "online_states"), status="PASS")
    except AssertionError as error:
        report.update(status="FAIL", blocked_status="LOCATEMOT_U_BLOCKED_U2_CAUSAL_CONTRACT", error=str(error))
        write_json(args.out, report)
        raise
    write_json(args.out, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
