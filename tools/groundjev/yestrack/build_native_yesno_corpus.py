#!/usr/bin/env python3
"""Build a causal native OneJev Yes/No corpus from legal TRAIN exports.

The labels are attached after reading the immutable TRAIN references.  They are
never copied into ``state`` or the question text.  MOT identity rows are
counterfactual proposals: the proposed identity is an input to the verifier,
and the TRAIN reference decides whether that proposal is correct.  This keeps
the YesTrack task binary without changing the underlying GroundAnything
candidate or history state.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_GROUPS = Path(
    "/data2/user/LocateMOT_GroundJEV/outputs/groundjev/training_data/"
    "broader_native_MOT1056_full_v3_groups_v1/clips"
)
DEFAULT_RMOT = Path(
    "/data2/user/LocateMOT_GroundJEV/outputs/groundjev/training_data/"
    "onejev_native_rmot_membership_v2_registered_release_root/train.jsonl"
)
DEFAULT_OVMOT = Path(
    "/data2/user/LocateMOT_GroundJEV/outputs/groundjev/training_data/"
    "onejev_base_categories_v1/train.jsonl"
)

# Keep validation video-disjoint.  These are development videos from the legal
# TRAIN exports, not MOT/RMOT/TAO official TEST videos.
DEFAULT_TRAIN_VIDEOS = {
    "MOT17-09-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN", "MOT17-02-FRCNN",
    "dancetrack0001", "dancetrack0024", "dancetrack0057", "dancetrack0074",
    "dancetrack0080", "dancetrack0098", "refer_kitti_0000", "refer_kitti_0002",
    "refer_kitti_0003", "refer_kitti_0006", "refer_kitti_0008", "tao_train_0",
    "tao_train_1", "tao_train_7", "tao_train_9", "tao_train_13",
}
DEFAULT_VAL_VIDEOS = {
    "MOT17-04-FRCNN", "MOT17-05-FRCNN", "dancetrack0006", "dancetrack0032",
    "dancetrack0068", "dancetrack0086", "dancetrack0096", "refer_kitti_0010",
    "refer_kitti_0012", "refer_kitti_0014", "refer_kitti_0017", "tao_train_16",
    "tao_train_38", "tao_train_47",
}


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"non-object row in {path}")
                yield value


def _target_key(ref: dict[str, Any]) -> str:
    target = ref.get("target") or {}
    if not isinstance(target, dict) or len(target) != 1:
        raise ValueError(f"non-single target in {ref.get('id')}: {target!r}")
    return str(next(iter(target)))


def _media(group: dict[str, Any]) -> list[dict[str, Any]]:
    media = copy.deepcopy(group.get("media") or [])
    if not media:
        raise ValueError(f"group {group.get('id')} has no media")
    return media


def _binary_question(instructions: str) -> dict[str, Any]:
    return {"type": "noul", "instructions": instructions}


def _row(*, identifier: str, split: str, source: str, task: str,
         state: dict[str, Any], media: list[dict[str, Any]], question: dict[str, Any],
         yes: bool, proposal_source: str, source_ref: str, family: str,
         extra: dict[str, Any] | None = None) -> dict[str, Any]:
    supervision = {
        "yestrack_version": "native-yesno-v2-stable-sha-seed",
        "proposal_source": proposal_source,
        "source_reference": source_ref,
        "family": family,
        "gt_used_for_prompt": False,
        "future_evidence": False,
        "official_test_labels_read": False,
        "target_attached_outside_model_input": True,
    }
    if extra:
        supervision.update(extra)
    return {
        "id": identifier,
        "split": split,
        "source": source,
        "task": task,
        "state": state,
        "question": question,
        "target": {"yes" if yes else "no": 1.0},
        "media": media,
        "supervision": supervision,
    }


def _clip_identity_rows(clip: Path, split: str, max_rows: int,
                        seed: int) -> list[dict[str, Any]]:
    groups_path = clip / "groups.jsonl.gz"
    refs_path = clip / "train_refs.jsonl.gz"
    if not groups_path.is_file() or not refs_path.is_file():
        return []
    groups: dict[str, dict[str, Any]] = {}
    with gzip.open(groups_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                group = json.loads(line)
                groups[str(group["id"])] = group
    identity_rows: list[dict[str, Any]] = []
    for ref in _read_gzip_jsonl(refs_path):
        if ref.get("task") != "identity":
            continue
        link = ref.get("groundjev_group_question_ref") or {}
        group = groups.get(str(link.get("group")))
        if group is None:
            raise KeyError(f"missing group {link.get('group')} for {ref.get('id')}")
        qid = str(link.get("question"))
        if not qid.startswith("match_"):
            continue
        try:
            candidate_index = int(qid.split("_", 1)[1])
        except Exception as exc:
            raise ValueError(f"bad match question {qid}") from exc
        state_base = group["state"]
        histories = state_base.get("historical_tracks") or []
        history_ids = [str(h.get("id")) for h in histories if h.get("id") is not None]
        gold = _target_key(ref)
        # A Yes/No identity proposal needs a listed historical identity.  NEW,
        # IGNORE and NO_MATCH rows become negative proposals when history exists.
        if not history_ids:
            continue
        if gold not in {"NEW", "IGNORE", "NO_MATCH"} and gold in history_ids:
            proposals = [(gold, True)]
            wrong = next((x for x in history_ids if x != gold), None)
            if wrong is not None:
                proposals.append((wrong, False))
        else:
            proposals = [(history_ids[0], False)]
        for proposal, yes in proposals:
            state = copy.deepcopy(state_base)
            state["yestrack_proposal"] = {
                "kind": "identity",
                "candidate_index": candidate_index,
                "proposed_identity": proposal,
            }
            identity_rows.append(_row(
                identifier=f"yes_{ref['id']}_{proposal}_{'yes' if yes else 'no'}",
                split=split,
                source=str(ref.get("source", clip.name)),
                task="identity_pair",
                state=state,
                media=_media(group),
                question=_binary_question(
                    f"Does proposed persistent identity {proposal} own current candidate "
                    f"{candidate_index}? Compare the current crop, history, motion and "
                    "competing identities. Answer yes only when the proposed association "
                    "is supported."
                ),
                yes=yes,
                proposal_source="groundanything_native_identity_candidate",
                source_ref=str(ref.get("id")),
                family="identity_correct_or_incorrect",
                extra={"candidate_index": candidate_index, "proposed_identity": proposal,
                       "reference_target": gold},
            ))
    # Keep the two MOT verifier families represented even when a clip has many
    # identity candidates.  In particular, do not let identity rows consume the
    # quota before the occlusion/recovery labels are read.
    # Python's built-in hash is process-randomized.  v2 derives the clip seed
    # from a stable digest; existing v1 artifacts remain immutable.
    stable_clip_seed = int.from_bytes(hashlib.sha256(clip.name.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed + stable_clip_seed % 1000003)
    rng.shuffle(identity_rows)
    identity_rows = identity_rows[: max(1, (2 * max_rows) // 3)]

    # Lifecycle references are the explicitly supervised occlusion/recovery
    # rows from the same causal state and media export.
    for ref in _read_gzip_jsonl(refs_path):
        if ref.get("task") != "lifecycle":
            continue
        link = ref.get("groundjev_group_question_ref") or {}
        group = groups.get(str(link.get("group")))
        if group is None:
            raise KeyError(f"missing group {link.get('group')} for {ref.get('id')}")
        qid = str(link.get("question"))
        if not qid.startswith("life_"):
            continue
        proposal = qid[len("life_"):]
        target = _target_key(ref)
        state = copy.deepcopy(group["state"])
        state["yestrack_proposal"] = {"kind": "occlusion_recovery", "proposed_identity": proposal}
        identity_rows.append(_row(
            identifier=f"yes_{ref['id']}", split=split,
            source=str(ref.get("source", clip.name)), task="occlusion_recovery",
            state=state, media=_media(group),
            question=_binary_question(
                f"Should proposed persistent identity {proposal} be reactivated after "
                "an occlusion in the current frame? Use only the causal history and "
                "current visual evidence."
            ), yes=target == "REACTIVATE",
            proposal_source="groundanything_native_lifecycle_candidate",
            source_ref=str(ref.get("id")), family="occlusion_recovery",
            extra={"proposed_identity": proposal, "reference_target": target},
        ))
    # Sample lifecycle rows independently, then combine.  The source labels
    # remain unchanged; this is only a deterministic corpus-size cap.
    identity_rows_count = len([r for r in identity_rows if r["supervision"]["family"] == "identity_correct_or_incorrect"])
    lifecycle_rows = [r for r in identity_rows if r["supervision"]["family"] == "occlusion_recovery"]
    identity_rows = [r for r in identity_rows if r["supervision"]["family"] != "occlusion_recovery"]
    rng.shuffle(lifecycle_rows)
    lifecycle_cap = max(1, max_rows - min(identity_rows_count, max(1, (2 * max_rows) // 3)))
    return (identity_rows[: max(1, (2 * max_rows) // 3)] + lifecycle_rows[:lifecycle_cap])[:max_rows]


def _read_gzip_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"non-object row in {path}")
                yield value


def _rmot_rows(path: Path, split_sources: set[str], max_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in _read_jsonl(path):
        source_name = str(source.get("source", "unknown"))
        split = "val" if source_name in split_sources else "train"
        if source_name == "MOT20-05" or "mot20-05" in source_name.lower():
            continue
        state = copy.deepcopy(source.get("state") or {})
        if not state:
            # Existing native RMOT rows always have a state; skip malformed
            # records instead of inventing boxes or expressions.
            continue
        media = copy.deepcopy(source.get("media") or [])
        expression = str(((state.get("task") or {}).get("expression")) or "").strip()
        if not expression or not media:
            continue
        target = str(next(iter(source.get("target", {"no": 1}))))
        rows.append(_row(
            identifier=f"yes_rmot_{source['id']}", split=split, source=source_name,
            task="rmot", state=state, media=media,
            question=_binary_question(
                "Does candidate 0 satisfy the supplied referring expression? Use the "
                "current visual evidence and expression, including attributes and relations."
            ), yes=target.lower() == "yes", proposal_source="native_rmot_candidate_export",
            source_ref=str(source.get("id")), family="rmot_expression_membership",
            extra={"expression": expression, "reference_target": target},
        ))
        if len([r for r in rows if r["split"] == "train"]) >= max_rows:
            # Keep validation examples after the train quota has been reached.
            if all(r["split"] != "val" for r in rows):
                continue
    return rows


def _ovmot_rows(path: Path, split_sources: set[str], max_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rng = random.Random(20261019)
    for source in _read_jsonl(path):
        if source.get("task") != "category_leaf":
            continue
        source_name = str(source.get("source", "unknown"))
        split = "val" if source_name in split_sources else "train"
        state_base = source.get("state")
        media = source.get("media") or []
        q = source.get("question") or {}
        criteria = q.get("criteria") or {}
        target = str(next(iter(source.get("target", {"UNKNOWN": 1}))))
        if not state_base or not media or not criteria:
            continue
        names = [str(k) for k in criteria]
        correct = str(criteria.get(target, "")) if target != "UNKNOWN" else ""
        if not correct:
            # UNKNOWN is a valid negative category query but has no positive
            # category text, so use a deterministic wrong category only.
            continue
        wrong = next((str(v) for k, v in criteria.items() if str(k) != target), None)
        if wrong is None:
            continue
        for category, yes in ((correct, True), (wrong, False)):
            state = copy.deepcopy(state_base)
            state["yestrack_proposal"] = {"kind": "ovmot_membership", "category": category}
            rows.append(_row(
                identifier=f"yes_ovmot_{source['id']}_{'pos' if yes else 'neg'}",
                split=split, source=source_name, task="ovmot", state=state,
                media=copy.deepcopy(media), question=_binary_question(
                    f"Does the visible object belong to the proposed category '{category}'? "
                    "Use visual evidence only; detector query labels are not category evidence."
                ), yes=yes, proposal_source="tao_base_train_category_candidate",
                source_ref=str(source.get("id")), family="ovmot_base_membership",
                extra={"category": category, "reference_target": target,
                       "novel_category": False},
            ))
        if len([r for r in rows if r["split"] == "train"]) >= max_rows and all(r["split"] != "val" for r in rows):
            # Continue only long enough to collect requested validation rows.
            continue
    rng.shuffle(rows)
    return rows


def build(args: argparse.Namespace) -> dict[str, Any]:
    train_videos = set(args.train_videos.split(",")) if args.train_videos else DEFAULT_TRAIN_VIDEOS
    val_videos = set(args.val_videos.split(",")) if args.val_videos else DEFAULT_VAL_VIDEOS
    if train_videos & val_videos:
        raise ValueError("train/val video split overlaps")
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    clips_seen = []
    for clip in sorted(args.groups_root.iterdir()):
        if not clip.is_dir() or clip.name == "MOT20-05":
            continue
        if clip.name not in train_videos and clip.name not in val_videos:
            continue
        split = "val" if clip.name in val_videos else "train"
        clip_rows = _clip_identity_rows(clip, split, args.max_mot_rows_per_video, args.seed)
        rows.extend(clip_rows)
        clips_seen.append({"video": clip.name, "split": split, "rows": len(clip_rows)})
    rows.extend(_rmot_rows(args.rmot, {x for x in val_videos if x.startswith("refer_kitti_")}, args.max_rmot_train))
    rows.extend(_ovmot_rows(args.ovmot, {x for x in val_videos if x.startswith("tao_train_")}, args.max_ovmot_train))
    # A fixed shuffle gives reproducible batches while retaining video-disjoint
    # split membership.  Training is restricted to train rows by the config.
    random.Random(args.seed).shuffle(rows)
    ids = [str(r["id"]) for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate YesTrack row IDs")
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            # Preserve state insertion order: QEV multimodal rendering expects
            # <image:1>, <image:2>, ... in the same order as ``media``.  Sorting
            # JSON keys would silently reverse the state placeholders for the
            # native tracking states.
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = Counter((r["split"], r["task"], next(iter(r["target"]))) for r in rows)
    manifest = {
        "format": "locatemot-native-yesno-v2",
        "status": "LEGAL_CAUSAL_NATIVE_YESNO_CORPUS_V2_STABLE_SEED_READY",
        "rows": str(out.resolve()),
        "rows_sha256": _digest(out),
        "row_count": len(rows),
        "counts": {"|".join(k): v for k, v in sorted(counts.items())},
        "clips": clips_seen,
        "train_videos": sorted(train_videos),
        "val_videos": sorted(val_videos),
        "sources": {"groups_root": str(args.groups_root.resolve()), "groups_root_sha256_not_applicable": True,
                    "rmot": str(args.rmot.resolve()), "rmot_sha256": _digest(args.rmot),
                    "ovmot": str(args.ovmot.resolve()), "ovmot_sha256": _digest(args.ovmot)},
        "gt_used_for_prompt": False,
        "future_evidence": False,
        "official_test_labels_read": False,
        "MOT20-05_included": False,
        "model_training_started": False,
        "identity_proposal_protocol": "candidate and historical identity proposals are inputs; TRAIN target remains outside state/question",
        "occlusion_protocol": "life_<identity> REACTIVATE versus other lifecycle labels; causal state/media unchanged",
        "ovmot_protocol": "TAO Base TRAIN category positives with deterministic hard-negative category proposals; novel labels excluded",
    }
    manifest_path = out.with_suffix(out.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--groups-root", type=Path, default=DEFAULT_GROUPS)
    p.add_argument("--rmot", type=Path, default=DEFAULT_RMOT)
    p.add_argument("--ovmot", type=Path, default=DEFAULT_OVMOT)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-mot-rows-per-video", type=int, default=220)
    p.add_argument("--max-rmot-train", type=int, default=240)
    p.add_argument("--max-ovmot-train", type=int, default=160)
    p.add_argument("--train-videos", default=None)
    p.add_argument("--val-videos", default=None)
    p.add_argument("--seed", type=int, default=20261019)
    args = p.parse_args()
    print(json.dumps(build(args), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
