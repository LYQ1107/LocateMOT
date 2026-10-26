"""Native pretrained-OneJev adapter for TrackJEV.

This is the only module that talks to the real OneJev-0.8B engine
(``external/OneJev/qev``).  It builds the TrackJEV request (admission, identity,
memory and risk questions) and parses the answers back into decisions.

Design rules enforced here:

* identity options come from the caller's shortlist, capped at
  ``MAX_SHORTLIST``; truncation is recorded so retrieval coverage stays visible;
* NEW and REJECT are always present, so "no supported identity" never collapses
  into a false NEW;
* the memory question is only asked for candidates that received an identity,
  and it is asked *after* assignment (assignment-conditioned memory);
* the adapter never reads GT and never uses future frames.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from .decision_schema import (
    ADMIT_TEXT,
    MAX_SHORTLIST,
    MEMORY_TEXT,
    RISK_TEXT,
    IdentityOption,
    ParsedDecision,
    build_identity_options,
    parse_identity_answer,
)
from .candidate_competition import competition_rows, entity_clusters, sparse_edges, runtime_competition_boxes

ROOT = Path(__file__).resolve().parents[3]
ONEJEV_ROOT = ROOT / "external" / "OneJev"


def current_candidate_sheet(image: Any, group: Sequence[dict[str, Any]]):
    """Render the current crops with the layout used by the J1 TRAIN builder.

    Runtime candidates are xyxy. Only the source pixels and local candidate
    numbers enter this image; no annotation or predicted identity is drawn.
    """
    from PIL import Image, ImageDraw

    sheet = Image.new('RGB', (640, max(1, math.ceil(len(group) / 4)) * 192), 'white')
    draw = ImageDraw.Draw(sheet)
    for index, candidate in enumerate(group):
        x, y = (index % 4) * 160, (index // 4) * 192
        draw.text((x + 2, y + 2), f'Candidate {index}', fill='black')
        crop = image.crop(tuple(int(value) for value in candidate['box'])).convert('RGB')
        crop.thumbnail((154, 166))
        sheet.paste(crop, (x + 3, y + 22))
    return sheet


def historical_track_sheet(tracks: Sequence[dict[str, Any]]):
    """Render prior anchor/long/recent images; track IDs are model predictions."""
    from PIL import Image, ImageDraw

    sheet=Image.new('RGB',(4*288,max(1,math.ceil(len(tracks)/4))*144),'white')
    draw=ImageDraw.Draw(sheet)
    for index,track in enumerate(tracks):
        x,y=(index%4)*288,(index//4)*144
        draw.text((x+2,y+2),f"ID {track['track_id']} anchor / long / recent",fill='black')
        for column,source in enumerate(track['causal_images']):
            crop=source.copy();crop.thumbnail((92,112))
            sheet.paste(crop,(x+column*96,y+22))
    return sheet


@dataclass
class AdapterOutput:
    decisions: list[ParsedDecision]
    metadata: dict[str, Any] = field(default_factory=dict)
    request_count: int = 0
    question_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"decisions": [d.to_dict() for d in self.decisions],
                "metadata": self.metadata,
                "request_count": self.request_count,
                "question_count": self.question_count}


class NativeOneJevAdapter:
    """Wraps ``qev.mm_engine.MMDecisionEngine`` behind the TrackJEV contract."""

    def __init__(self, checkpoint: str | Path, *, device: str = "cuda:0",
                 dtype: str = "bfloat16", batch_candidates: int = 8,
                 shortlist: int = MAX_SHORTLIST, evidence_format: str = "trackjev_v1",
                 risk_horizon: int = 30, ask_risk: bool = False,
                 engine: Optional[Any] = None, competition_geometry: str = 'xyxy',
                 candidate_evidence: str = 'full_frame',
                 memory_transport: str = 'legacy',
                 empty_history_question: str = 'identity',
                 history_evidence: str = 'none') -> None:
        self.checkpoint = str(Path(checkpoint).resolve())
        self.batch_candidates = int(batch_candidates)
        self.shortlist = min(int(shortlist), MAX_SHORTLIST)
        self.evidence_format = evidence_format
        self.risk_horizon = int(risk_horizon)
        self.ask_risk = bool(ask_risk)
        self.device = device
        self.competition_geometry = competition_geometry
        self.candidate_evidence = candidate_evidence
        self.memory_transport = memory_transport
        self.empty_history_question = empty_history_question
        self.history_evidence = history_evidence
        if engine is not None:
            self.engine = engine
        else:
            import sys

            sys.path.insert(0, str(ONEJEV_ROOT))
            from qev.mm_engine import MMDecisionEngine

            self.engine = MMDecisionEngine(self.checkpoint, device=device, dtype=dtype,
                                           cuda_graphs=False)

    # ------------------------------------------------------------------ state
    def build_state(self, *, image: Any, group: Sequence[dict[str, Any]],
                    competition: Optional[dict[str, Any]], tracks: Sequence[dict[str, Any]],
                    timestamp: float, specification: dict[str, Any],
                    task_query: Optional[str] = None) -> dict[str, Any]:
        state: dict[str, Any] = {
            "current_frame": "<image:1>",
            "timestamp": timestamp,
            "task": specification,
            "candidates": [{"index": i, "box": list(c["box"]),
                            **({"label": c["label"]} if c.get("label") else {})}
                           for i, c in enumerate(group)],
            "historical_tracks": [
                {"id": t["track_id"], "box": list(t.get("box") or []),
                 "predicted_box": list(t.get("predicted_box") or []),
                 "age": t.get("age"), "lost_age": t.get("lost_age"),
                 "state": t.get("state"), "last_reliable_timestamp": t.get("last_reliable_timestamp")}
                for t in tracks],
            "identity_option_limit": self.shortlist,
            "decision_vocabulary": {
                "existing": "an id in the option list (MATCH or REACTIVATE if the id is lost)",
                "NEW": "a genuinely new physical entity",
                "REJECT": "invalid or duplicate proposal that must not create an id",
            },
        }
        if task_query:
            state["referring_expression"] = task_query
        if competition:
            state.update(competition)
        return state

    def _image_media(self, state, image, group, tracks=()):
        media = [{"type": "image", "image": image}]
        if self.candidate_evidence == 'train_current_crops' and group:
            state['current_candidate_crop_sheet'] = '<image:2>'
            media.append({'type': 'image', 'image': current_candidate_sheet(image, group)})
        if self.history_evidence=='causal_crops' and tracks:
            state['historical_anchor_and_recent_crops']=f'<image:{len(media)+1}>'
            media.append({'type':'image','image':historical_track_sheet(tracks)})
        return media

    # ---------------------------------------------------------------- request
    def build_request(self, *, image: Any, group: Sequence[dict[str, Any]],
                      tracks: Sequence[dict[str, Any]], timestamp: float,
                      specification: dict[str, Any], task_query: Optional[str] = None,
                      competition: Optional[dict[str, Any]] = None) -> tuple[Any, list[dict[str, Any]]]:
        from qev.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion, SystemOneRequest

        shortlist_entries = [{"track_id": t["track_id"], "lost_age": t.get("lost_age"),
                              "prior": t.get("prior")} for t in tracks[:self.shortlist]]
        options = build_identity_options(shortlist_entries)
        state = self.build_state(image=image, group=group, competition=competition,
                                 tracks=tracks[:self.shortlist], timestamp=timestamp,
                                 specification=specification, task_query=task_query)
        media = self._image_media(state, image, group, tracks[:self.shortlist])

        questions: dict[str, Any] = {}
        for index in range(len(group)):
            questions[f"admit_{index}"] = NoulQuestion(
                type="noul", instructions=ADMIT_TEXT.format(index=index))
            criteria = {}
            for option in options:
                if option.kind == "existing":
                    criteria[option.token] = option.text
                else:
                    criteria[option.token] = option.text
            instructions = (f"Which identity owns current candidate {index}? Judge only from the "
                            f"visual evidence, the competition rows and the identity history. "
                            f"A class label alone is never identity evidence.")
            if not tracks and self.empty_history_question == 'train_birth_v1':
                # Exact question from the frozen step1200 TRAIN birth rows.
                # Its missing candidate index is an audited corpus defect;
                # this isolated control does not invent a new training prompt.
                instructions = ("No persistent identity exists yet in this sequence. "
                                "Which option owns current candidate?")
            questions[f"identity_{index}"] = ChoiceQuestion(
                type="choice", instructions=instructions, criteria=criteria)
        for track in tracks[:self.shortlist]:
            questions[f"life_{track['track_id']}"] = ChoiceQuestion(
                type="choice",
                instructions=(f"What is the current lifecycle of identity {track['track_id']} "
                              f"given only current and past evidence?"),
                criteria={"ACTIVE": "currently visible and associated",
                          "LOST": "temporarily unobserved, memory retained",
                          "REACTIVATE": "returned after a gap",
                          "TERMINATE": "permanently leaves this tracked video"})
        request = SystemOneRequest(state=state, model="jev-latest", questions=questions)
        self._pending_options = options
        return request, media

    def build_memory_questions(self, assigned: dict[int, str]) -> dict[str, Any]:
        """Assignment-conditioned memory questions (asked after identity allocation)."""
        from qev.schema import NoulQuestion, ScoreQuestion

        questions: dict[str, Any] = {}
        for candidate, token in assigned.items():
            if token in ("NEW", "REJECT", "NO_MATCH"):
                continue
            questions[f"memory_{candidate}"] = NoulQuestion(
                type="noul",
                instructions=MEMORY_TEXT.format(index=candidate, track_id=token))
            if self.ask_risk:
                questions[f"risk_{candidate}"] = ScoreQuestion(
                    type="score",
                    instructions=RISK_TEXT.format(index=candidate, track_id=token,
                                                  horizon=self.risk_horizon),
                    criteria=["negligible", "low", "moderate", "high"])
        return questions

    # --------------------------------------------------------------- decode
    def decide(self, *, image: Any, group: Sequence[dict[str, Any]],
               tracks: Sequence[dict[str, Any]], timestamp: float,
               specification: dict[str, Any], task_query: Optional[str] = None,
               competition: Optional[dict[str, Any]] = None) -> AdapterOutput:
        """One inference over one candidate group (may split on context limits)."""
        if not group:
            return AdapterOutput(decisions=[], metadata={"backend": "trackjev_native_onejev",
                                                         "empty_group": True})

        collected: list[dict[str, float]] = [{} for _ in group]
        admission: list[Optional[float]] = [None] * len(group)
        lifecycle: dict[str, Any] = {}
        batches: list[dict[str, Any]] = []
        options = None

        pending = [list(range(len(group)))]
        split_events: list[dict[str, Any]] = []
        while pending:
            indices = pending.pop(0)
            subgroup = [group[i] for i in indices]
            # The full-frame competition matrix is quadratic in the number of
            # detector proposals. Rebuild it for a split subgroup; retaining
            # the original NxN matrix here would make even a one-candidate
            # retry exceed OneJev's context limit on dense frames.
            local_competition = competition
            if competition is not None and len(indices) != len(group):
                boxes = runtime_competition_boxes([c["box"] for c in subgroup],self.competition_geometry)
                features = [c.get("native_region_feature") for c in subgroup]
                edges = sparse_edges(boxes, features if any(f is not None for f in features) else None)
                clusters = entity_clusters(boxes, edges)
                local_competition = competition_rows(boxes, clusters, edges)
            request, media = self.build_request(
                image=image, group=subgroup, tracks=tracks, timestamp=timestamp,
                specification=specification, task_query=task_query, competition=local_competition)
            options = getattr(self, "_pending_options", options)
            try:
                response, meta = self.engine.decide(request, media=media)
            except ValueError as error:
                if not str(error).startswith("state plus longest question is ") or len(subgroup) < 2:
                    raise
                middle = len(subgroup) // 2
                split_events.append({"group_size": len(subgroup), "error": str(error),
                                     "new_group_sizes": [middle, len(subgroup) - middle]})
                pending[:0] = [indices[:middle], indices[middle:]]
                continue
            batches.append({"size": len(subgroup), "metadata": meta,
                            "questions": len(request.questions)})
            for local_index, global_index in enumerate(indices):
                answer = response.answers.get(f"identity_{local_index}")
                probabilities = dict(getattr(answer, "probabilities", {}) or {})
                collected[global_index] = probabilities
                admit = response.answers.get(f"admit_{local_index}")
                if admit is not None:
                    admission[global_index] = float(getattr(admit, "noul", 0.0))
            for track in tracks[:self.shortlist]:
                key = f"life_{track['track_id']}"
                if key in response.answers and key not in lifecycle:
                    lifecycle[key] = response.answers[key].model_dump()

        options = options or []
        decisions = [parse_identity_answer(i, probs, options) for i, probs in enumerate(collected)]
        for index, decision in enumerate(decisions):
            decision.admitted = admission[index]
        shortlist_ids = [int(t["track_id"]) for t in tracks[:self.shortlist]]
        truncated = max(0, len(tracks) - self.shortlist)
        metadata = {
            "backend": "trackjev_native_onejev",
            "checkpoint": self.checkpoint,
            "evidence_format": self.evidence_format,
            "competition_geometry": self.competition_geometry,
            "candidate_evidence": self.candidate_evidence,
            "memory_transport": self.memory_transport,
            "empty_history_question": self.empty_history_question,
            "history_evidence": self.history_evidence,
            "native_probabilities": True,
            "gt_read": False,
            "shortlist_limit": self.shortlist,
            "shortlist_ids": shortlist_ids,
            "shortlist_truncated_ids": truncated,
            "option_count": len(options),
            "batches": batches,
            "context_splits": split_events,
            "lifecycle": lifecycle,
            "memory_questions_deferred_until_after_assignment": True,
        }
        return AdapterOutput(decisions=decisions, metadata=metadata,
                             request_count=len(batches),
                             question_count=sum(b["questions"] for b in batches))

    # ------------------------------------------------------- memory decisions
    def decide_memory(self, *, image: Any, group: Sequence[dict[str, Any]],
                      assigned: dict[int, str], tracks: Sequence[dict[str, Any]],
                      timestamp: float, specification: dict[str, Any],
                      competition: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Ask UPDATE/HOLD for the identities that were actually assigned."""
        questions = self.build_memory_questions(assigned)
        if not questions:
            return {"asked": 0, "updates": {}, "policy": "no identity assigned in this frame"}
        from qev.schema import SystemOneRequest

        # Memory is queried only for candidates that received an existing
        # identity. Do not carry the full detector proposal bank (and its
        # quadratic competition rows) into this second request: dense frames
        # otherwise exceed the OneJev context limit even when one identity is
        # being updated. The versioned local_xyxy contract aligns state,
        # questions and assignment keys to the compact group's local indices.
        # The legacy transport remains available for historical controls.
        selected = sorted(int(i) for i in assigned)
        local_group = [group[i] for i in selected]
        local_assigned = {local: assigned[original]
                          for local, original in enumerate(selected)}
        local_questions = self.build_memory_questions(local_assigned)
        local_competition = None
        if competition is not None:
            boxes = ([c["box"] for c in local_group] if self.memory_transport == 'legacy' else
                     runtime_competition_boxes([c["box"] for c in local_group], self.competition_geometry))
            features = [c.get("native_region_feature") for c in local_group]
            edges = sparse_edges(boxes, features if any(f is not None for f in features) else None)
            clusters = entity_clusters(boxes, edges)
            local_competition = competition_rows(boxes, clusters, edges)
        state = self.build_state(image=image, group=local_group, competition=local_competition,
                                 tracks=tracks[:self.shortlist], timestamp=timestamp,
                                 specification=specification)
        if self.memory_transport == 'legacy':
            state["assigned_identity_per_candidate"] = {
                str(original): assigned[original] for original in selected
            }
        else:
            state["assigned_identity_per_candidate"] = {
                str(local): identity for local, identity in local_assigned.items()
            }
            state["original_candidate_indices"] = selected
        state["assignment_condition"] = ("these ids are the actual final allocation of this frame; "
                                         "decide whether each one deserves a long-term memory update")
        media = self._image_media(state, image, local_group, tracks[:self.shortlist])
        request = SystemOneRequest(state=state, model="jev-latest", questions=local_questions)
        response, meta = self.engine.decide(request, media=media)
        updates: dict[int, dict[str, Any]] = {}
        for local, candidate in enumerate(selected):
            key = f"memory_{local}"
            if key in response.answers:
                updates[candidate] = {"noul": float(response.answers[key].noul)}
            risk_key = f"risk_{local}"
            if risk_key in response.answers:
                updates.setdefault(candidate, {})["risk"] = response.answers[risk_key].score
        return {"asked": len(questions), "updates": updates, "metadata": meta,
                "candidate_evidence": self.candidate_evidence, "memory_transport": self.memory_transport,
                "selected_original_candidate_indices": selected,
                "policy": "memory write conditioned on the final assigned identity"}
