"""End-to-end TrackJEV step test with a fake engine (no GPU, no weights, no data).

Validates the runtime chain: retrieval shortlist -> competition context ->
OneJev decisions -> one-to-one assignment -> assignment-conditioned memory ->
tracker operations, including an explicit MEMORY HOLD case.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "OneJev"))

from locatemot.groundjev.trackjev import decision_schema as ds
from locatemot.groundjev.trackjev.model import TrackJEVConfig, TrackJEVModel


class FakeAnswer:
    def __init__(self, probabilities=None, noul=None, score=None):
        self.probabilities = probabilities or {}
        self.noul = noul
        self.score = score

    def model_dump(self):
        return {"probabilities": self.probabilities, "noul": self.noul, "score": self.score}


class FakeEngine:
    """Candidate 0/1 compete for id 7; memory answer is configurable (HOLD by default)."""

    def __init__(self, memory_noul=0.2, id_prob=0.75):
        self.memory_noul = memory_noul
        self.id_prob = id_prob

    def decide(self, request, media=None):
        answers = {}
        identity_questions = [n for n in request.questions if n.startswith("identity_")]
        memory_questions = [n for n in request.questions if n.startswith("memory_")]
        for name in identity_questions:
            criteria = list(request.questions[name].criteria)
            existing = [c for c in criteria if c not in (ds.NEW_TOKEN, ds.REJECT_TOKEN)]
            probs = {}
            for token in criteria:
                if token == ds.NEW_TOKEN:
                    probs[token] = 0.1
                elif token == ds.REJECT_TOKEN:
                    probs[token] = 0.1
                elif existing and token == existing[0]:
                    probs[token] = self.id_prob
                else:
                    probs[token] = 0.02
            total = sum(probs.values())
            answers[name] = FakeAnswer(probabilities={k: v / total for k, v in probs.items()})
        for name in request.questions:
            if name.startswith("admit_"):
                answers[name] = FakeAnswer(noul=0.9)
            elif name.startswith("life_"):
                answers[name] = FakeAnswer(probabilities={"ACTIVE": 0.9, "LOST": 0.05,
                                                          "REACTIVATE": 0.03, "TERMINATE": 0.02})
        for name in memory_questions:
            answers[name] = FakeAnswer(noul=self.memory_noul)
        return SimpleNamespace(answers=answers), {"engine": "fake"}

    def decide_memory(self, request=None, **kwargs):  # not used: model calls decide()
        raise AssertionError("model must use decide() for memory questions")


class FakeAdapter:
    """Thin adapter shim: uses the fake engine for both decision and memory calls."""

    def __init__(self, engine):
        self.engine = engine
        self.metadata_calls = 0

    def decide(self, *, image, group, tracks, timestamp, specification,
               task_query=None, competition=None):
        from locatemot.groundjev.trackjev.native_onejev_adapter import NativeOneJevAdapter

        native = NativeOneJevAdapter("fake", engine=self.engine, shortlist=8)
        return native.decide(image=image, group=group,
                             tracks=[{"track_id": t["track_id"], "lost_age": t.get("lost_age", 0),
                                      "box": t.get("box"), "predicted_box": t.get("predicted_box")}
                                     for t in tracks],
                             timestamp=timestamp, specification=specification,
                             task_query=task_query, competition=competition)

    def decide_memory(self, *, image, group, assigned, tracks, timestamp, specification,
                      competition=None):
        from locatemot.groundjev.trackjev.native_onejev_adapter import NativeOneJevAdapter

        native = NativeOneJevAdapter("fake", engine=self.engine, shortlist=8)
        return native.decide_memory(image=image, group=group, assigned=assigned,
                                    tracks=[{"track_id": t["track_id"], "lost_age": t.get("lost_age", 0),
                                             "box": t.get("box"), "predicted_box": t.get("predicted_box")}
                                            for t in tracks],
                                    timestamp=timestamp, specification=specification,
                                    competition=competition)


def main() -> int:
    candidates = [
        {"box": [100.0, 100.0, 50.0, 100.0]},
        {"box": [104.0, 102.0, 50.0, 100.0]},   # duplicate of candidate 0
        {"box": [500.0, 200.0, 40.0, 90.0]},    # a different object
    ]
    tracks = [
        {"track_id": 7, "lost_age": 0, "box": [100.0, 100.0, 50.0, 100.0],
         "predicted_box": [100.0, 100.0, 50.0, 100.0]},
        {"track_id": 9, "lost_age": 12, "box": [0.0, 0.0, 10.0, 10.0],
         "predicted_box": [0.0, 0.0, 10.0, 10.0]},
    ]

    model = TrackJEVModel(FakeAdapter(FakeEngine(memory_noul=0.2)),
                          TrackJEVConfig(use_competition=True,
                                         assignment_conditioned_memory=True,
                                         duplicate_penalty=0.0))
    step = model.step(image=None, candidates=candidates, tracks=tracks, timestamp=3.0,
                      specification={"task": "mot", "vocabulary": ["person"], "expression": None})

    assert step.diagnostics["shortlist_size"] == 2, step.diagnostics
    assert step.diagnostics["competition_enabled"] is True
    assert step.diagnostics["competition_groups"] is not None
    # exactly one candidate owns id 7 after global assignment
    owners = [c for c, token in step.assignment.items() if token == "7"]
    assert len(owners) == 1, step.assignment
    assert step.diagnostics["assignment_diagnostics"]["one_identity_per_frame"] is True
    # memory noul=0.2 < 0.5 -> HOLD, and the tracker op must say so explicitly
    assert step.diagnostics["memory_holds"] >= 1, step.diagnostics
    holds = [op for op in step.tracker_ops if op["op"] == "ASSIGN_HOLD_MEMORY"]
    assert holds, step.tracker_ops
    assert step.diagnostics["config"]["arm_id"] == "B4", step.diagnostics["config"]

    # ablation: no assignment-conditioned memory -> unconditional write (B3)
    model_b3 = TrackJEVModel(FakeAdapter(FakeEngine(memory_noul=0.2)),
                             TrackJEVConfig(use_competition=True,
                                            assignment_conditioned_memory=False))
    step_b3 = model_b3.step(image=None, candidates=candidates, tracks=tracks, timestamp=3.0,
                            specification={"task": "mot"})
    assert step_b3.diagnostics["config"]["arm_id"] == "B3"
    assert step_b3.diagnostics["memory_holds"] == 0, step_b3.diagnostics
    assert not [op for op in step_b3.tracker_ops if op["op"] == "ASSIGN_HOLD_MEMORY"]

    print(json.dumps({
        "status": "TRACKJEV_MODEL_SELFTEST_PASS",
        "arm": step.diagnostics["config"]["arm_id"],
        "assignment": {str(k): v for k, v in step.assignment.items()},
        "memory_updates": {str(k): v for k, v in step.memory_updates.items()},
        "tracker_ops": step.tracker_ops,
        "competition_groups": step.diagnostics["competition_groups"],
        "b3_memory_holds": step_b3.diagnostics["memory_holds"],
    }, indent=1, ensure_ascii=False)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
