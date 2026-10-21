"""Adapter plumbing test for TrackJEV with a fake OneJev engine (no GPU, no weights).

It validates:
* the real ``qev.schema`` request objects can be constructed,
* identity options cap NEW/REJECT correctly,
* answer parsing maps probabilities back to MATCH / NEW / REJECT,
* assignment-conditioned memory questions are only asked for assigned ids,
* shortlist truncation is reported instead of being silent.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "OneJev"))

from locatemot.groundjev.trackjev import decision_schema as ds
from locatemot.groundjev.trackjev.native_onejev_adapter import NativeOneJevAdapter


class FakeAnswer:
    def __init__(self, probabilities=None, noul=None, score=None, criteria=None):
        self.probabilities = probabilities or {}
        self.noul = noul
        self.score = score
        self.criteria = criteria or []

    def model_dump(self):
        return {"probabilities": self.probabilities, "noul": self.noul, "score": self.score}


class FakeEngine:
    """Deterministic responder: boosts the first existing id, then NEW."""

    def __init__(self, existing_boost=0.62, new_boost=0.22):
        self.existing_boost = existing_boost
        self.new_boost = new_boost
        self.requests = []

    def decide(self, request, media=None):
        self.requests.append(request)
        answers = {}
        for name, question in request.questions.items():
            if name.startswith("identity_"):
                criteria = list(question.criteria)
                probs = {}
                existing = [c for c in criteria if c not in (ds.NEW_TOKEN, ds.REJECT_TOKEN)]
                for token in criteria:
                    if token == ds.NEW_TOKEN:
                        probs[token] = self.new_boost
                    elif token == ds.REJECT_TOKEN:
                        probs[token] = 0.06
                    else:
                        probs[token] = self.existing_boost if existing and token == existing[0] else 0.02
                total = sum(probs.values())
                probs = {k: v / total for k, v in probs.items()}
                answers[name] = FakeAnswer(probabilities=probs, criteria=criteria)
            elif name.startswith("admit_"):
                answers[name] = FakeAnswer(noul=0.83)
            elif name.startswith("memory_"):
                answers[name] = FakeAnswer(noul=0.71)
            elif name.startswith("life_"):
                answers[name] = FakeAnswer(probabilities={"ACTIVE": 0.8, "LOST": 0.1,
                                                          "REACTIVATE": 0.05, "TERMINATE": 0.05})
        return SimpleNamespace(answers=answers), {"engine": "fake", "requests": len(self.requests)}


def main() -> int:
    group = [{"box": [10.0, 10.0, 20.0, 40.0], "label": "person"},
             {"box": [12.0, 11.0, 20.0, 40.0], "label": "person"},
             {"box": [400.0, 50.0, 30.0, 60.0], "label": "person"}]
    tracks = [{"track_id": 7, "lost_age": 0, "box": [10.0, 10.0, 20.0, 40.0],
               "predicted_box": [10.0, 10.0, 20.0, 40.0], "age": 12, "state": "ACTIVE",
               "last_reliable_timestamp": 1.0},
              {"track_id": 9, "lost_age": 15, "box": [0.0, 0.0, 5.0, 5.0],
               "predicted_box": [0.0, 0.0, 5.0, 5.0], "age": 40, "state": "LOST",
               "last_reliable_timestamp": 0.2}]
    competition = {"competition_groups": [[0, 1], [2]],
                   "competition_group_of_candidate": [0, 0, 1],
                   "competition_iou_rows_milli": ["1000,930,0", "930,1000,0", "0,0,1000"]}

    engine = FakeEngine()
    adapter = NativeOneJevAdapter("fake-checkpoint", engine=engine, shortlist=2,
                                  evidence_format="trackjev_v1", ask_risk=True)
    out = adapter.decide(image=None, group=group, tracks=tracks, timestamp=2.0,
                         specification={"task": "mot", "vocabulary": ["person"], "expression": None},
                         competition=competition)

    assert out.request_count == 1, out.to_dict()
    assert len(out.decisions) == 3
    assert out.decisions[0].decision == "MATCH" and out.decisions[0].track_id == 7, out.decisions[0]
    assert out.decisions[0].admitted == 0.83
    assert out.metadata["shortlist_truncated_ids"] == 0
    assert out.metadata["option_count"] == 2 + 3  # two ids + NEW + NO_MATCH + REJECT

    # assignment-conditioned memory questions: only for ids that were assigned
    memory = adapter.decide_memory(image=None, group=group,
                                   assigned={0: "7", 1: ds.REJECT_TOKEN, 2: ds.NEW_TOKEN},
                                   tracks=tracks, timestamp=2.0,
                                   specification={"task": "mot"}, competition=competition)
    assert memory["asked"] >= 1, memory
    assert 0 in memory["updates"] and abs(memory["updates"][0]["noul"] - 0.71) < 1e-6, memory
    assert 1 not in memory["updates"] and 2 not in memory["updates"], memory

    # shortlist truncation must be reported, not silent
    truncated = NativeOneJevAdapter("fake-checkpoint", engine=FakeEngine(), shortlist=1)
    out2 = truncated.decide(image=None, group=group[:1], tracks=tracks, timestamp=2.0,
                            specification={"task": "mot"})
    assert out2.metadata["shortlist_truncated_ids"] == 1, out2.metadata

    print(json.dumps({
        "status": "TRACKJEV_ADAPTER_SELFTEST_PASS",
        "decisions": [d.to_dict() for d in out.decisions],
        "adapter_metadata": {k: v for k, v in out.metadata.items()
                             if k in ("backend", "shortlist_limit", "shortlist_truncated_ids",
                                      "option_count", "evidence_format", "native_probabilities")},
        "memory": memory["asked"],
        "memory_updates": {str(k): v for k, v in memory["updates"].items()},
        "truncation_check": out2.metadata["shortlist_truncated_ids"],
    }, indent=1, ensure_ascii=False)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
