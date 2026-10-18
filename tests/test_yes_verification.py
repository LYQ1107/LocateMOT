import json
import importlib.util
import time
from pathlib import Path


def _load_script(name):
    path = Path(__file__).parents[1] / "tools/groundjev/yestrack" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_probability_direction_and_prompt_contract():
    from locatemot.groundjev.yes_verification import build_verification_prompt, parse_yes_no_output
    p_yes, answer, _ = parse_yes_no_output({"yes_logit": 3.0, "no_logit": 0.0})
    p_no, answer_no, _ = parse_yes_no_output({"yes_logit": 0.0, "no_logit": 3.0})
    assert p_yes > 0.5 and answer == "Yes"
    assert p_no < 0.5 and answer_no == "No"
    prompt = build_verification_prompt(task="rmot", expression="a red coat", bbox=[0, 0, 1, 1])
    assert "ground truth" not in prompt.lower()


def test_causal_history_rejects_future_frame():
    from locatemot.groundjev.yes_verification import VerificationRequest, causal_history_frame_ids
    assert causal_history_frame_ids([{"frame": 1}, {"frame": 4}], 4) == (1,)
    try:
        VerificationRequest(task="rmot", expression="person", question="Answer Yes or No", media=(), current_frame=4, history_frame_ids=(4,))
    except ValueError as error:
        assert "causal" in str(error)
    else:
        raise AssertionError("future history was accepted")


def test_no_does_not_create_new_identity_and_global_assignment_is_one_to_one():
    from locatemot.groundjev.verification_policy import CandidateEvidence, SelectiveVerificationPolicy, VerificationPolicyConfig, global_assignment
    from locatemot.groundjev.yes_verification import YesNoResult
    policy = SelectiveVerificationPolicy(VerificationPolicyConfig(enabled=True))
    no = YesNoResult("VERIFIED", True, 0.1, "No", "membership", (0,), None, 1.0)
    assert policy.lifecycle_action(base_state="NEW", result=no, candidate_is_new=True) == "IGNORE"
    assignment = global_assignment([[0.9, 0.8], [0.85, 0.1]])
    assert assignment == {0: 1, 1: 0}


def test_cache_invalidation_and_disabled_baseline_equivalence():
    from locatemot.groundjev.semantic_temporal_cache import SemanticTemporalCache
    from locatemot.groundjev.verification_policy import SelectiveVerificationPolicy, VerificationPolicyConfig, baseline_equivalent
    cache = SemanticTemporalCache(enabled=True, ttl_frames=3)
    key_a = cache.make_key(track_id=1, task="rmot", expression=" Person  ", verifier_checkpoint_hash="ckpt")
    key_b = cache.make_key(track_id=2, task="rmot", expression="person", verifier_checkpoint_hash="ckpt")
    cache.put(key_a, p_yes=0.8, frame=2, evidence_frame_ids=(1,), query_version="q1")
    cache.put(key_b, p_yes=0.2, frame=2, evidence_frame_ids=(1,), query_version="q1")
    assert cache.get(key_a, current_frame=3).p_yes == 0.8
    assert cache.invalidate(track_id=1, reason="conflict") == 1
    assert cache.get(key_a, current_frame=3) is None
    assert cache.get(key_b, current_frame=3) is not None
    policy = SelectiveVerificationPolicy()
    assert baseline_equivalent() is True
    assert policy.fuse_association_score(0.7, None) == 0.7


def test_corpus_split_leakage_gt_prompt_and_rejected_frame_audits(tmp_path):
    audit = _load_script("audit_yesno_corpus")
    rows = [
        {"id": "a", "split": "train", "source": "x", "task": "rmot", "question": "Answer Yes or No", "media": ["a.jpg"], "video": "v", "frame": 1, "supervision": {"gt_used_for_prompt": False, "future_evidence": False}},
        {"id": "b", "split": "val", "source": "x", "task": "rmot", "question": "Answer Yes or No", "media": ["b.jpg"], "video": "v", "frame": 2, "supervision": {"gt_used_for_prompt": False, "future_evidence": False}, "rejected_frame": True},
    ]
    report = audit.audit(rows)
    assert report["status"] == "YESNO_CORPUS_AUDIT_FAILED"
    assert report["split_leakage"]["v"] == ["train", "val"]
    assert report["rejected_frame_rows_preserved"] == 1
    rows[0]["question"] = "Ground truth says Yes"
    report = audit.audit(rows)
    assert report["gt_in_prompt"] is True


def test_timeout_and_cpu_mock_lifecycle():
    from locatemot.groundjev.yes_verification import VerificationRequest, YesNoVerifier
    request = VerificationRequest(task="rmot", expression="person", question="Is this person? Answer Yes or No.", media=("frame.jpg",), current_frame=2, history_frame_ids=(1,))
    verifier = YesNoVerifier(backend=lambda _: {"p_yes": 0.9}, enabled=True, timeout_ms=100)
    result = verifier.verify(request)
    assert result.status == "VERIFIED" and result.p_yes > 0.5
    slow = YesNoVerifier(backend=lambda _: (time.sleep(0.01), {"p_yes": 0.9})[1], enabled=True, timeout_ms=1)
    assert slow.verify(request).status == "FAILED_TIMEOUT"
    disabled = YesNoVerifier(enabled=False).verify(request)
    assert disabled.status == "DISABLED" and disabled.used is False


def test_identity_and_semantic_adapters_share_engine_and_keep_history_causal():
    from locatemot.groundjev.yes_verification import YesNoVerifier

    seen = []
    verifier = YesNoVerifier(backend=lambda request: seen.append(request) or {"p_yes": 0.75}, enabled=True)
    identity = verifier.verify_identity(
        track_id=3,
        candidate_id=9,
        current_image="current.jpg",
        current_frame=5,
        history_images=("anchor.jpg", "recent.jpg"),
        history_frame_ids=(1, 4),
    )
    semantic = verifier.verify_semantic(
        task="rmot",
        expression="the red car",
        current_image="current.jpg",
        current_frame=6,
        track_id=3,
        history_images=("recent.jpg",),
        history_frame_ids=(5,),
    )
    assert identity.p_yes == semantic.p_yes == 0.75
    assert seen[0].history_frame_ids == (1, 4)
    assert seen[1].history_frame_ids == (5,)
    assert all(frame < request.current_frame for request in seen for frame in request.history_frame_ids)


def test_calibrated_fusion_is_explicit_and_not_default():
    from locatemot.groundjev.verification_policy import (
        SelectiveVerificationPolicy,
        VerificationPolicyConfig,
        fit_logistic_calibrator,
    )
    from locatemot.groundjev.yes_verification import YesNoResult

    temperature, bias = fit_logistic_calibrator([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], steps=20)
    policy = SelectiveVerificationPolicy(VerificationPolicyConfig(
        enabled=True,
        semantic_weight=1.0,
        fusion_mode="calibrated",
        calibration_temperature=temperature,
        calibration_bias=bias,
    ))
    result = YesNoResult("VERIFIED", True, 0.9, "Yes", "identity", (1,), "ckpt", 1.0)
    assert policy.fuse_association_score(0.1, result) > 0.5
