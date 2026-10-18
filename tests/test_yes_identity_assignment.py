from dataclasses import dataclass


@dataclass
class Track:
    track_id: int


def test_verifier_is_fused_inside_one_global_assignment_without_duplicate_tracks():
    from locatemot.groundjev.assignment import assign_candidates
    from locatemot.groundjev.verification_policy import SelectiveVerificationPolicy, VerificationPolicyConfig
    from locatemot.groundjev.yes_verification import YesNoResult

    tracks = [Track(10), Track(11)]
    decisions = [
        {"probabilities": {"10": 0.60, "11": 0.59, "NEW": 0.58, "IGNORE": 0.01}},
        {"probabilities": {"10": 0.59, "11": 0.10, "NEW": 0.58, "IGNORE": 0.01}},
    ]
    policy = SelectiveVerificationPolicy(VerificationPolicyConfig(enabled=True, semantic_weight=1.0))
    no_for_track_10 = YesNoResult("VERIFIED", True, 0.0, "No", "identity", (1,), "ckpt", 1.0)
    yes_for_track_11 = YesNoResult("VERIFIED", True, 1.0, "Yes", "identity", (1,), "ckpt", 1.0)
    assignment, actions, masked, audit = assign_candidates(
        decisions,
        tracks,
        native_probabilities=True,
        verification_results={(0, 10): no_for_track_10, (0, 11): yes_for_track_11},
        verification_policy=policy,
        return_audit=True,
    )
    assert actions == ("NEW", "IGNORE")
    assert not masked
    # The verifier changes the pair costs but never adds a second assignment
    # pass or duplicates a track column.
    assigned_track_columns = [column for column in assignment.values() if column < len(tracks)]
    assert len(assigned_track_columns) == len(set(assigned_track_columns))
    assert any(item["track_id"] == 10 and item["p_yes"] == 0.0 for item in audit)


def test_no_does_not_turn_into_new_and_disabled_policy_is_baseline_equivalent():
    from locatemot.groundjev.assignment import assign_candidates
    from locatemot.groundjev.verification_policy import SelectiveVerificationPolicy
    from locatemot.groundjev.yes_verification import YesNoResult

    tracks = [Track(1)]
    decisions = [{"probabilities": {"1": 0.8, "NEW": 0.7, "IGNORE": 0.01}}]
    no = YesNoResult("VERIFIED", True, 0.0, "No", "identity", (0,), "ckpt", 1.0)
    baseline = assign_candidates(decisions, tracks, native_probabilities=True)
    disabled = assign_candidates(
        decisions,
        tracks,
        native_probabilities=True,
        verification_results={(0, 1): no},
        verification_policy=SelectiveVerificationPolicy(),
    )
    assert baseline == disabled
