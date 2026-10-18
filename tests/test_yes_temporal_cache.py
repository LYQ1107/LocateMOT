def test_cache_key_separates_track_task_query_and_checkpoint():
    from locatemot.groundjev.semantic_temporal_cache import SemanticTemporalCache

    cache = SemanticTemporalCache(enabled=True, ttl_frames=4)
    first = cache.make_key(track_id=7, task="rmot", expression="red car", verifier_checkpoint_hash="a")
    other_query = cache.make_key(track_id=7, task="rmot", expression="blue car", verifier_checkpoint_hash="a")
    other_task = cache.make_key(track_id=7, task="ovmot", expression="red car", verifier_checkpoint_hash="a")
    other_ckpt = cache.make_key(track_id=7, task="rmot", expression="red car", verifier_checkpoint_hash="b")
    cache.put(first, p_yes=0.8, frame=10, evidence_frame_ids=(9,), query_version="q1")
    assert cache.get(first, current_frame=11).p_yes == 0.8
    assert cache.get(other_query, current_frame=11) is None
    assert cache.get(other_task, current_frame=11) is None
    assert cache.get(other_ckpt, current_frame=11) is None


def test_query_change_reappearance_conflict_and_ttl_invalidate_semantics_only():
    from locatemot.groundjev.semantic_temporal_cache import SemanticTemporalCache

    cache = SemanticTemporalCache(enabled=True, ttl_frames=2)
    key = cache.make_key(track_id=3, task="rmot", expression="person", verifier_checkpoint_hash="ckpt")
    cache.put(key, p_yes=0.7, frame=1, evidence_frame_ids=(0,), query_version="q1")
    assert cache.invalidate(track_id=3, reason="query_change") == 1
    assert len(cache) == 0
    cache.put(key, p_yes=0.7, frame=3, evidence_frame_ids=(2,), query_version="q2")
    assert cache.invalidate(track_id=3, reason="reappearance") == 1
    cache.put(key, p_yes=0.7, frame=3, evidence_frame_ids=(2,), query_version="q3")
    assert cache.get(key, current_frame=6) is None
    assert len(cache) == 0


def test_future_evidence_cannot_be_cached_or_read():
    from locatemot.groundjev.semantic_temporal_cache import SemanticTemporalCache

    cache = SemanticTemporalCache(enabled=True)
    key = cache.make_key(track_id=1, task="mot", expression="", verifier_checkpoint_hash="ckpt")
    try:
        cache.put(key, p_yes=0.5, frame=4, evidence_frame_ids=(5,), query_version="q")
    except ValueError as error:
        assert "causal" in str(error)
    else:
        raise AssertionError("future evidence was accepted")
