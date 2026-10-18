import importlib.util
from pathlib import Path


def _load_audit_module():
    path = Path(__file__).parents[1] / "tools/groundjev/g2/audit_global_event_accounting.py"
    spec = importlib.util.spec_from_file_location("g2_event_audit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_trainer_module():
    path = Path(__file__).parents[1] / "tools/groundjev/g2/train_g2_dlm.py"
    spec = importlib.util.spec_from_file_location("g2_trainer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_two_rank_global_statistics_keep_counts_as_sums():
    trainer = _load_trainer_module()
    result = trainer.global_event_statistics(
        loss_sum_across_ranks=1.0 + 1.4,
        tokens_sum_across_ranks=100 + 120,
        samples_sum_across_ranks=16 + 16,
        world_size=2,
    )
    assert result["loss"] == 1.2
    assert result["effective_tokens_global"] == 220.0
    assert result["samples_global"] == 32.0


def test_historical_audit_reconstructs_counts_without_rewriting_raw_log():
    audit = _load_audit_module()
    raw = b'{"step": 4, "samples_global": 16, "effective_tokens_global": 220, "loss": 1.2}\n'
    report = audit.audit_events(raw, world_size=2, effective_global_batch=32)
    event = report["events"][0]
    assert event["event_schema_version"] == "g2-training-event-v1-local-mean-ambiguous"
    assert event["corrected_samples_global"] == 32
    assert event["corrected_effective_tokens_global"] == 440
    assert report["raw_log_unchanged"] is True


def test_v2_audit_preserves_global_counts():
    audit = _load_audit_module()
    raw = (b'{"event_schema_version":"g2-training-event-v2-global-accounting", '
           b'"step": 4, "samples_global": 32, "effective_tokens_global": 220, "loss": 1.2}\n')
    report = audit.audit_events(raw, world_size=2, effective_global_batch=32)
    event = report["events"][0]
    assert event["corrected_samples_global"] == 32
    assert event["corrected_effective_tokens_global"] == 220
    assert event["expected_global_batch_mismatch"] is False
