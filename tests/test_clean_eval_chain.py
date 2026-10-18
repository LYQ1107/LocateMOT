import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).parents[1] / "tools/groundjev/g2/audit_clean_eval_chain.py"
    spec = importlib.util.spec_from_file_location("clean_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_clean_bundle_receipt_rejects_unreceipted_historical_bundle(tmp_path):
    audit = _load_module()
    bundle = tmp_path / "GroundAnything-DLM-G2-CLEAN-V2"
    base = tmp_path / "GroundAnything-DLM"
    bundle.mkdir()
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"base")
    (bundle / "model.safetensors").write_bytes(b"merged")
    adapter = tmp_path / "adapter.pt"
    adapter.write_bytes(b"adapter")
    manifest = tmp_path / "rows.jsonl"
    manifest.write_text("rows")
    receipt_path = tmp_path / "export_receipt.json"
    receipt_path.write_text(json.dumps({
        "status": "G2_NATIVE_MERGED_BUNDLE_EXPORTED",
        "manifest_sha256": "clean",
        "source_bundle_overwritten": False,
        "native_tensor_count": 569,
        "output_bundle": str(bundle.resolve()),
        "output_model_safetensors_sha256": audit.sha256(bundle / "model.safetensors"),
        "base_bundle": str(base.resolve()),
        "base_model_safetensors_sha256": audit.sha256(base / "model.safetensors"),
        "adapter": str(adapter),
    }))
    assert audit.validate_bundle(bundle, receipt_path, expected_manifest_sha256="clean")["native_tensor_count"] == 569
    bad = tmp_path / "formal_g2_1200"
    bad.mkdir()
    (bad / "model.safetensors").write_bytes(b"merged")
    bad_receipt = tmp_path / "bad_receipt.json"
    bad_receipt.write_text(receipt_path.read_text().replace(str(bundle.resolve()), str(bad.resolve())))
    try:
        audit.validate_bundle(bad, bad_receipt, expected_manifest_sha256="clean")
    except ValueError as error:
        assert "contaminated" in str(error)
    else:
        raise AssertionError("historical G2-PL bundle was accepted")


def test_same_scope_result_keeps_rejected_frames_and_gt_is_forbidden(tmp_path):
    audit = _load_module()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "images": ["0.jpg", "1.jpg", "2.jpg"],
        "task": "mot",
        "scope": "fixed",
        "official_test_labels_read": False,
        "runtime_gt_fields": False,
    }))
    result = tmp_path / "result.json"
    result.write_text(json.dumps({
        "status": "VIDEO_INFERENCE_COMPLETE_WITH_REJECTED_PERCEPTION",
        "task": "mot",
        "scope": "fixed",
        "frames": 3,
        "all_input_frames_processed": True,
        "gt_read_by_runtime": False,
        "future_frames_read_by_runtime": False,
        "perception_rejected_frames": [2],
        "latencies_seconds": [0.1, 0.1, 0.1],
    }))
    detail = audit.validate_result(result, manifest)
    assert detail["rejected_frames"] == 1
    assert audit.denominator_explanation()["rate_percent"] == 11.04
    payload = json.loads(result.read_text())
    payload["future_frames_read_by_runtime"] = True
    result.write_text(json.dumps(payload))
    try:
        audit.validate_result(result, manifest)
    except ValueError as error:
        assert "forbidden" in str(error)
    else:
        raise AssertionError("GT/future-frame fallback was accepted")
