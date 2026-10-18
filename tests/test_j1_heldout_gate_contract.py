import importlib.util
import json
from pathlib import Path


def _load_gate():
    path = Path(__file__).parents[1] / "tools/groundjev/run_j1_heldout_gate.py"
    spec = importlib.util.spec_from_file_location("j1_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_final_training_tree(tmp_path: Path, *, step: int = 1200):
    training = tmp_path / "training"
    adapter = training / "adapter-step1200"
    adapter.mkdir(parents=True)
    durable = training / "latest_training_state.pt"
    durable.write_bytes(b"durable-state")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (adapter / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "/models/OneJev-0.8B",
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj"],
    }))
    manifest = tmp_path / "rows.jsonl"
    manifest.write_text('{"id":"row-1"}\n')
    (training / "args.json").write_text(json.dumps({"max_steps": 1200, "train_file": str(manifest)}))
    (training / "latest_training_state.json").write_text(json.dumps({
        "format": "groundjev-full-fp32-adam-rng-v1",
        "step": step,
        "path": str(durable),
        "bytes": durable.stat().st_size,
        "FP32_masters_and_Adam_moments_saved": True,
        "scheduler_and_all_RNG_saved": True,
    }))
    return training, manifest


def test_resumed_gate_environment_is_available_before_any_branch():
    gate = _load_gate()
    env = gate.build_eval_env(5)
    assert env["CUDA_VISIBLE_DEVICES"] == "5"
    assert env["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"


def test_final_receipt_requires_exact_step_and_durable_contract(tmp_path):
    gate = _load_gate()
    training, manifest = _write_final_training_tree(tmp_path)
    ready, receipt = gate.final_training_receipt(training, training_manifest=manifest)
    assert ready is True
    assert receipt["step"] == 1200
    bad_training, bad_manifest = _write_final_training_tree(tmp_path / "bad", step=1201)
    ready, detail = gate.final_training_receipt(bad_training, training_manifest=bad_manifest)
    assert ready is False
    assert "equal 1200" in detail["reason"]


def test_inference_receipt_preserves_rejected_frames_and_rejects_gt_fallback(tmp_path):
    gate = _load_gate()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"images": ["a.jpg", "b.jpg", "c.jpg"], "scope": "fixed"}))
    result = tmp_path / "result.json"
    result.write_text(json.dumps({
        "status": "VIDEO_INFERENCE_COMPLETE_WITH_REJECTED_PERCEPTION",
        "task": "mot",
        "scope": "fixed",
        "frames": 3,
        "all_input_frames_processed": True,
        "gt_read_by_runtime": False,
        "future_frames_read_by_runtime": False,
        "perception_rejected_frames": [1],
        "latencies_seconds": [0.1, 0.2, 0.3],
    }))
    ready, detail = gate.valid_inference_receipt(result, manifest, expected_task="mot", expected_scope="fixed")
    assert ready is True
    assert detail["rejected_frames"] == 1
    payload = json.loads(result.read_text())
    payload["gt_read_by_runtime"] = True
    result.write_text(json.dumps(payload))
    ready, detail = gate.valid_inference_receipt(result, manifest, expected_task="mot", expected_scope="fixed")
    assert ready is False
    assert "GT" in detail["reason"]


def _write_valid_result(path: Path, *, task="rmot", scope="fixed", frames=2):
    path.mkdir(parents=True, exist_ok=True)
    manifest = path.parent / "manifest.json"
    manifest.write_text(json.dumps({"images": ["a.jpg"] * frames, "scope": scope, "task": task}))
    (path / "result.json").write_text(json.dumps({
        "status": "VIDEO_INFERENCE_COMPLETE",
        "task": task,
        "scope": scope,
        "frames": frames,
        "all_input_frames_processed": True,
        "gt_read_by_runtime": False,
        "future_frames_read_by_runtime": False,
        "perception_rejected_frames": [],
    }))
    return manifest


def test_resume_classifier_covers_unfinished_completed_and_corrupt_outputs(tmp_path):
    gate = _load_gate()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"images": ["a.jpg"], "scope": "fixed", "task": "rmot"}))
    unfinished = gate.inspect_eval_output(tmp_path / "missing", manifest, expected_task="rmot", expected_scope="fixed")
    assert unfinished["status"] == "RUN_INFERENCE"
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "log.txt").write_text("partial")
    assert gate.inspect_eval_output(partial, manifest, expected_task="rmot", expected_scope="fixed")["status"] == "INVALID_PARTIAL_OUTPUT"
    valid_dir = tmp_path / "valid"
    (valid_dir).mkdir()
    (valid_dir / "result.json").write_text(json.dumps({
        "status": "VIDEO_INFERENCE_COMPLETE", "task": "rmot", "scope": "fixed", "frames": 1,
        "all_input_frames_processed": True, "gt_read_by_runtime": False,
        "future_frames_read_by_runtime": False, "perception_rejected_frames": [],
    }))
    assert gate.inspect_eval_output(valid_dir, manifest, expected_task="rmot", expected_scope="fixed")["status"] == "RUN_METRICS"
    (valid_dir / "official_metrics.json").write_text(json.dumps({
        "status": "OFFICIAL_REGISTERED_MULTITASK_SCOPE_EVALUATION_COMPLETE", "task": "rmot", "frames": 1, "metrics": {"HOTA": 0.1}
    }))
    assert gate.inspect_eval_output(valid_dir, manifest, expected_task="rmot", expected_scope="fixed")["status"] == "COMPLETE"
    (valid_dir / "result.json").write_text("{}")
    assert gate.inspect_eval_output(valid_dir, manifest, expected_task="rmot", expected_scope="fixed")["status"] == "INVALID_RESULT"
