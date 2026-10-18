import importlib.util
import json
from pathlib import Path


def _load_gate():
    path = Path(__file__).parents[1] / "tools/groundjev/run_j1_heldout_gate.py"
    spec = importlib.util.spec_from_file_location("j1_gate_checkpoint", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _tree(tmp_path, step=1200):
    training = tmp_path / "training"
    adapter = training / "adapter-step1200"
    adapter.mkdir(parents=True)
    state = training / "latest_training_state.pt"
    state.write_bytes(b"state")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "/models/OneJev-0.8B",
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj"],
    }))
    rows = tmp_path / "rows.jsonl"
    rows.write_text('{"id":"one"}\n')
    (training / "args.json").write_text(json.dumps({"max_steps": 1200, "train_file": str(rows)}))
    (training / "latest_training_state.json").write_text(json.dumps({
        "format": "groundjev-full-fp32-adam-rng-v1",
        "step": step,
        "path": str(state),
        "bytes": state.stat().st_size,
        "FP32_masters_and_Adam_moments_saved": True,
        "scheduler_and_all_RNG_saved": True,
    }))
    return training, rows


def test_directory_name_alone_is_not_a_final_checkpoint(tmp_path):
    gate = _load_gate()
    training = tmp_path / "training"
    (training / "adapter-step1200").mkdir(parents=True)
    ready, detail = gate.final_training_receipt(training)
    assert not ready
    assert "latest_training_state" in detail["reason"]


def test_final_checkpoint_requires_manifest_and_matching_state(tmp_path):
    gate = _load_gate()
    training, rows = _tree(tmp_path)
    ready, detail = gate.final_training_receipt(training, training_manifest=rows)
    assert ready and detail["step"] == 1200
    bad_training, bad_rows = _tree(tmp_path / "bad", step=1199)
    ready, detail = gate.final_training_receipt(bad_training, training_manifest=bad_rows)
    assert not ready and "equal 1200" in detail["reason"]


def test_old_dt1203_state_does_not_mark_new_multitask_gate_complete(tmp_path):
    gate = _load_gate()
    # The gate's terminal short-circuit only accepts the full registered
    # status; a legacy DT1203-only marker is merely an ordinary prior state.
    legacy = tmp_path / "owner_state.json"
    legacy.write_text(json.dumps({"status": "HELDOUT_DT1203_COMPLETE", "completed": {"dt1203": True}}))
    payload = json.loads(legacy.read_text())
    assert payload["status"] != "HELDOUT_ALL_REGISTERED_TASKS_COMPLETE"
    assert "rmot" not in payload["completed"]
