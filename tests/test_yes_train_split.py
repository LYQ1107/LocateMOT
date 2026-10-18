import importlib.util
from pathlib import Path


def _load(name):
    path = Path(__file__).parents[1] / "tools/groundjev/yestrack" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _row(row_id, video, split, target="Yes", **extra):
    row = {
        "id": row_id,
        "split": split,
        "source": "native_candidates",
        "task": "rmot",
        "state": "verification",
        "question": "Does this candidate match the expression? Answer Yes or No.",
        "target": target,
        "media": ["candidate.jpg"],
        "video": video,
        "frame": 3,
        "supervision": {"gt_used_for_prompt": False, "future_evidence": False},
    }
    row.update(extra)
    return row


def test_validator_accepts_native_rows_and_rejects_video_split_leakage():
    validator = _load("validate_yesno_inputs")
    report = validator.validate([_row("a", "seq-train", "train")])
    assert report["status"] == "YESNO_INPUTS_VALID"
    report = validator.validate([_row("a", "seq", "train"), _row("b", "seq", "calibration", "No")])
    assert report["status"] == "YESNO_INPUTS_INVALID"
    assert any("video_split_leakage" in error for error in report["errors"])


def test_validator_rejects_novel_training_and_noncausal_history():
    validator = _load("validate_yesno_inputs")
    row = _row("a", "seq", "train", novel_category=True, history_frame=3)
    report = validator.validate([row])
    assert report["status"] == "YESNO_INPUTS_INVALID"
    assert any("novel_training_label" in error for error in report["errors"])
    assert any("noncausal_history_frame" in error for error in report["errors"])


def test_ovmot_preparer_never_accepts_novel_training_labels():
    preparer = _load("prepare_ovmot_yesno")
    row = _row("a", "tao-base", "train", category="car", novel=True)
    try:
        preparer.prepare([row])
    except ValueError as error:
        assert "novel" in str(error).lower()
    else:
        raise AssertionError("Novel OVMOT training label was accepted")
