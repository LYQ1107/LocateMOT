import json
import tempfile
import unittest
from pathlib import Path

from tools.groundjev.coupled.build_joint_identity_corpus import build, normalize_row


class JointIdentityCorpusContract(unittest.TestCase):
    def _row(self, *, task="identity", candidate_source="actual_native_GroundAnything_DLM"):
        return {
            "id": "fixture-1",
            "split": "train",
            "source": "MOT17-02-FRCNN",
            "task": task,
            "state": {
                "current_frame": "<image:1>",
                "timestamp": 3,
                "task": {"task": "mot"},
                "candidates": [{"index": 0, "box": [0, 0, 10, 10], "label": "person"}],
                "historical_tracks": [],
            },
            "question": {"type": "choice", "instructions": "choose"},
            "target": {"NEW": 1.0},
            "media": [{"type": "image", "path": "/tmp/frame.jpg"}],
            "supervision": {
                "candidate_source": candidate_source,
                "GT_IDs_in_model_input": False,
                "future_images_read": False,
            },
        }

    def test_normalization_preserves_provenance_and_causal_options(self):
        row = normalize_row(self._row(), source_path=Path("fixture.jsonl"), source_row=1)
        self.assertEqual(row["task"], "mot")
        self.assertEqual(row["native_candidates"][0]["box"], [0, 0, 10, 10])
        self.assertEqual(row["dynamic_identity_options"], ["NEW", "NO_MATCH", "REJECT"])
        self.assertEqual(row["source_provenance"]["gt_in_model_input"], False)
        self.assertEqual(row["proposal_source"], "actual_native_GroundAnything_DLM")

    def test_future_evidence_is_rejected(self):
        row = self._row()
        row["supervision"]["future_images_read"] = True
        with self.assertRaises(ValueError):
            normalize_row(row, source_path=Path("fixture.jsonl"), source_row=1)

    def test_native_gate_rejects_teacher_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.jsonl"
            output = Path(tmp) / "out.jsonl"
            source.write_text(json.dumps(self._row(
                task="category_leaf", candidate_source="registered train-Base GT teacher crops"
            )) + "\n")
            with self.assertRaises(ValueError):
                build([source], output=output, require_native_candidates=True)

    def test_native_gate_rejects_explicit_gt_injection(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.jsonl"
            output = Path(tmp) / "out.jsonl"
            row = self._row()
            row["supervision"]["GT_IDs_in_model_input"] = True
            source.write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                build([source], output=output, require_native_candidates=True)

    def test_build_writes_manifest_with_explicit_teacher_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.jsonl"
            output = Path(tmp) / "out.jsonl"
            source.write_text(json.dumps(self._row(
                task="category_leaf", candidate_source="registered train-Base GT teacher crops"
            )) + "\n")
            receipt = build([source], output=output)
            self.assertEqual(receipt["rows"], 1)
            self.assertFalse(receipt["native_ovmot_training_rows"])
            self.assertEqual(json.loads(output.read_text())["task"], "ovmot")
            self.assertTrue(output.with_suffix(".jsonl.manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
