"""Bulk integer-label transfer for unchanged native GroundingDINO metadata."""
from __future__ import annotations

from contextlib import contextmanager
from types import MethodType

import torch


@contextmanager
def bulk_label_metadata(runtime):
    """Keep native prediction, replacing per-label CUDA reads with one copy.

    GroundingDINO's final label_names loop reads every predicted class ID as
    a Python scalar, before cross-chunk NMS. The head's integer labels can
    travel to CPU together for that native loop and return to their original
    device before downstream consumers. Boxes/scores/features stay untouched.
    """
    model, head = runtime.model, runtime.model.bbox_head
    if model.training or head.training:
        raise ValueError("Bulk label metadata is an evaluation-only operation")
    original_model, original_head = model.predict, head.predict
    model_override, head_override = "predict" in model.__dict__, "predict" in head.__dict__
    stats = {"head_calls": 0, "integer_labels_transferred": 0}

    def head_predict(self, *args, **kwargs):
        predictions = original_head(*args, **kwargs)
        for prediction in predictions:
            stats["integer_labels_transferred"] += len(prediction.labels)
            prediction.labels = prediction.labels.cpu()
        stats["head_calls"] += 1
        return predictions

    def model_predict(self, *args, **kwargs):
        results = original_model(*args, **kwargs)
        for sample in results:
            sample.pred_instances.labels = sample.pred_instances.labels.to(runtime.device)
        return results

    head.predict = MethodType(head_predict, head)
    model.predict = MethodType(model_predict, model)
    try:
        yield stats
    finally:
        for module, original, overridden in ((model, original_model, model_override),
                                             (head, original_head, head_override)):
            if overridden:
                module.predict = original
            else:
                delattr(module, "predict")


__all__ = ["bulk_label_metadata"]
