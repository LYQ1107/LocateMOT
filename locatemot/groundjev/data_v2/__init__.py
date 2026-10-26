"""FullDataV2 unified tracking data interface (M2).

Public entry points:

* :data:`registry.DATASETS` — the only place that knows physical dataset paths
* :class:`schema.TrainingRecord` — the unified supervision row contract
* :func:`coverage.build_report` — real physical coverage audit
* :mod:`mot_challenge`, :mod:`rmot`, :mod:`ovmot` — format adapters
"""
from .registry import DATASETS, DATA_V2_ROOT, DatasetSpec, availability_table, dataset, load_frozen_protocol
from .schema import AnnotationQuality, IdentityDecision, SCHEMA_VERSION, TrainingRecord, validate_record

__all__ = [
    "DATASETS", "DATA_V2_ROOT", "DatasetSpec", "availability_table", "dataset",
    "load_frozen_protocol", "AnnotationQuality", "IdentityDecision", "SCHEMA_VERSION",
    "TrainingRecord", "validate_record",
]
