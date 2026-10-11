"""One-file actual GroundingDINO/BERT plus persistent U2 core inference.

No shape-only visual fallback or external pretrained model is constructed.
Native MMDetection code owns preprocessing and all detection computations.
"""
from __future__ import annotations

import copy
from collections import OrderedDict
from pathlib import Path
import tempfile

import torch
from torch import nn

from .grounding_inference import GroundingRuntime, PREPROCESSING_CONTRACT
from .u2_grounding_candidates import run_u2_candidates
from ..models.u2.shared_track_core import SharedTrackCore
from ..models.u2.track_state import UnifiedTrackState

FORMAT = "locatemot-u-u2-shared-track-core-v1"
U1R2_BASE = "56507f4aa9a12e76d6f9f6702f0f0b38163666ec"
U1_FOUNDATION_SHA = "4efc35ce70be5cddf2c4d9a0f91bc4edef3e9ec5277a819211ba4ac42e0d7d26"


def parameter_sharing(foundation: nn.Module, core: SharedTrackCore) -> dict:
    """Count actual unique parameters, including the full real language model."""
    foundation_n = sum(p.numel() for p in foundation.parameters())
    adapter_ids = {id(p) for p in core.ovmot.parameters()}
    shared_core = sum(p.numel() for p in core.parameters() if id(p) not in adapter_ids)
    specific = sum(p.numel() for p in core.ovmot.parameters())
    shared = foundation_n + shared_core
    head_modules = (foundation.bbox_head, core.association, core.lifecycle, core.motion)
    shared_head_ids = {id(p) for module in head_modules for p in module.parameters()}
    shared_head_n = sum(p.numel() for p in list(foundation.parameters()) + list(core.parameters())
                        if id(p) in shared_head_ids)
    shared_non_head = shared - shared_head_n
    # Exclude the actual shared detection/association/motion/lifecycle heads.
    # Retain the entire task-specific adapter in the denominator, including
    # its heads; this is a conservative lower bound for non-head sharing.
    fraction = shared_non_head / (shared_non_head + specific)
    return {"foundation_shared": foundation_n, "core_shared": shared_core,
            "shared_parameters": shared, "task_specific_parameters": {"mot": 0, "ovmot": specific},
            "task_embedding_parameters": core.ovmot.task_embeddings.weight.numel(),
            "shared_head_parameters_excluded": shared_head_n,
            "shared_non_head_parameters": shared_non_head,
            "shared_non_head_fraction": fraction, "minimum": .90, "pass": fraction >= .90,
            "method": "unique actual parameters; shared detection/association/motion/lifecycle heads excluded; entire adapter conservatively task-specific non-head"}


def make_payload(runtime: GroundingRuntime, core: SharedTrackCore, class_text: dict,
                 vocabulary: dict, tokenizer_files: dict[str, bytes], metadata: dict,
                 *, reference_only: bool) -> dict:
    """Embed exact tensors/config/assets; eligibility is the formal packer's job."""
    if runtime.missing_keys or runtime.unexpected_keys:
        raise ValueError("Only a complete native foundation can be packaged")
    if class_text["foundation_sha256"] != runtime.checkpoint_sha256:
        raise ValueError("Class features and actual embedded foundation differ")
    cfg = copy.deepcopy(runtime.config.to_dict())
    # Native detector construction inserts these derived fields into its
    # supplied bbox-head config. Its constructor supplies them again on load.
    for key in ("share_pred_layer", "num_pred_layer", "as_two_stage"):
        cfg["model"]["bbox_head"].pop(key, None)
    for key in ("num_classes", "embed_dims", "num_matching_queries"):
        cfg["model"]["dn_cfg"].pop(key, None)
    cfg["model"]["language_model"]["name"] = "embedded-u2-shared-bert"
    bert_config = runtime.model.language_model.language_backbone.body.model.config.to_dict()
    # HF from_pretrained selects SDPA on this installation; a bare model
    # constructor otherwise chooses eager despite identical tensor weights.
    bert_config["attn_implementation"] = runtime.model.language_model.language_backbone.body.model.config._attn_implementation
    sharing = parameter_sharing(runtime.model, core)
    if not sharing["pass"]:
        raise ValueError("Actual shared-parameter gate failed")
    return {"format": FORMAT, "stage": "U2", "reference_only": reference_only,
            "foundation": {"state_dict": {k: v.detach().cpu() for k, v in runtime.model.state_dict().items()},
                           "config": cfg, "bert_config": bert_config, "tokenizer_files": tokenizer_files},
            "core": {k: v.detach().cpu() for k, v in core.state_dict().items()},
            "core_config": {"hidden_dim": core.hidden_dim, "max_age": core.max_age},
            "class_text": class_text, "vocabulary": vocabulary,
            "metadata": {**metadata, "u1r2_base_commit": U1R2_BASE,
                         "u1_foundation_source_sha256": U1_FOUNDATION_SHA,
                         "foundation_sha256": runtime.checkpoint_sha256,
                         "parameter_sharing": sharing, "preprocessing": PREPROCESSING_CONTRACT,
                         "task_embeddings": {"mapping": {"mot": 0, "ovmot": 1},
                                             "location": "ovmot.task_embeddings.weight",
                                             "use": "bounded semantic gate only; ordinary gate exactly zero"}}}


def save_payload(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def _register_embedded_language() -> None:
    from mmengine.model import BaseModel
    from mmdet.registry import MODELS
    from mmdet.models.language_models.bert import BertEncoder, BertModel
    from transformers import AutoTokenizer, BertConfig, BertModel as HFBertModel

    if MODELS.get("U2EmbeddedBertModel") is not None:
        return

    @MODELS.register_module()
    class U2EmbeddedBertModel(BertModel):
        """Native forward with construction from embedded architecture/tokens."""

        def __init__(self, name, bert_config, tokenizer_files, max_tokens=256,
                     pad_to_max=True, use_sub_sentence_represent=False,
                     special_tokens_list=None, add_pooling_layer=False,
                     num_layers_of_embedded=1, use_checkpoint=False, **kwargs):
            BaseModel.__init__(self, **kwargs)
            self.max_tokens, self.pad_to_max = max_tokens, pad_to_max
            # Tokenizers load these small assets eagerly. No pretrained BERT
            # weight file, external vocabulary or network request is needed.
            with tempfile.TemporaryDirectory(prefix="locatemot-u2-tokenizer-") as directory:
                for filename, content in tokenizer_files.items():
                    (Path(directory) / filename).write_bytes(content)
                self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
            config = BertConfig.from_dict(copy.deepcopy(bert_config),
                                          attn_implementation=bert_config["attn_implementation"])
            config.gradient_checkpointing = use_checkpoint
            body = BertEncoder.__new__(BertEncoder)
            nn.Module.__init__(body)
            body.model = HFBertModel(config, add_pooling_layer=add_pooling_layer)
            body.language_dim, body.num_layers_of_embedded = config.hidden_size, num_layers_of_embedded
            self.language_backbone = nn.Sequential(OrderedDict([("body", body)]))
            self.use_sub_sentence_represent = use_sub_sentence_represent
            if use_sub_sentence_represent:
                self.special_tokens = self.tokenizer.convert_tokens_to_ids(special_tokens_list)


class U2CheckpointRuntime(nn.Module):
    """Both supported tasks use this single actual foundation/core object."""

    def __init__(self, runtime: GroundingRuntime, core: SharedTrackCore, payload: dict):
        super().__init__()
        self.foundation = runtime.model
        self.core = core
        self.runtime = runtime
        self.register_buffer("class_text", payload["class_text"]["embeddings"].to(runtime.device), persistent=False)
        self.categories = payload["vocabulary"]["categories"]
        self.metadata = payload["metadata"]
        self.reference_only = payload["reference_only"]

    @torch.inference_mode()
    def forward(self, image_path: str | Path, *, task_type: str,
                frame_index: int, online_state: UnifiedTrackState | None = None,
                vocabulary: str | None = None, max_candidates: int = 150) -> dict:
        if task_type == "ovmot":
            vocabulary = [c["name"] for c in self.categories]
            max_candidates = 300
        elif task_type != "mot" or vocabulary is None:
            raise ValueError("U2 supports MOT with registered vocabulary or full-vocabulary OVMOT")
        candidates = run_u2_candidates(self.runtime, image_path, vocabulary, max_candidates=max_candidates)
        output = self.core(candidates, online_state, task_type=task_type,
                           class_text=self.class_text if task_type == "ovmot" else None,
                           frame_index=frame_index, training=False)
        return {"candidates": candidates, "tracks": output, "online_state": output["state"], "task_type": task_type}


def load_u2_checkpoint(checkpoint: Path, *, device="cuda:0") -> U2CheckpointRuntime:
    """Strict reconstruction using only this one file and installed code."""
    from mmcv.transforms import Compose
    from mmengine.config import Config
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["format"] != FORMAT:
        raise ValueError("Not a native U2 foundation/core checkpoint")
    register_all_modules(init_default_scope=True)
    _register_embedded_language()
    cfg = Config(copy.deepcopy(payload["foundation"]["config"]))
    cfg.model.language_model.update(type="U2EmbeddedBertModel",
                                   bert_config=payload["foundation"]["bert_config"],
                                   tokenizer_files=payload["foundation"]["tokenizer_files"])
    model = MODELS.build(cfg.model)
    model.load_state_dict(payload["foundation"]["state_dict"], strict=True)
    model.cfg = cfg
    runtime_device = torch.device(device)
    model.to(runtime_device).eval().requires_grad_(False)
    runtime = GroundingRuntime(model=model, config=cfg, test_pipeline=Compose(cfg.test_pipeline),
                               checkpoint=Path(checkpoint), checkpoint_sha256=payload["metadata"]["foundation_sha256"],
                               device=runtime_device, missing_keys=(), unexpected_keys=())
    core = SharedTrackCore(**payload["core_config"])
    core.load_state_dict(payload["core"], strict=True)
    core.to(runtime_device).eval().requires_grad_(False)
    return U2CheckpointRuntime(runtime, core, payload).eval()
