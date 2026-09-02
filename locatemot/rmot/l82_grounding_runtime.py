"""Private L82 runtime for the frozen GroundingDINO candidate-reference probe.

The verified MMDetection runtime has a Torch 2.1/CUDA 12.1 installation whose
torchvision extension was compiled for CUDA 11.8.  OpenAI CLIP itself only
needs torchvision transforms at import time, so L82 installs a small in-process
transform implementation before importing the pure-Python CLIP package.  No
environment package, model weight, or third-party checkout is modified.
"""
from __future__ import annotations

import copy
import importlib.machinery
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
LOCAL_MMDET = Path("/data1/LWR/vranlee/LLM/mmdetection-3.3.0").resolve()
CONFIG = LOCAL_MMDET / "configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.py"
WEIGHT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TTAOD-F-main/download/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth").resolve()
BERT = Path("/home/lwr/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594").resolve()
CLIP_SOURCE = Path("/data1/LWR/vranlee/LLM/CLIP").resolve()


def install_clip_torchvision_compat() -> dict[str, Any]:
    """Install only the transforms API required by the local OpenAI CLIP.

    The function is deliberately called before importing either ``clip`` or
    MMDetection.  The shim has a valid module spec so Transformers/MMEngine's
    optional-package checks remain well behaved.  The real CLIP model code is
    still the locally checked-out OpenAI implementation and the fixed local
    checkpoint is loaded by the existing L80 loader.
    """
    existing = sys.modules.get("clip")
    if existing is not None and str(getattr(existing, "__file__", "")).startswith(str(CLIP_SOURCE)):
        return {"installed": False, "reason": "already_imported_local_clip", "source": str(CLIP_SOURCE)}
    for name in ("clip", "clip.clip", "clip.model", "clip.simple_tokenizer"):
        sys.modules.pop(name, None)

    transforms = __import__("types").ModuleType("torchvision.transforms")
    transforms.__spec__ = importlib.machinery.ModuleSpec("torchvision.transforms", loader=None)

    class Compose:
        def __init__(self, items: Iterable[Any]) -> None:
            self.transforms = list(items)

        def __call__(self, value: Any) -> Any:
            for item in self.transforms:
                value = item(value)
            return value

    class Resize:
        def __init__(self, size: int | tuple[int, int], interpolation: Any = Image.BICUBIC) -> None:
            self.size = size
            self.interpolation = interpolation

        def __call__(self, value: Image.Image) -> Image.Image:
            if isinstance(self.size, int):
                width, height = value.size
                if width <= height:
                    new_width = self.size
                    new_height = int(round(height * self.size / max(width, 1)))
                else:
                    new_height = self.size
                    new_width = int(round(width * self.size / max(height, 1)))
                return value.resize((new_width, new_height), self.interpolation)
            return value.resize(tuple(self.size), self.interpolation)

    class CenterCrop:
        def __init__(self, size: int | tuple[int, int]) -> None:
            self.size = (size, size) if isinstance(size, int) else tuple(size)

        def __call__(self, value: Image.Image) -> Image.Image:
            crop_w, crop_h = self.size
            width, height = value.size
            left = int(round((width - crop_w) / 2.0))
            top = int(round((height - crop_h) / 2.0))
            if crop_w > width or crop_h > height:
                canvas = Image.new(value.mode, (max(width, crop_w), max(height, crop_h)))
                canvas.paste(value, ((canvas.size[0] - width) // 2, (canvas.size[1] - height) // 2))
                value = canvas
                width, height = value.size
                left = int(round((width - crop_w) / 2.0))
                top = int(round((height - crop_h) / 2.0))
            return value.crop((left, top, left + crop_w, top + crop_h))

    class ToTensor:
        def __call__(self, value: Image.Image) -> torch.Tensor:
            array = np.asarray(value.convert("RGB"), dtype=np.float32) / 255.0
            return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    class Normalize:
        def __init__(self, mean: Iterable[float], std: Iterable[float]) -> None:
            self.mean = torch.tensor(tuple(mean), dtype=torch.float32)[:, None, None]
            self.std = torch.tensor(tuple(std), dtype=torch.float32)[:, None, None]

        def __call__(self, value: torch.Tensor) -> torch.Tensor:
            return (value - self.mean.to(value)) / self.std.to(value)

    transforms.Compose = Compose
    transforms.Resize = Resize
    transforms.CenterCrop = CenterCrop
    transforms.ToTensor = ToTensor
    transforms.Normalize = Normalize

    torchvision = __import__("types").ModuleType("torchvision")
    torchvision.__spec__ = importlib.machinery.ModuleSpec("torchvision", loader=None)
    torchvision.__path__ = []
    torchvision.__version__ = "0.20.1-l82-transforms-only"
    torchvision.transforms = transforms
    sys.modules["torchvision"] = torchvision
    sys.modules["torchvision.transforms"] = transforms
    if str(CLIP_SOURCE) not in sys.path:
        sys.path.insert(0, str(CLIP_SOURCE))
    else:
        sys.path.remove(str(CLIP_SOURCE))
        sys.path.insert(0, str(CLIP_SOURCE))
    import clip  # noqa: F401
    return {
        "installed": True,
        "source": str(CLIP_SOURCE),
        "torchvision_contract": "in-process transforms-only shim; no compiled torchvision ops",
        "clip_file": str(Path(clip.__file__).resolve()),
    }


def build_groundingdino(device: torch.device) -> tuple[Any, dict[str, Any]]:
    """Build the frozen local MMDetection model on the caller's local GPU."""
    from mmengine.config import Config
    from mmengine.runner import load_checkpoint
    from mmdet.registry import MODELS
    import mmdet.datasets  # noqa: F401
    import mmdet.models  # noqa: F401
    from mmdet.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(CONFIG))
    cfg.model.backbone.init_cfg = None
    cfg.model.language_model.name = str(BERT)
    model = MODELS.build(cfg.model)
    loaded = load_checkpoint(model, str(WEIGHT), map_location="cpu", strict=False)
    model.to(device).eval()
    model.cfg = cfg
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    missing = loaded.get("missing_keys", []) if isinstance(loaded, dict) else []
    unexpected = loaded.get("unexpected_keys", []) if isinstance(loaded, dict) else []
    return model, {
        "config": str(CONFIG),
        "weight": str(WEIGHT),
        "bert": str(BERT),
        "device": str(device),
        "checkpoint_missing_keys": list(missing),
        "checkpoint_unexpected_keys": list(unexpected),
        "parameters_frozen": all(not p.requires_grad for p in model.parameters()),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
    }


class GroundingCandidateReferenceRuntime:
    """One native visual pass plus expression replays for one frame group."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.model, self.model_info = build_groundingdino(device)
        from mmdet.apis import inference_detector
        from tools.l82_audit_grounding_interface import (
            capture_candidate_state, get_arg, make_text_dict, set_sample_text,
        )

        self.inference_detector = inference_detector
        self.capture_candidate_state = capture_candidate_state
        self.get_arg = get_arg
        self.make_text_dict = make_text_dict
        self.set_sample_text = set_sample_text
        self.encoder_events: list[dict[str, Any]] = []
        self.capture: dict[str, Any] = {}

        def encoder_hook(module: Any, hook_args: tuple[Any, ...], hook_kwargs: dict[str, Any], output: Any) -> None:
            if not isinstance(output, (tuple, list)) or len(output) != 2:
                raise AssertionError("L82 encoder output is not (visual_memory,memory_text)")
            key_padding = self.get_arg(hook_args, hook_kwargs, "key_padding_mask", 2)
            spatial = self.get_arg(hook_args, hook_kwargs, "spatial_shapes", 3)
            starts = self.get_arg(hook_args, hook_kwargs, "level_start_index", 4)
            valid = self.get_arg(hook_args, hook_kwargs, "valid_ratios", 5)
            text_attention = self.get_arg(hook_args, hook_kwargs, "text_attention_mask", 7)
            text_mask = None if text_attention is None else ~text_attention.bool()
            self.encoder_events.append({
                "memory": output[0].detach().clone(),
                "memory_text": output[1].detach().clone(),
                "memory_mask": key_padding.detach().clone() if torch.is_tensor(key_padding) else key_padding,
                "spatial_shapes": spatial.detach().clone() if torch.is_tensor(spatial) else spatial,
                "level_start_index": starts.detach().clone() if torch.is_tensor(starts) else starts,
                "valid_ratios": valid.detach().clone() if torch.is_tensor(valid) else valid,
                "text_attention_mask": text_attention.detach().clone() if torch.is_tensor(text_attention) else text_attention,
                "text_token_mask": text_mask.detach().clone() if torch.is_tensor(text_mask) else text_mask,
            })

        self.encoder_handle = self.model.encoder.register_forward_hook(encoder_hook, with_kwargs=True)
        original_extract = self.model.extract_feat
        original_forward_transformer = self.model.forward_transformer

        def wrapped_extract(batch_inputs: Any) -> Any:
            features = original_extract(batch_inputs)
            self.capture["visual_feats"] = tuple(x.detach().clone() for x in features)
            return features

        def wrapped_forward_transformer(img_feats: Any, text_dict: dict[str, Any], batch_data_samples: Any = None) -> Any:
            self.capture["sample_template"] = copy.deepcopy(batch_data_samples)
            return original_forward_transformer(img_feats, text_dict, batch_data_samples)

        self.model.extract_feat = wrapped_extract
        self.model.forward_transformer = wrapped_forward_transformer
        self.original_forward_transformer = original_forward_transformer
        self.native_visual_forward_count = 0
        self.replay_count = 0

    @staticmethod
    def _finite(value: torch.Tensor, name: str) -> None:
        if not bool(torch.isfinite(value.float()).all()):
            raise FloatingPointError(f"nonfinite {name}")

    def extract_group(self, batches: list[Any]) -> dict[str, Any]:
        if not batches:
            raise ValueError("empty L82 frame group")
        first = batches[0]
        image = Path(first.image_path)
        expression = str(first.sentence)
        self.encoder_events.clear()
        self.capture.clear()
        started = time.perf_counter()
        with torch.inference_mode():
            native = self.inference_detector(
                self.model, str(image), text_prompt=expression, custom_entities=True)
        native_seconds = time.perf_counter() - started
        if len(self.encoder_events) != 1:
            raise AssertionError(f"expected one native encoder event, got {len(self.encoder_events)}")
        visual_feats = self.capture.get("visual_feats")
        sample_template = self.capture.get("sample_template")
        if visual_feats is None or not isinstance(sample_template, (list, tuple)) or len(sample_template) != 1:
            raise AssertionError("native path did not expose reusable visual features/sample")
        image_shape = tuple(int(x) for x in native.metainfo["img_shape"][:2])
        scale_factor = native.metainfo["scale_factor"]
        if tuple(int(x) for x in first.boxes.shape) != (first.candidate_count, 4):
            raise AssertionError("L82 candidate box shape drift")
        self.native_visual_forward_count += 1
        sample_template = sample_template[0]
        outputs = []
        replay_seconds = 0.0
        for batch in batches:
            text_dict, caption, token_map = self.make_text_dict(
                self.model, str(batch.sentence), self.device, force_pad_to_max=True)
            sample = copy.deepcopy(sample_template)
            self.set_sample_text(sample, caption, token_map)
            before = len(self.encoder_events)
            t0 = time.perf_counter()
            with torch.inference_mode():
                self.original_forward_transformer(visual_feats, text_dict, [sample])
            replay_seconds += time.perf_counter() - t0
            self.replay_count += 1
            if len(self.encoder_events) != before + 1:
                raise AssertionError(f"encoder replay event drift for {batch.unit_key}")
            event = self.encoder_events[-1]
            state = self.capture_candidate_state(
                self.model, event, batch.boxes.to(self.device), image_shape, scale_factor)
            visual_seed = state["visual_seed"].float().detach().cpu().contiguous()
            final_hidden = state["final_hidden"].float().detach().cpu().contiguous()
            self._finite(visual_seed, "L59 visual seed")
            self._finite(final_hidden, "L82 final hidden")
            if visual_seed.shape != (batch.candidate_count, 256):
                raise AssertionError(f"L59 seed shape drift {tuple(visual_seed.shape)}")
            if final_hidden.shape != (batch.candidate_count, 256):
                raise AssertionError(f"L82 hidden shape drift {tuple(final_hidden.shape)}")
            outputs.append({
                "visual_seed": visual_seed,
                "candidate_reference": final_hidden,
                "candidate_reference_event": {
                    "memory_shape": list(event["memory"].shape),
                    "memory_text_shape": list(event["memory_text"].shape),
                    "spatial_shapes": event["spatial_shapes"].cpu().tolist(),
                    "level_start_index": event["level_start_index"].cpu().tolist(),
                    "memory_mask_supplied": event["memory_mask"] is not None,
                },
            })
            del state, event, text_dict, sample
            self.encoder_events.clear()
        # visual_feats are process-local only and must not survive the frame.
        del visual_feats, sample_template, native
        self.capture.clear()
        return {
            "outputs": outputs,
            "native_seconds": float(native_seconds),
            "replay_seconds": float(replay_seconds),
            "native_image_shape": list(image_shape),
            "native_scale_factor": np.asarray(scale_factor).reshape(-1).tolist(),
            "candidate_count": int(first.candidate_count),
            "image_path": str(image.resolve()),
        }

    def close(self) -> None:
        self.encoder_handle.remove()
        self.encoder_events.clear()
        self.capture.clear()
        del self.model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


__all__ = [
    "BERT", "CLIP_SOURCE", "CONFIG", "GroundingCandidateReferenceRuntime",
    "LOCAL_MMDET", "ROOT", "WEIGHT", "build_groundingdino",
    "install_clip_torchvision_compat",
]
