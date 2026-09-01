"""Shared streaming GroundingDINO fused-memory utilities for L59.

This module keeps detector outputs in inference mode and returns only the
candidate ROI tensors needed by the small adapter; it never writes feature
cache files.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path('/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT').resolve()
MMDET = Path('/data1/LWR/vranlee/LLM/mmdetection-3.3.0')
CONFIG = MMDET / 'configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.py'
WEIGHT = ROOT.parent / 'TTAOD-F-main/download/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'
BERT = Path('/home/lwr/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594')
IMAGE_ROOT = ROOT.parent / 'KITTI_tracking/training/image_02'
DEVICE = torch.device('cuda:0')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def build_detector():
    from mmengine.config import Config
    from mmengine.runner import load_checkpoint
    from mmdet.registry import MODELS
    import mmdet.models  # noqa: F401
    import mmdet.datasets  # noqa: F401
    from mmdet.utils import register_all_modules
    register_all_modules(init_default_scope=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = Config.fromfile(str(CONFIG))
    cfg.model.backbone.init_cfg = None
    cfg.model.language_model.name = str(BERT)
    model = MODELS.build(cfg.model)
    load = load_checkpoint(model, str(WEIGHT), map_location='cpu', strict=False)
    model.cfg = cfg
    model.to(DEVICE).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError('detector parameter is trainable')
    return model, cfg, load


def detector_provenance(load):
    return {
        'interpreter': '/home/lwr/anaconda3/envs/masaenv_debug/bin/python',
        'config': str(CONFIG), 'config_sha256': sha256(CONFIG),
        'weight': str(WEIGHT), 'weight_sha256': sha256(WEIGHT),
        'bert': str(BERT),
        'load_missing_keys': load.get('missing_keys', []) if isinstance(load, dict) else [],
        'load_unexpected_keys': load.get('unexpected_keys', []) if isinstance(load, dict) else [],
        'warning': 'language_model...position_ids may be reported; local newer checkout has no verified git HEAD',
    }


def _normalize_memory(memory: torch.Tensor) -> torch.Tensor:
    if memory.dim() != 3:
        raise RuntimeError(f'unexpected visual rank {list(memory.shape)}')
    if memory.shape[0] == 1:
        return memory
    if memory.shape[1] == 1:
        return memory.permute(1, 0, 2).contiguous()
    raise RuntimeError(f'ambiguous visual orientation {list(memory.shape)}')


def _roi_sample(memory, shapes, starts, invalid, boxes, image_hw, grid_size=4):
    _, _, dim = memory.shape
    height, width = image_hw
    frac = (torch.arange(grid_size, device=memory.device, dtype=torch.float32) + .5) / grid_size
    tokens = []
    valid = []
    for level, (h0, w0) in enumerate(shapes.tolist()):
        h0, w0, start = int(h0), int(w0), int(starts[level])
        fmap = memory[0, start:start + h0 * w0].reshape(h0, w0, dim).permute(2, 0, 1).unsqueeze(0)
        bad = invalid[0, start:start + h0 * w0].reshape(1, 1, h0, w0).float()
        x1, x2 = boxes[:, 0], boxes[:, 2]
        y1, y2 = boxes[:, 1], boxes[:, 3]
        x = (x1[:, None] + (x2 - x1)[:, None] * frac[None, :])[:, None, :].expand(-1, grid_size, -1)
        y = (y1[:, None] + (y2 - y1)[:, None] * frac[None, :])[:, :, None].expand(-1, -1, grid_size)
        gx = 2 * ((x / width) * w0 + .5) / w0 - 1
        gy = 2 * ((y / height) * h0 + .5) / h0 - 1
        grid = torch.stack([gx, gy], -1)
        feat = F.grid_sample(fmap.expand(len(boxes), -1, -1, -1), grid, mode='bilinear', align_corners=False)
        badv = F.grid_sample(bad.expand(len(boxes), -1, -1, -1), grid, mode='nearest', align_corners=False)[:, 0]
        tokens.append(feat.permute(0, 2, 3, 1).reshape(len(boxes), grid_size * grid_size, dim))
        # Count all grid samples per candidate; badv is [N, grid, grid].
        valid.append((badv < .5).sum((-1, -2)))
    return torch.cat(tokens, 1), torch.stack(valid, 1)


def stream_fused_roi(detector, unit: dict, bank: dict, sentence: str | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    from mmdet.apis import inference_detector

    begin, end = int(unit['begin']), int(unit['end'])
    tensors = bank['tensors']
    candidates = tensors['box'][begin:end].float().to(DEVICE)
    numeric = torch.cat([
        tensors['geometry'][begin:end].float(),
        tensors['motion'][begin:end].float(),
        tensors['lifecycle'][begin:end].float(),
        tensors['objectness'][begin:end].float().reshape(-1, 1),
    ], 1).to(DEVICE)
    captured = {}

    def encoder_hook(module, args, kwargs, result):
        if not isinstance(result, (tuple, list)) or len(result) < 2:
            raise RuntimeError('encoder output is not a pair')
        captured['encoder_visual_none'] = result[0] is None
        captured['encoder_text_none'] = result[1] is None
        for name in ('spatial_shapes', 'level_start_index', 'key_padding_mask', 'text_attention_mask'):
            value = kwargs.get(name)
            captured[name] = None if value is None else value.detach()

    def fusion_hook(module, args, result):
        if not isinstance(result, (tuple, list)) or len(result) < 2 or result[0] is None or result[1] is None:
            raise RuntimeError('final fusion output is not (visual,text)')
        captured['visual'] = result[0].detach()
        captured['text'] = result[1].detach()

    enc_handle = detector.encoder.register_forward_hook(encoder_hook, with_kwargs=True)
    fusion_handle = detector.encoder.fusion_layers[-1].register_forward_hook(fusion_hook)
    image_path = IMAGE_ROOT / str(unit['video']) / f"{int(unit['frame_id']):06d}.png"
    start = time.time()
    with torch.inference_mode():
        native = inference_detector(detector, str(image_path), text_prompt=sentence or unit['sentence'], custom_entities=True)
    elapsed = time.time() - start
    enc_handle.remove(); fusion_handle.remove()
    for key in ('visual', 'text', 'spatial_shapes', 'level_start_index'):
        if key not in captured or captured[key] is None:
            raise RuntimeError(f'missing fused capture {key}')
    visual = _normalize_memory(captured['visual'])
    text = captured['text']
    shapes = captured['spatial_shapes']; starts = captured['level_start_index']
    if shapes.dim() == 3: shapes = shapes[0]
    if starts.dim() > 1: starts = starts[0]
    if shapes.dim() != 2 or shapes.shape[-1] != 2:
        raise RuntimeError(f'bad spatial_shapes {list(shapes.shape)}')
    total = int((shapes[:, 0] * shapes[:, 1]).sum())
    if visual.shape[1] != total or starts.numel() != shapes.shape[0]:
        raise RuntimeError('visual/spatial metadata mismatch')
    invalid = captured['key_padding_mask']
    if invalid is None:
        invalid = torch.zeros((1, total), device=visual.device, dtype=torch.bool)
        key_padding_none = True
    else:
        key_padding_none = False
        if invalid.dim() == 2 and invalid.shape[0] != 1: invalid = invalid[:1]
    text_mask = captured['text_attention_mask']
    if text_mask is None:
        text_valid = torch.ones(text.shape[:2], device=text.device, dtype=torch.bool)
        text_mask_none = True
    else:
        if text_mask.dim() == 1: text_mask = text_mask.unsqueeze(0)
        text_valid = ~text_mask.bool()
        text_mask_none = False
    scale = torch.as_tensor(native.metainfo['scale_factor'], device=DEVICE, dtype=torch.float32).flatten()
    scale_xyxy = scale.repeat(2) if scale.numel() == 2 else scale
    boxes_resized = candidates * scale_xyxy
    if not (torch.isfinite(visual).all() and torch.isfinite(text).all() and torch.isfinite(boxes_resized).all()):
        raise RuntimeError('nonfinite fused ROI input')
    roi, valid_counts = _roi_sample(visual, shapes, starts, invalid, boxes_resized,
                                     (int(native.metainfo['img_shape'][0]), int(native.metainfo['img_shape'][1])))
    if not torch.isfinite(roi).all(): raise RuntimeError('nonfinite ROI tokens')
    bank_path = str(Path(unit['bank_path']).resolve())
    row_keys = [(str(unit['video']), int(unit['frame_id']), bank_path, begin + i) for i in range(end - begin)]
    meta = {
        'candidate_count': end - begin, 'candidate_keys': row_keys,
        'roi_tokens_shape': list(roi.shape), 'text_shape': list(text.shape),
        'text_valid_count': int(text_valid.sum()), 'roi_valid_fraction': float((valid_counts > 0).float().mean()),
        'roi_valid_counts_per_candidate_level': valid_counts.detach().cpu().tolist(),
        'roi_valid_fraction_per_level': (valid_counts.float().mean(0) / 16.0).detach().cpu().tolist(),
        'roi_total_samples': int(valid_counts.numel() * 16),
        'roi_valid_samples': int(valid_counts.sum()),
        'zero_padding_mask': key_padding_none, 'text_attention_mask_missing': text_mask_none,
        'spatial_shapes': shapes.cpu().tolist(), 'level_start_index': starts.cpu().tolist(),
        'scale_factor': scale.detach().cpu().tolist(), 'img_shape': list(native.metainfo['img_shape']),
        'ori_shape': list(native.metainfo['ori_shape']), 'forward_sec': elapsed,
        'native_postprocessed_count': int(len(native.pred_instances)), 'all_rows_retained': True,
        'candidate_truncation': False, 'representation_finite': True,
    }
    return roi.float().clone(), text.float().clone(), text_valid.clone(), numeric.float().clone(), meta
