#!/usr/bin/env python3
"""L72-A0: one real, label-free LocateAnything API smoke.

The detector is loaded only from the checked-out local model directory.  This
script intentionally exercises the public processor + custom ``generate``
entrypoint and ``extract_feature`` interface; it does not read labels, build a
bank, or persist image/features.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
MODEL_DIR = ROOT / "models/LocateAnything-3B"
IMAGE_ROOT = ROOT / "data/kitti_tracking_training/image_02"
SEED = 20260829


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def first_fit_job() -> dict[str, Any]:
    """Read only the non-label fields needed to select one fit image."""
    path = ROOT / "outputs/l49/data/train_units.jsonl"
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("dataset") not in {"refer_kitti_v1", "refer_kitti_v2"}:
                continue
            if row.get("split") != "fit":
                continue
            sentence = str(row.get("sentence") or row.get("expression") or "")
            image = IMAGE_ROOT / str(row["video"]) / f"{int(row['frame_id']):06d}.png"
            if sentence and image.exists():
                return {
                    "dataset": str(row["dataset"]),
                    "video": str(row["video"]),
                    "query_id": int(row["query_id"]),
                    "frame_id": int(row["frame_id"]),
                    "sentence": sentence,
                    "image_path": str(image),
                    "source_manifest": str(path),
                    "labels_used": False,
                }
    raise FileNotFoundError("no real fit image/expression pair found")


def model_manifest() -> dict[str, Any]:
    files: dict[str, Any] = {}
    for path in sorted(MODEL_DIR.iterdir()):
        if not path.is_file():
            continue
        if path.suffix in {".json", ".py", ".txt", ".md", ".yaml", ".yml"} or path.name.endswith(".safetensors"):
            files[path.name] = {
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/l72/audit/api_smoke_attempt1",
    )
    parser.add_argument("--max-new-tokens", type=int, default=6)
    args = parser.parse_args()
    out = args.out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    base = {
        "format": "locatemot-l72-raw-la-api-smoke-v1",
        "status": "running",
        "project_root": str(ROOT),
        "cwd": os.getcwd(),
        "command": " ".join(sys.argv),
        "seed": SEED,
        "interpreter": sys.executable,
        "runtime_flags": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        },
        "model_path": str(MODEL_DIR),
        "model_manifest": {},
        "outputs": {
            "environment": str(out / "environment.json"),
            "api_contract": str(out / "api_contract.json"),
        },
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False,
        "training_run": False,
        "raw_dense_feature_cache_written": False,
        "hota_trackeval_run": False,
    }
    write_json(out / "status.json", base)
    started = time.perf_counter()
    try:
        if ROOT.resolve() != Path.cwd().resolve():
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if not MODEL_DIR.is_dir():
            raise FileNotFoundError(MODEL_DIR)
        job = first_fit_job()
        base["job"] = job
        base["model_manifest"] = model_manifest()

        import torch
        import transformers
        from PIL import Image
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        torch.manual_seed(SEED)
        if not torch.cuda.is_available():
            raise RuntimeError("L72 requires the verified CUDA runtime; CUDA is unavailable")
        device = torch.device("cuda:0")
        model_dtype = torch.bfloat16
        tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_DIR), trust_remote_code=True, local_files_only=True
        )
        processor = AutoProcessor.from_pretrained(
            str(MODEL_DIR), trust_remote_code=True, local_files_only=True
        )
        model = AutoModel.from_pretrained(
            str(MODEL_DIR),
            torch_dtype=model_dtype,
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="sdpa",
        ).to(device).eval()
        # ``eval()`` controls module behavior but does not freeze parameters;
        # the L72 detector is explicitly inference-only.
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        detector_frozen = all(not parameter.requires_grad for parameter in model.parameters())
        if not detector_frozen:
            raise AssertionError("LocateAnything parameter unexpectedly requires gradients")

        image = Image.open(job["image_path"]).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": job["sentence"]},
            ],
        }]
        prompt = processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = processor.process_vision_info(messages)
        inputs = processor(
            text=[prompt], images=images, videos=videos, return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        image_grid_hws = torch.as_tensor(
            inputs["image_grid_hws"], dtype=torch.int32, device=device
        )
        pixel_values = inputs["pixel_values"].to(device=device, dtype=model_dtype)
        if image_grid_hws.ndim != 2 or image_grid_hws.shape[-1] != 2:
            raise AssertionError(f"unexpected image_grid_hws shape {tuple(image_grid_hws.shape)}")

        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            raw_features = model.extract_feature(pixel_values, image_grid_hws)
            response = model.generate(
                pixel_values=pixel_values,
                visual_features=raw_features,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_grid_hws=image_grid_hws,
                tokenizer=tokenizer,
                max_new_tokens=int(args.max_new_tokens),
                use_cache=True,
                generation_mode="fast",
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                verbose=False,
            )
        if isinstance(raw_features, (tuple, list)):
            feature_shapes = [list(value.shape) for value in raw_features]
            feature_finite = [bool(torch.isfinite(value.float()).all()) for value in raw_features]
        else:
            feature_shapes = [list(raw_features.shape)]
            feature_finite = [bool(torch.isfinite(raw_features.float()).all())]
        elapsed = time.perf_counter() - t0
        peak = int(torch.cuda.max_memory_allocated(device))
        response_type = type(response).__name__
        answer = response[0] if isinstance(response, tuple) else response
        if not isinstance(answer, str):
            raise AssertionError(f"expected text generation result, got {response_type}")
        payload = {
            **base,
            "status": "complete",
            "format": "locatemot-l72-raw-la-api-smoke-v1",
            "packages": {
                "torch": torch.__version__,
                "torchvision": __import__("torchvision").__version__,
                "transformers": transformers.__version__,
                "model_class": type(model).__name__,
                "tokenizer_class": type(tokenizer).__name__,
                "processor_class": type(processor).__name__,
            },
            "device": str(device),
            "model_dtype": str(model_dtype),
            "image": {
                "path": str(job["image_path"]),
                "width": int(image.width),
                "height": int(image.height),
            },
            "inputs": {
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "input_ids_shape": list(input_ids.shape),
                "attention_mask_shape": list(attention_mask.shape) if attention_mask is not None else None,
                "image_grid_hws": image_grid_hws.cpu().tolist(),
                "pixel_values_shape": list(pixel_values.shape),
                "pixel_values_finite": bool(torch.isfinite(pixel_values.float()).all()),
            },
            "raw_vision_features": {
                "shapes": feature_shapes,
                "finite": feature_finite,
                "query_conditioned": False,
            },
            "generation": {
                "return_type": response_type,
                "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
                "answer_chars": len(answer),
                "generation_mode": "fast",
                "max_new_tokens": int(args.max_new_tokens),
            },
            "runtime": {
                "elapsed_seconds": elapsed,
                "peak_cuda_bytes": peak,
            },
            "contract": {
                "real_image": True,
                "original_expression_used": True,
                "labels_used": False,
                "model_parameters_frozen": detector_frozen,
                "persistent_feature_cache_written": False,
                "direct_model_forward_used": False,
                "generation_is_text": True,
            },
        }
        write_json(out / "environment.json", payload)
        write_json(out / "api_contract.json", payload)
        write_json(out / "status.json", payload)
        del raw_features, inputs, model, processor, tokenizer
        torch.cuda.empty_cache()
        return 0
    except Exception as exc:
        failure = {
            **base,
            "status": "incomplete",
            "failure_root_cause": f"{type(exc).__name__}: {exc}",
            "next_action": "inspect the first API/runtime error and perform at most one minimal targeted retry",
            "elapsed_seconds": time.perf_counter() - started,
            "traceback": traceback.format_exc(),
        }
        write_json(out / "status.json", failure)
        (out / "INCOMPLETE.md").write_text(
            "# INCOMPLETE\n\n"
            f"First actionable root cause: `{failure['failure_root_cause']}`\n\n"
            "The attempt is preserved. No zero-vector or remote-weight fallback was used.\n"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
