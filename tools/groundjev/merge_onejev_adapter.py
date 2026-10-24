"""Merge a completed OneJev LoRA adapter into a standalone inference model.

This is a post-training step.  It refuses a non-empty output directory and
never touches the base checkpoint or the adapter directory.  The resulting
directory is the format consumed by ``NativeOneJevAdapter``/``MMDecisionEngine``
for the real held-out TrackJEV evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge(*, base: Path, adapter: Path, out: Path, dtype: str = "bfloat16") -> dict[str, Any]:
    if not base.is_dir() or not (base / "config.json").is_file():
        raise FileNotFoundError(f"base checkpoint is not a complete model directory: {base}")
    if not adapter.is_dir() or not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"adapter is not a PEFT directory: {adapter}")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty merge output: {out}")
    out.mkdir(parents=True, exist_ok=True)

    import torch
    import transformers
    from peft import PeftModel

    common = {"local_files_only": True, "trust_remote_code": False}
    config = transformers.AutoConfig.from_pretrained(str(base), **common)
    processor = transformers.AutoProcessor.from_pretrained(str(base), **common)
    model_cls = (transformers.Qwen3_5ForConditionalGeneration
                 if config.model_type == "qwen3_5"
                 else transformers.AutoModelForImageTextToText)
    model = model_cls.from_pretrained(
        str(base), config=config, dtype=getattr(torch, dtype),
        device_map={"": "cpu"}, low_cpu_mem_usage=True, **common)
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False,
                                      local_files_only=True)
    merged = model.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(str(out), safe_serialization=True)
    processor.save_pretrained(str(out))

    receipt = {
        "status": "ONEJEV_LORA_MERGE_COMPLETE",
        "base_checkpoint": str(base.resolve()),
        "adapter": str(adapter.resolve()),
        "output": str(out.resolve()),
        "base_config_sha256": sha256_file(base / "config.json"),
        "adapter_config_sha256": sha256_file(adapter / "adapter_config.json"),
        "model_type": config.model_type,
        "dtype": dtype,
        "transformers_version": transformers.__version__,
        "peft_version": __import__("peft").__version__,
        "base_and_adapter_preserved": True,
        "output_load_contract": "qev.mm_engine.MMDecisionEngine local multimodal checkpoint",
    }
    (out / "merge_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    args = parser.parse_args()
    print(json.dumps(merge(base=args.base.resolve(), adapter=args.adapter.resolve(),
                           out=args.out.resolve(), dtype=args.dtype), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
