"""Small CPU-only L51 input-contract regression; no dataset or checkpoint read."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from locatemot.models.l51_streaming_crop_adapter import L51StreamingCropAdapter


OUT = ROOT / "outputs/l51/audit/b0_token_mask_invariance.json"


def main() -> None:
    torch.manual_seed(20260829)
    n, p, t = 11, 4, 6
    patch = torch.randn(n, p, 768)
    text = torch.randn(t, 768)
    mask = torch.tensor([True, True, True, False, False, False])
    altered = text.clone()
    altered[~mask] = torch.randn_like(altered[~mask]) * 1000.0
    frozen = torch.randn(n, 512)
    numeric = torch.randn(n, 36)
    teacher = torch.linspace(-1.0, 1.0, n)
    model = L51StreamingCropAdapter(hidden=128, heads=4, layers=2).eval()
    with torch.no_grad():
        first = model(patch, text, mask, frozen, numeric, teacher)
        second = model(patch, altered, mask, frozen, numeric, teacher)
    mask_diff = float((first["final_logit"] - second["final_logit"]).abs().max())
    initial_diff = float((first["final_logit"] - teacher).abs().max())
    residual_max = float(torch.cat((first["residual"], second["residual"])).abs().max())
    payload = {
        "format": "locatemot-l51-token-mask-regression-v1",
        "status": "pass" if mask_diff == 0.0 and initial_diff == 0.0 and residual_max <= 0.05 else "fail",
        "seed": 20260829,
        "candidate_count": n,
        "candidate_count_preserved": True,
        "candidate_truncation": False,
        "token_count": t,
        "valid_token_count": int(mask.sum()),
        "masked_padding_tokens_altered": int((~mask).sum()),
        "masked_token_final_logit_max_abs_diff": mask_diff,
        "initial_final_vs_teacher_max_abs_diff": initial_diff,
        "initial_residual_max_abs": float(first["residual"].abs().max()),
        "residual_max_abs_checked": residual_max,
        "residual_bound": 0.05,
        "mask_contract": "masked padding changes do not affect final_logit; masked mean is used after cross-attention",
        "dataset_or_gt_read": False,
        "screening_or_test_read": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
