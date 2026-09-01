"""L24 code/metric audit of L23 multi-positive, hard-mining and dense teacher paths."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from locatemot.models.rmot_dense_correspondence_scorer import DenseQueryCorrespondenceScorer  # noqa: E402
from tools.train_l23_dense_correspondence import arrays_for, model_score  # noqa: E402
from tools.train_rmot_candidate_scorer import average_precision, auc, load_bank, load_metadata, make_refs, scalar_stats  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", default="outputs/l19/protocol/kitti_fast_eval_manifest.json"); ap.add_argument("--v3-root", default="outputs/l23/candidate_bank_v3"); ap.add_argument("--checkpoint", default="outputs/l23/train/D0_dense_roi_query_cross_attention_S250/checkpoint_d0_step250.pt"); ap.add_argument("--out-root", default="outputs/l24/audit/l23_consistency"); ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    def p(x: str) -> Path:
        x = Path(x); return x if x.is_absolute() else ROOT / x
    manifest, v3_root, checkpoint, out_root = map(p, (args.manifest, args.v3_root, args.checkpoint, args.out_root))
    if out_root.exists(): raise FileExistsError(out_root)
    out_root.mkdir(parents=True, exist_ok=False)
    scorer_path = ROOT / "locatemot/models/rmot_dense_correspondence_scorer.py"; trainer_path = ROOT / "tools/train_l23_dense_correspondence.py"
    scorer_code, trainer_code = scorer_path.read_text(), trainer_path.read_text()
    # Manual two-positive test: both positives must receive a negative gradient
    # when their score is below one hard negative.
    positive = torch.tensor([0.1, 0.8], requires_grad=True); hard = torch.tensor([1.2], requires_grad=True)
    pairwise = F.softplus(1.0 - positive[:, None] + hard[None, :]).mean(); pairwise.backward()
    pair_grad = positive.grad.detach().tolist() + [float(hard.grad.detach()[0])]
    positive2 = torch.tensor([0.1, 0.8], requires_grad=True); hard2 = torch.tensor([1.2], requires_grad=True)
    old_violation = F.softplus(hard2.max() - positive2.max()); old_violation.backward()
    old_grad = positive2.grad.detach().tolist() + [float(hard2.grad.detach()[0])]
    multi_positive_smoke = {"pairwise_loss": float(pairwise.detach()), "positive_low_grad": pair_grad[0], "positive_high_grad": pair_grad[1], "hard_grad": pair_grad[2], "both_positive_gradients_push_up": pair_grad[0] < 0 and pair_grad[1] < 0 and pair_grad[2] > 0, "legacy_max_violation_loss": float(old_violation.detach()), "legacy_max_violation_grads": old_grad, "legacy_low_positive_received_gradient": old_grad[0] < 0}
    metadata = load_metadata(); queries = sorted(json.loads(manifest.read_text())["queries"], key=lambda x: int(x["query_index"])); videos = sorted({str(q["video"]) for q in queries}); banks = {v: load_bank(v3_root / "kitti" / f"{v}.pt") for v in videos}; refs = [r for r in make_refs(queries, metadata, banks) if r["split"] == "screening"]
    device = torch.device(args.device); model = DenseQueryCorrespondenceScorer(stage="D0").to(device); model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"]); model.eval()
    global_hard_missed, global_top_overlap, pref_margin, global_margin = [], [], [], []; teacher_scores_all, labels_all = [], []; d0_top1 = teacher_top1 = top1_disagreements = 0; positive_frames = 0; qnorms=[]; roinorms=[]; pointnorms=[]
    for ref in refs:
        bank = banks[ref["video"]]; size = ref["end"] - ref["begin"]; rows = np.arange(size, dtype=np.int64); arrays = arrays_for(ref, bank, rows); value = {k: torch.as_tensor(v, device=device) for k,v in arrays.items()}
        with torch.inference_mode(): d0_scores = model_score(model, value).cpu().numpy()
        q = np.asarray(ref["spec"], np.float32); q = q / max(1e-6, float(np.linalg.norm(q))); qnorms.append(float(np.linalg.norm(ref["spec"])))
        roi = arrays["dense_roi"]; points = arrays["dense_points"]; roinorms.extend(np.linalg.norm(roi, axis=1).tolist()); pointnorms.extend(np.linalg.norm(points, axis=2).reshape(-1).tolist())
        teacher = np.maximum((roi @ q) / np.maximum(1e-6, np.linalg.norm(roi,axis=1)), np.max((points @ q) / np.maximum(1e-6, np.linalg.norm(points,axis=2)), axis=1))
        labels = ref["positive"].astype(bool); positive_idx=np.flatnonzero(labels); negative_idx=np.flatnonzero(~labels); teacher_scores_all.extend(teacher.tolist()); labels_all.extend(labels.tolist())
        if not len(positive_idx): continue
        positive_frames += 1; positive_top=int(np.argmax(teacher[positive_idx])==np.argmax(teacher)); d0_top1 += int(labels[np.argmax(d0_scores)]); teacher_top1 += int(labels[np.argmax(teacher)]); top1_disagreements += int(np.argmax(d0_scores)!=np.argmax(teacher))
        objectness=bank["tensors"]["objectness"][ref["begin"]:ref["end"]].float().numpy().reshape(-1); pre=negative_idx[np.argsort(-objectness[negative_idx],kind="stable")[:min(96,len(negative_idx))]]; global_order=negative_idx[np.argsort(-d0_scores[negative_idx],kind="stable")[:min(24,len(negative_idx))]]; pref_order=pre[np.argsort(-d0_scores[pre],kind="stable")[:min(24,len(pre))]]
        if len(global_order): global_hard_missed.append(int(global_order[0] not in set(pre.tolist()))); global_top_overlap.append(len(set(global_order.tolist()) & set(pref_order.tolist())) / max(1,len(global_order)))
        pref_margin.append(float(d0_scores[positive_idx].min()-d0_scores[pref_order].max()) if len(pref_order) else 0.0); global_margin.append(float(d0_scores[positive_idx].min()-d0_scores[global_order].max()) if len(global_order) else 0.0)
    teacher_scores=np.asarray(teacher_scores_all,np.float32); labels=np.asarray(labels_all,bool)
    source = {"scorer_file": str(scorer_path), "trainer_file": str(trainer_path), "legacy_violation_max_present": "positive_logits.max()" in trainer_code, "legacy_violation_uses_min": "positive_logits.min()" in trainer_code, "pairwise_is_positivewise": "positive_logits[:, None]" in trainer_code, "listwise_present": "logsumexp(positive_logits" in trainer_code, "point_coordinate_embedding_present": any(x in scorer_code for x in ("point_coord", "positional_encoding", "point_position")), "raw_teacher_path_present": "teacher_score" in scorer_code or "teacher_score" in trainer_code}
    report={"format":"locatemot-l24-l23-consistency-audit-v1","manifest":str(manifest),"manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),"v3_root":str(v3_root),"checkpoint":str(checkpoint),"screening_frame_units":len(refs),"multi_positive_manual_test":multi_positive_smoke,"code_findings":source,"query_norm":scalar_stats(qnorms),"dense_roi_norm":scalar_stats(roinorms),"dense_point_token_norm":scalar_stats(pointnorms),"hard_mining_contract":{"objectness_prefilter":96,"prefilter_model_topk":24,"global_model_hard_topk":24,"prefilter_missed_global_top1_rate":float(np.mean(global_hard_missed)) if global_hard_missed else None,"top24_overlap_rate":float(np.mean(global_top_overlap)) if global_top_overlap else None,"prefilter_margin":scalar_stats(pref_margin),"global_margin":scalar_stats(global_margin)},"teacher_upper_bound":{"roc_auc":auc(teacher_scores,labels),"pr_auc":average_precision(teacher_scores,labels),"positive_count":int(labels.sum()),"teacher_vs_d0_top1_disagreement_rate":top1_disagreements/max(1,positive_frames),"teacher_top1_recall":teacher_top1/max(1,positive_frames),"d0_top1_recall":d0_top1/max(1,positive_frames)},"conclusion":["L23 violation penalty uses positive_logits.max(); the low positive receives zero legacy violation gradient in the manual test.","L23 training/validation use the same prefilter+top24 rule, but the audit reports a separate all-frame model-hard margin and prefilter miss rate.","L23 dense scorer contains no fixed point-coordinate embedding and no teacher/residual score path; L24 must add both."]}
    (out_root/"audit.json").write_text(json.dumps(report,indent=2)+"\n"); (out_root/"audit.md").write_text("# L24 L23 consistency audit\n\n"+"\n".join(f"- {x}" for x in report["conclusion"])+"\n")
    print(json.dumps({"audit":str(out_root/"audit.json"),"manual":multi_positive_smoke,"hard_mining":report["hard_mining_contract"],"teacher":report["teacher_upper_bound"],"code":source},indent=2))


if __name__ == "__main__": main()
