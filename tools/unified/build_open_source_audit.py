#!/usr/bin/env python3
"""Materialize the reproducible U0 audit of the ten reference repositories."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = Path("/data2/user/reference_repos")

RECORDS = [
    {
        "name": "MOTR",
        "path": "MOTR",
        "url": "https://github.com/megvii-research/MOTR.git",
        "commit": "8690da3392159635ca37c31975126acf40220724",
        "license": "MIT (root); Deformable-DETR portions retain Apache-2.0 notices",
        "entrypoint": "configs/r50_motr_train.sh -> main.py",
        "recipe": "8 GPUs; 200 epochs; lr=2e-4, backbone lr=2e-5; batch=1; random interval=10; sampler steps 50/90/150 with lengths 2/3/4/5; random_drop=.1; fp_ratio=.3; QIM; extra_track_attn",
        "mechanisms": "Persistent track instances, QIM active-track selection/drop and false-positive injection, optional memory bank (default history length 4), Deformable-DETR multi-frame decoder.",
        "sources": ["configs/r50_motr_train.sh", "main.py", "models/motr.py", "models/qim.py", "models/memory_bank.py"],
        "portability": "Recipe is commented and assumes local pretrained COCO/Deformable-DETR weights and MOT17 paths.",
        "unified_use": "Primary identity-query and track-memory reference for the shared MOT/OVMOT/RMOT core.",
    },
    {
        "name": "TransRMOT",
        "path": "TransRMOT",
        "url": "https://github.com/wudongming97/RMOT.git",
        "commit": "d4fedb35538e79a743ff78ff946abc6c84453cab",
        "license": "MIT (root); Deformable-DETR portions retain Apache-2.0 notices",
        "entrypoint": "configs/r50_rmot_train.sh -> main.py",
        "recipe": "8 GPUs; 100 epochs; lr=1e-4, backbone lr=1e-5; lr_drop=50; batch=1; four sampler lengths of 2; random_drop=.1; fp_ratio=.3; QIM; refer_loss_coef=2",
        "mechanisms": "MOTR-style persistent track queries plus a frozen Roberta text encoder, vision-language fusion and a refer_embed/loss_refers branch.",
        "sources": ["configs/r50_rmot_train.sh", "models/transrmot.py", "models/qim.py", "models/memory_bank.py"],
        "portability": "The production path in transrmot.py contains a hard-coded /data_2/zyn/.../roberta_base path; pretrained weights are external.",
        "unified_use": "Reference for RMOT specification conditioning and refer-score supervision, while retaining one shared track state.",
    },
    {
        "name": "TempRMOT",
        "path": "TempRMOT",
        "url": "https://github.com/zyn213/TempRMOT.git",
        "commit": "6a65640d849fdee4a32bb055945ee34c3b0edeb1",
        "license": "No root license file observed; bundled TrackEval is MIT",
        "entrypoint": "configs/temp_rmot_train.sh -> main.py",
        "recipe": "4 GPUs; 60 epochs; lr=1e-4, backbone lr=1e-5; lr_drop=40; batch=1; four sampler lengths of 5; random_drop=.1; fp_ratio=.3; hist_len=5; refer_loss_coef=2",
        "mechanisms": "MOTR/QIM track queries with frozen Roberta text features, spatial-temporal reasoner and separate temporal parameter group (lr_trans=1e-5).",
        "sources": ["configs/temp_rmot_train.sh", "main.py", "models/transrmot_pro.py", "models/spatial_temporal_reason.py", "models/memory_bank.py"],
        "portability": "Recipe uses /home/zyn and local Refer-KITTI paths; no root license was found in the shallow checkout.",
        "unified_use": "Reference for causal history and language-conditioned temporal reasoning in the shared dynamics core.",
    },
    {
        "name": "DKGTrack",
        "path": "DKGTrack",
        "url": "https://github.com/acyddl/DKGTrack.git",
        "commit": "197f354443bd1e7b490d204456a7654b7d1e4ccd",
        "license": "No root license file observed; bundled TrackEval is MIT",
        "entrypoint": "configs/dkgtrack_rmot_train.sh -> main.py",
        "recipe": "4 GPUs; 100 epochs; lr=1e-4, backbone lr=1e-5; lr_drop=40; batch=1; sampler steps 60/80/90 with lengths 5/5/5/5; random_drop=.1; fp_ratio=.3; hist_len=5; refer_loss_coef=2. RK variant uses 3 GPUs, lengths 4 and hist_len=4.",
        "mechanisms": "Deformable-DETR-plus two-stage/refinement path with spatial-temporal reasoner, bidirectional vision-language fusion and a history-conditioned refer branch.",
        "sources": ["configs/dkgtrack_rmot_train.sh", "configs/dkgtrack_rmot_train_rk.sh", "models/spatial_temporal_reason.py", "models/fuse_modules.py", "models/deformable_transformer_plus.py"],
        "portability": "Scripts contain /dkgtrack and /data2/lgy absolute paths and a resume checkpoint in the RK variant.",
        "unified_use": "Reference for explicit motion/history and relation cues; its temporal modules are candidates for adapters around one shared foundation.",
    },
    {
        "name": "FlexHook",
        "path": "FlexHook",
        "url": "https://github.com/buptLwz/FlexHook.git",
        "commit": "bd1acc38634b28525d54dc6e0fcb38335f0029f9",
        "license": "MIT",
        "entrypoint": "main.py with configs/train/train-kitti1.yaml or train-kitti2.yaml",
        "recipe": "20 epochs; AdamW base lr=3e-5; weight_decay=1e-4; multistep milestones 12/18; image size 224x672; 4 sampled frames from an 8-frame window at stride 2; 36 train expressions (V1) or the V2 config's scale 10 validation setting.",
        "mechanisms": "ResNet-34 visual encoder plus Roberta text encoder, conditional positional embeddings, feature sampling/grid_sample hooks, point-dispersion/noise losses and expression-conditioned visual fusion.",
        "sources": ["configs/train/train-kitti1.yaml", "configs/train/train-kitti2.yaml", "models/mymodel.py", "main.py"],
        "portability": "Config refers to local dataset_root and pretrained language/visual assets; this is a referring model rather than an end-to-end persistent MOT tracker.",
        "unified_use": "Reference for conditioned spatial sampling and prompt-to-region hooks, to be implemented as a small shared-path adapter.",
    },
    {
        "name": "iKUN",
        "path": "iKUN",
        "url": "https://github.com/dyhBUPT/iKUN.git",
        "commit": "4db56bfaec703590e0fdfd1684d9769467a67e05",
        "license": "MIT",
        "entrypoint": "train.py (opts.py controls the run)",
        "recipe": "Default GPUs 0,1; train batch 8; AdamW base lr=1e-5; weight_decay=1e-5; max_epoch=100; frame window 8, 2 sampled frames at stride 4; one expression per sample; CLIP RN50 feature_dim=1024.",
        "mechanisms": "Frozen/partly adapted CLIP image-text similarity model with local/global image crops, text-guided pooling and SimilarityLoss; it is a similarity learner, not a persistent query tracker.",
        "sources": ["opts.py", "train.py", "model.py", "loss.py", "similarity_calibration.py"],
        "portability": "Default save/data/track roots are under /data1/dyh/results/RMOT/Git; CLIP weights are not in the repository.",
        "unified_use": "Reference for prompt/text embedding calibration and candidate ranking, not for replacing the unified identity dynamics.",
    },
    {
        "name": "OVTR",
        "path": "OVTR",
        "url": "https://github.com/jinyanglii/OVTR.git",
        "commit": "500e72c19bf5f7f8717546911a5639fdc26bfee5",
        "license": "MIT",
        "entrypoint": "ovtr/tools/ovtr_multi_frame_train.sh -> ovtr/main.py",
        "recipe": "4 GPUs; staged 1 epoch at lr=2e-4/backbone lr=2e-5 then 16 epochs resumed at 4e-5/4e-6; batch=1; sampler steps 4/7/14 and lengths 2/3/4/5; random_drop=.1; fp_ratio=.3; track query iteration CIP; two-stage and box refinement.",
        "mechanisms": "Open-vocabulary LVIS/TAO tracking with 900 queries, text cross-attention, extra track attention, artificial image sequences and track embedding update.",
        "sources": ["ovtr/tools/ovtr_multi_frame_train.sh", "ovtr/config/ovtr_5_frame_train_val.py", "ovtr/models/ovtr.py", "ovtr/models/updater.py"],
        "portability": "Requires TAO/LVIS files, HDF5 image store, DetPro/CLIP embeddings and local model_zoo paths; it is not a Refer-KITTI loader.",
        "unified_use": "Reference for OVMOT text cross-attention and open-vocabulary track association under the shared track representation.",
    },
    {
        "name": "COVTrack",
        "path": "COVTrack",
        "url": "https://github.com/zekunqian/COVTrack.git",
        "commit": "9b0ced5779ee36f5dd73dbe39b5ae5d57abb4b3b",
        "license": "Apache-2.0",
        "entrypoint": "tools/dist_train.sh with configs/uncertainty-ovtrack-teta/ovtrack_r50_ctao_train.py",
        "recipe": "C-TAO config: total_epochs=10; SGD lr=.002, momentum=.9, weight_decay=1e-4; step schedule at 3/5/10; samples_per_gpu=12; one reference image sampled within scope 30; detector frozen; optional bsub wrapper.",
        "mechanisms": "Faster-RCNN/FPN proposal and ROI track embedding, prompt-based open-vocabulary classification, uncertainty consistency from association/bbox/class cues, memo_frames=10, momentum embedding=.8 and object score=.5, bisoftmax plus cosine matching.",
        "sources": ["configs/uncertainty-ovtrack-teta/ovtrack_r50_ctao_train.py", "ovtrack/models/roi_heads/ovtrack_roi_head.py", "ovtrack/models/trackers/ovtracker.py", "tools/bsub_train.sh"],
        "portability": "MMDetection/MMCV stack and TAO/C-TAO annotations are required; default config uses local saved_models and data/tao paths.",
        "unified_use": "Reference for reliability-aware cue fusion and open-vocabulary association, to be distilled into the shared online state.",
    },
    {
        "name": "ReferDINO",
        "path": "ReferDINO",
        "url": "https://github.com/iSEE-Laboratory/ReferDINO.git",
        "commit": "3cfc01f57dff97f7d801b1bd54c251e0f34fcef8",
        "license": "No root license file observed in the shallow checkout",
        "entrypoint": "main.py -c configs/davis_swinb.yaml -rm train -ng N",
        "recipe": "Swin-B GroundingDINO; lr=5e-5; weight_decay=1e-4; clip norm=.1; 900 queries; temporal_layer=3; LoRA rank=32 in encoder/decoder; tracking_alpha=.1; DAVIS uses 6 frames and eval batch 1. CLI defaults are epochs=10, batch=2, lr_drop=[8].",
        "mechanisms": "GroundingDINO text/image fusion, temporal decoder and query matching, low-rank adaptation, and a 3-layer segmentation head for referring video object segmentation.",
        "sources": ["configs/coco_swinb.yaml", "configs/davis_swinb.yaml", "main.py", "trainer.py", "models/GroundingDINO/groundingdino.py", "models/GroundingDINO/temporal_modules.py"],
        "portability": "Requires loralib, GroundingDINO custom CUDA operators, pretrained Swin-B/BERT weights and DAVIS/RefVOS data; no root license was found.",
        "unified_use": "Reference for temporal language conditioning and mask output, while retaining a single shared track/online state.",
    },
    {
        "name": "Open-GroundingDino",
        "path": "Open-GroundingDino",
        "url": "https://github.com/longzw1997/Open-GroundingDino.git",
        "commit": "d248268ac9cab808d4aa2691f4a76972ec5d9ab4",
        "license": "MIT",
        "entrypoint": "train_dist.sh -> main.py",
        "recipe": "Default ODVG config: batch=4; 15 epochs; lr=1e-4; backbone/BERT group lr=1e-5; weight_decay=1e-4; lr_drop=4; Swin-T; 900 queries; 4 feature levels; DN scalar=100; Hungarian matching; focal alpha=.25/gamma=2.",
        "mechanisms": "Open-domain GroundingDINO detector with BERT text encoder, multi-scale deformable attention, denoising queries and box/class Hungarian losses; no persistent video identity dynamics.",
        "sources": ["train_dist.sh", "config/cfg_odvg.py", "config/datasets_mixed_odvg.json", "models/GroundingDINO/groundingdino.py", "models/GroundingDINO/matcher.py"],
        "portability": "Dataset JSON uses placeholder path/ and nine external datasets; pretrained model and BERT paths are shell environment overrides.",
        "unified_use": "Reference for the U1 shared visual/specification foundation and proposal training, not a complete tracker.",
    },
]


def main() -> int:
    output_dir = ROOT / "outputs/unified"
    report_dir = ROOT / "reports/unified"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "locatemot-u-open-source-audit-v1",
        "project_root": str(ROOT),
        "reference_root": str(REFERENCE_ROOT),
        "repository_count": len(RECORDS),
        "repositories": RECORDS,
        "official_test_labels_read": False,
        "status": "audited_source_and_recipe_only",
        "notes": [
            "The audit records source-level training contracts from shallow clones; it does not claim external training or benchmark scores.",
            "Absolute paths and missing licenses are recorded as portability/provenance findings rather than silently repaired.",
        ],
    }
    (output_dir / "open_source_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# LocateMOT-U open-source training audit — U0",
        "",
        "This is a source and recipe audit of the ten requested shallow clones. It records exact HEAD commits, license evidence, entrypoints, training values, architectural mechanisms and portability constraints. No external training run or benchmark score is claimed here.",
        "",
        "| Repository | HEAD | License evidence | Entry point | Unified role |",
        "|---|---|---|---|---|",
    ]
    for item in RECORDS:
        lines.append(
            f"| {item['name']} | `{item['commit'][:12]}` | {item['license']} | `{item['entrypoint']}` | {item['unified_use']} |"
        )
    lines += ["", "## Per-repository findings", ""]
    for item in RECORDS:
        lines += [
            f"### {item['name']}",
            "",
            f"- Clone: `{item['url']}` at `{item['commit']}` (`{REFERENCE_ROOT / item['path']}`).",
            f"- License: {item['license']}.",
            f"- Recipe: {item['recipe']}.",
            f"- Mechanisms: {item['mechanisms']}",
            f"- Read sources: {', '.join('`' + source + '`' for source in item['sources'])}.",
            f"- Portability/provenance: {item['portability']}",
            f"- LocateMOT-U use: {item['unified_use']}",
            "",
        ]
    lines += [
        "## Cross-repository conclusions",
        "",
        "1. MOTR, TransRMOT, TempRMOT and DKGTrack share the most reusable persistent-query and multi-frame machinery. Their QIM false-positive/drop schedules are training mechanisms, not separate tracker identities.",
        "2. OVTR and Open-GroundingDino provide open-vocabulary text-conditioned detection/association ingredients, but neither supplies the required Refer-KITTI legal boundary or a unified online state.",
        "3. FlexHook, iKUN and ReferDINO provide complementary prompt, embedding and temporal language-conditioning ideas. Their separate visual encoders or similarity-only interfaces cannot become separate LocateMOT-U backbones.",
        "4. COVTrack contributes reliability and memo fusion; its TAO/C-TAO data path is an external acquisition item and is not evidence on Refer-KITTI.",
        "5. All ten repositories are reference implementations. U1 must reimplement the selected components behind one `SharedVisualFoundation`, one `UniversalTrackDecoder`, one memory and one online-state schema, with provenance for every imported checkpoint and operator.",
        "",
        "The U0 external baseline sanity is a separate status gate. Missing operators, checkpoints, data or license evidence remain recorded and are not replaced with an untracked substitute.",
        "",
    ]
    (report_dir / "OPEN_SOURCE_TRAINING_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"repository_count": len(RECORDS), "status": payload["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
