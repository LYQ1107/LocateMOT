"""Cache frozen RN50 local crops and full-frame context for Stage L17 A2.

The L16 bank remains the sole candidate/identity source.  This script only
reads the bank boxes and the corresponding raw images.  It stores a pooled
2048-channel output of the RN50 spatial trunk for each crop, plus one pooled
full-frame feature per frame.  No GT or expression labels are read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
BANK_ROOT = ROOT / "outputs/l16/track_banks_dedup"
DEFAULT_OUT = ROOT / "outputs/l17/ikun_rn50_cache"
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]


def image_path(dataset: str, video: str, frame_id: int) -> Path:
    if dataset == "kitti":
        return (ROOT / "data/kitti_tracking_training/image_02" / video /
                f"{int(frame_id):06d}.png")
    return (ROOT / "data/refer_dance/DanceTrack/training/image_02" /
            video / "img1" / f"{int(frame_id):08d}.jpg")


def square_image(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas.resize((size, size), Image.Resampling.BICUBIC)


def tensor_image(image: Image.Image, size: int) -> torch.Tensor:
    value = torch.from_numpy(np.array(square_image(image, size), copy=True)).permute(2, 0, 1)
    value = value.float() / 255.0
    return (value - MEAN) / STD


def crop_image(image: Image.Image, box) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = [int(float(value)) for value in box]
    x1, y1 = max(0, min(width - 1, x1)), max(0, min(height - 1, y1))
    x2, y2 = max(x1 + 1, min(width, x2)), max(y1 + 1, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return Image.new("RGB", (2, 2), (0, 0, 0))
    return image.crop((x1, y1, x2, y2))


def spatial_trunk(visual, images: torch.Tensor) -> torch.Tensor:
    """Run the OpenAI CLIP RN50 trunk before attention pooling."""
    value = images.to(device=next(visual.parameters()).device,
                      dtype=visual.conv1.weight.dtype)
    value = visual.relu1(visual.bn1(visual.conv1(value)))
    value = visual.relu2(visual.bn2(visual.conv2(value)))
    value = visual.relu3(visual.bn3(visual.conv3(value)))
    value = visual.avgpool(value)
    value = visual.layer1(value)
    value = visual.layer2(value)
    value = visual.layer3(value)
    value = visual.layer4(value)
    return value.mean(dim=(-1, -2))


@torch.no_grad()
def encode_images(model, images: list[torch.Tensor], device,
                  batch_size: int) -> np.ndarray:
    if not images:
        return np.zeros((0, 2048), np.float16)
    output = np.zeros((len(images), 2048), np.float16)
    visual = model.visual
    for start in range(0, len(images), batch_size):
        batch = torch.stack(images[start:start + batch_size]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=device.type == "cuda"):
            features = spatial_trunk(visual, batch)
        output[start:start + len(features)] = features.float().cpu().numpy().astype(
            np.float16)
    return output


def cache_video(dataset: str, video: str, output_root: Path, model,
                device, batch_size: int, bank_root: Path) -> dict:
    bank_path = bank_root / dataset / f"{video}.pt"
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    tensors = bank["tensors"]
    frame_ids = tensors["frame_ids"].tolist()
    frame_ptr = tensors["frame_ptr"].tolist()
    boxes = tensors["box"].numpy()
    # Keep only the encoded cache and one frame's decoded images in memory.
    # The previous implementation accumulated every resized crop and full
    # frame for a video before encoding, which made concurrent workers needlessly
    # expensive in host RAM.
    local = np.empty((len(boxes), 2048), dtype=np.float16)
    global_frame = np.empty((len(frame_ids), 2048), dtype=np.float16)
    for frame_index, frame_id in enumerate(frame_ids):
        path = image_path(dataset, video, int(frame_id))
        if not path.exists():
            raise FileNotFoundError(f"missing raw image: {path}")
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            global_image = tensor_image(image, 672)
            global_frame[frame_index] = encode_images(
                model, [global_image], device, 1)[0]
            start, end = int(frame_ptr[frame_index]), int(frame_ptr[frame_index + 1])
            frame_images = [
                tensor_image(crop_image(image, box), 224)
                for box in boxes[start:end]
            ]
            if end > start:
                local[start:end] = encode_images(
                    model, frame_images, device, batch_size)
            del frame_images, global_image, image
    output_root = output_root / dataset
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / f"{video}.pt"
    payload = {
        "metadata": {
            "format": "locatemot-l17-ikun-rn50-cache-v1",
            "dataset": dataset, "video_id": video,
            "model": "OpenAI CLIP RN50 spatial trunk, channel-pooled",
            "input_local": 224, "input_global": 672,
            "frames": len(frame_ids), "observations": len(boxes),
            "bank_path": str(bank_path),
            "bank_reuse_equivalence_sha256": bank["metadata"].get(
                "reuse_equivalence_sha256"),
            "raw_image_rule": "KITTI frame id + 0; Dance frame id as 8-digit jpg",
        },
        "local": torch.from_numpy(local),
        "global_frame": torch.from_numpy(global_frame),
        "frame_ptr": tensors["frame_ptr"].clone(),
        "frame_ids": tensors["frame_ids"].clone(),
    }
    temporary = output.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    audit = dict(payload["metadata"])
    audit["bytes"] = output.stat().st_size
    audit["sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["kitti", "dance_train", "dance_eval", "all"],
                        required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--videos", nargs="*", default=None)
    parser.add_argument("--bank-root", default=str(BANK_ROOT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A2 cache requires CUDA for the RN50 trunk")
    datasets = ["kitti", "dance_train", "dance_eval"] \
        if args.dataset == "all" else [args.dataset]
    jobs = []
    bank_root = Path(args.bank_root)
    for dataset in datasets:
        videos = sorted(path.stem for path in (bank_root / dataset).glob("*.pt"))
        if args.videos is not None:
            videos = [video for video in videos if video in set(args.videos)]
        jobs.extend((dataset, video) for video in videos)
    jobs = [job for job in jobs if int(hashlib.md5(
        f"{job[0]}:{job[1]}".encode()).hexdigest(), 16) % args.num_shards == args.shard]
    output_root = Path(args.out)
    output_root.mkdir(parents=True, exist_ok=True)
    import clip
    model, _ = clip.load("RN50", device=device)
    model.eval()
    rows = []
    started = time.time()
    for index, (dataset, video) in enumerate(jobs):
        output = output_root / dataset / f"{video}.pt"
        if output.exists() and not args.overwrite:
            print(f"[l17-rn50-cache] skip {dataset}/{video}", flush=True)
            continue
        t0 = time.time()
        audit = cache_video(dataset, video, output_root, model, device,
                            args.batch_size, bank_root)
        rows.append(audit)
        print(f"[l17-rn50-cache] {dataset}/{video} {index + 1}/{len(jobs)} "
              f"frames={audit['frames']} obs={audit['observations']} "
              f"seconds={time.time() - t0:.1f}", flush=True)
    (output_root / f"worker_{args.shard:02d}.json").write_text(json.dumps({
        "dataset": args.dataset, "shard": args.shard,
        "num_shards": args.num_shards, "videos": rows,
        "wall_seconds": time.time() - started,
    }, indent=2) + "\n")
    print(f"[l17-rn50-cache] done shard={args.shard} "
          f"seconds={time.time() - started:.1f}", flush=True)


if __name__ == "__main__":
    main()
