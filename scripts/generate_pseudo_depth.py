#!/usr/bin/env python3
"""Generate pseudo-depth labels for KITTI using Depth-Anything-V2.

Usage:
    python scripts/generate_pseudo_depth.py \
        --kitti_root data/kitti \
        --split train_files.txt \
        --output_dir data/kitti/pseudo_depth \
        --device cuda:0
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

DA_V2_ROOT = Path("/workspace/Depth-Anything-V2")
sys.path.insert(0, str(DA_V2_ROOT))

from depth_anything_v2.dpt import DepthAnythingV2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_root", type=str, default="data/kitti")
    parser.add_argument("--split", type=str, default="train_files.txt")
    parser.add_argument("--output_dir", type=str, default="data/kitti/pseudo_depth")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    kitti_root = Path(args.kitti_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
    }
    model = DepthAnythingV2(**model_configs["vits"])
    model.load_state_dict(
        torch.load(str(DA_V2_ROOT / "checkpoints/depth_anything_v2_vits.pth"), map_location="cpu")
    )
    model = model.to(args.device).eval()
    print(f"Depth-Anything-V2 (Small) loaded on {args.device}")

    # Read split file
    split_file = kitti_root / args.split
    with open(split_file) as f:
        lines = [l.strip().split()[0] for l in f if l.strip()]
    print(f"Processing {len(lines)} images from {split_file}")

    for line in tqdm(lines, desc="Generating pseudo-depth"):
        # line format: "2011_09_26/2011_09_26_drive_0002_sync/0000000048.png"
        parts = line.split("/")
        date = parts[0]
        drive = parts[1]
        frame = parts[2]

        img_path = kitti_root / date / drive / "image_02" / "data" / frame
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        with torch.no_grad():
            depth = model.infer_image(img)  # [H, W] relative depth

        # Normalize to [0, 1] (relative depth)
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        # Save as float16 npz (compact)
        out_path = output_dir / date / drive / frame.replace(".png", ".npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(out_path), depth=depth_norm.astype(np.float16))

    print(f"Done. Pseudo-depth saved to {output_dir}")


if __name__ == "__main__":
    main()
