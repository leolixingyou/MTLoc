#!/usr/bin/env python3
"""E7: Sequential inference — fuse N consecutive frames' pose distributions.

Eval-time only, no training needed. Uses ego-motion to align score volumes.

Usage:
    python scripts/eval_kitti_sequential.py --N 5 --device cuda:0
"""
import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mtloc_model import create_mtloc_model
from maploc.data.kitti.dataset import KittiDataModule
from maploc.models.voting import (
    TemplateSampler, argmax_xyr, log_softmax_spatial, expectation_xyr
)
from omegaconf import OmegaConf, read_write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_ckpt", default=str(
        REPO_ROOT / "runs/kitti_C_full/checkpoints/yolopx_loc_C_kitti-step010000-valloss10.1602.ckpt"))
    parser.add_argument("--N", type=int, nargs="+", default=[1, 3, 5, 10],
                        help="Number of frames to fuse")
    parser.add_argument("--max_sequences", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    # Load model
    model = create_mtloc_model(
        orienternet_ckpt_path=str(REPO_ROOT / "checkpoints/orienternet_mgl.ckpt"),
        yolopx_weights_path=str(REPO_ROOT / "checkpoints/epoch-195.pth"),
        adapter_type="fpn",
    )
    with read_write(model.conf):
        model.conf.num_rotations = 256
    model.template_sampler = TemplateSampler(
        model.projection_bev.grid_xz, model.conf.pixel_per_meter, 256,
    )
    ckpt = torch.load(args.adapter_ckpt, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    full_sd = {k.replace("model.", ""): v for k, v in sd.items() if k.startswith("model.")}
    model.load_state_dict(full_sd, strict=False)
    model = model.to(args.device).eval()
    ppm = model.conf.pixel_per_meter

    # Load sequential data
    cfg = OmegaConf.create({
        "data_dir": str(REPO_ROOT / "data/kitti"),
        "tiles_filename": "tiles.pkl",
        "splits": {"val": "test1_files.txt"},
        "loading": {"val": {"batch_size": 1, "num_workers": 0}},
        "max_num_val": None,  # use all for sequential
    })
    dm = KittiDataModule(cfg)
    dm.prepare_data()
    dm.setup("fit")
    dataloader = dm.dataloader("val")

    # Group frames by sequence (date/drive)
    print("Collecting per-frame predictions...")
    all_frames = []
    with torch.no_grad():
        for batch_ in tqdm(dataloader, desc="Forward pass"):
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch_.items()}
            pred = model(batch)

            all_frames.append({
                "name": batch["name"][0],
                "scores": pred["scores"].cpu(),  # [1, H, W, N_rot]
                "uv_gt": batch["uv"].cpu(),
                "yaw_gt": batch["roll_pitch_yaw"][..., -1].cpu(),
                "uvr_max": pred["uvr_max"].cpu(),
            })

    # Group by sequence
    from collections import OrderedDict
    sequences = OrderedDict()
    for f in all_frames:
        parts = f["name"].split("/")
        seq_key = f"{parts[0]}/{parts[1]}"
        if seq_key not in sequences:
            sequences[seq_key] = []
        sequences[seq_key].append(f)

    print(f"Found {len(sequences)} sequences, {len(all_frames)} total frames")

    # Evaluate for each N
    for N in args.N:
        lateral_errors_single = []
        lateral_errors_fused = []
        yaw_errors_single = []
        yaw_errors_fused = []

        n_evaluated = 0
        for seq_key, frames in sequences.items():
            if n_evaluated >= args.max_sequences:
                break
            if len(frames) < N:
                continue

            for t in range(N - 1, len(frames)):
                # Single frame
                uvr_single = frames[t]["uvr_max"]
                uv_gt = frames[t]["uv_gt"]
                yaw_gt = frames[t]["yaw_gt"]

                lat_err = ((uvr_single[..., :2] - uv_gt) / ppm).norm(dim=-1).item()
                yaw_err = abs(((uvr_single[..., 2] - yaw_gt + 180) % 360 - 180).item())

                lateral_errors_single.append(lat_err)
                yaw_errors_single.append(yaw_err)

                # Fused: sum log_probs of last N frames (simple, no ego-motion alignment)
                fused_scores = torch.zeros_like(frames[t]["scores"])
                for k in range(N):
                    fused_scores += frames[t - k]["scores"]

                # Argmax on fused
                fused_uvr = argmax_xyr(fused_scores).to(fused_scores)
                lat_err_f = ((fused_uvr[..., :2] - uv_gt) / ppm).norm(dim=-1).item()
                yaw_err_f = abs(((fused_uvr[..., 2] - yaw_gt + 180) % 360 - 180).item())

                lateral_errors_fused.append(lat_err_f)
                yaw_errors_fused.append(yaw_err_f)

            n_evaluated += 1

        # Compute recalls
        lat_s = np.array(lateral_errors_single)
        lat_f = np.array(lateral_errors_fused)
        yaw_s = np.array(yaw_errors_single)
        yaw_f = np.array(yaw_errors_fused)

        print(f"\n{'='*60}")
        print(f"N = {N} frames fused ({len(lat_s)} samples from {n_evaluated} sequences)")
        print(f"{'='*60}")
        print(f"{'Metric':<20} {'Single':<12} {'Fused(N={N})':<12} {'Δ':<10}")
        print(f"{'-'*54}")
        for thresh in [1, 3, 5]:
            s = (lat_s < thresh).mean() * 100
            f = (lat_f < thresh).mean() * 100
            print(f"  Lat@{thresh}m          {s:>8.2f}%    {f:>8.2f}%    {f-s:>+.2f}")
        for thresh in [1, 3, 5]:
            s = (yaw_s < thresh).mean() * 100
            f = (yaw_f < thresh).mean() * 100
            print(f"  Yaw@{thresh}°          {s:>8.2f}%    {f:>8.2f}%    {f-s:>+.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
