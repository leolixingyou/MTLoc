#!/usr/bin/env python3
"""
Evaluate MTLoc on nuScenes (zero-shot from MGL training).

Usage:
    python scripts/eval_nuscenes.py \
        --ckpt checkpoints/mtloc_445k.ckpt \
        --version v1.0-trainval --tiles_dir osm_tiles_trainval
"""

import argparse
import sys
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf, read_write

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mtloc_model import create_mtloc_model
from maploc.data.nuscenes.dataset import NuScenesDataModule
from maploc.data.torch import collate
from maploc.models.voting import TemplateSampler


def deg2rad(deg):
    return deg * torch.pi / 180.0


def rotmat2d(angle):
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack([c, -s, s, c], dim=-1).reshape(*angle.shape, 2, 2)


def compute_metrics(pred, data, ppm):
    uv_pred, yaw_pred = pred["uv_max"].cpu(), pred["yaw_max"].cpu()
    uv_gt = data["uv"].cpu()
    rpw_gt = data["roll_pitch_yaw"].cpu()
    yaw_gt = rpw_gt[..., -1]

    xy_error = torch.norm(uv_pred - uv_gt, dim=-1) / ppm
    yaw_rad = deg2rad(rpw_gt[..., -1])
    shift = (uv_pred - uv_gt) * yaw_rad.new_tensor([-1, 1])
    shift = (rotmat2d(yaw_rad) @ shift.unsqueeze(-1)).squeeze(-1)
    lateral_error = torch.abs(shift[..., 0]) / ppm
    longitudinal_error = torch.abs(shift[..., 1]) / ppm
    yaw_error = torch.abs(yaw_pred - yaw_gt)
    yaw_error = torch.min(yaw_error, 360.0 - yaw_error)

    return {
        "xy_error": xy_error.item(),
        "lateral_error": lateral_error.item(),
        "longitudinal_error": longitudinal_error.item(),
        "yaw_error": yaw_error.item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate MTLoc on nuScenes")
    parser.add_argument("--ckpt", required=True, help="MTLoc checkpoint")
    parser.add_argument("--ckpt_path", default=str(REPO_ROOT / "checkpoints/orienternet_mgl.ckpt"))
    parser.add_argument("--yolopx_weights", default=str(REPO_ROOT / "checkpoints/epoch-195.pth"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--tiles_dir", default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)

    # Build model
    model = create_mtloc_model(
        orienternet_ckpt_path=args.ckpt_path,
        yolopx_weights_path=args.yolopx_weights,
        adapter_type="fpn",
    )
    with read_write(model.conf):
        model.conf.num_rotations = 256
    model.template_sampler = TemplateSampler(
        model.projection_bev.grid_xz, model.conf.pixel_per_meter, 256,
    )

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()
          if k.startswith("model.")}
    model.load_state_dict(sd, strict=True)
    model = model.to(device).eval()
    ppm = model.conf.pixel_per_meter

    # Data
    cfg_dict = {"version": args.version}
    if args.tiles_dir:
        cfg_dict["tiles_dir"] = args.tiles_dir
    dm = NuScenesDataModule(OmegaConf.create(cfg_dict))
    dm.prepare_data()
    dm.setup(stage="test")
    dset = dm.dataset("test")
    n = args.num_samples or len(dset)

    # Evaluate
    all_metrics = defaultdict(list)
    with torch.no_grad():
        for i in tqdm(range(n), desc="nuScenes eval"):
            item = dset[i]
            batch = collate([item])
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            if "camera" in batch:
                batch_dev["camera"] = batch["camera"].to(device)

            pred = model(batch_dev)
            m = compute_metrics(pred, batch_dev, ppm)
            for k, v in m.items():
                all_metrics[k].append(v)

    # Results
    print(f"\n{'=' * 60}")
    print(f"  MTLoc on nuScenes ({n} samples)")
    print(f"{'=' * 60}")
    for key in ["xy_error", "lateral_error", "longitudinal_error", "yaw_error"]:
        errors = torch.tensor(all_metrics[key])
        label = key.replace("_error", "")
        unit = "deg" if "yaw" in key else "m"
        for t in [1, 2, 5]:
            r = (errors < t).float().mean().item() * 100
            print(f"  {label:>12} @{t}{unit}: {r:.1f}%")
        print(f"  {label:>12}   mean: {errors.mean():.2f}, median: {errors.median():.2f}")


if __name__ == "__main__":
    main()
