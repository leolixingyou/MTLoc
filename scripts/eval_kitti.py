#!/usr/bin/env python3
"""
Evaluate MTLoc on KITTI (zero-shot from MGL training).

Usage:
    python scripts/eval_kitti.py \
        --adapter_ckpt checkpoints/mtloc_445k.ckpt \
        --ckpt_path checkpoints/orienternet_mgl.ckpt \
        --yolopx_weights checkpoints/epoch-195.pth
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mtloc_model import create_mtloc_model
from maploc.data.kitti.dataset import KittiDataModule
from maploc.models.voting import TemplateSampler
from omegaconf import OmegaConf, read_write


def load_model(args):
    """Build model and load trained weights."""
    model = create_mtloc_model(
        orienternet_ckpt_path=args.ckpt_path,
        yolopx_weights_path=args.yolopx_weights,
        adapter_type=args.adapter_type,
    )

    with read_write(model.conf):
        model.conf.num_rotations = args.num_rotations
    model.template_sampler = TemplateSampler(
        model.projection_bev.grid_xz, model.conf.pixel_per_meter, args.num_rotations,
    )

    # Load checkpoint (full model or adapter-only)
    ckpt = torch.load(args.adapter_ckpt, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    has_backbone = any("backbone" in k for k in sd)

    if has_backbone:
        full_sd = {k.replace("model.", ""): v for k, v in sd.items() if k.startswith("model.")}
        model.load_state_dict(full_sd, strict=True)
    else:
        adapter_sd = {k.replace("model.image_encoder.adapter.", ""): v
                      for k, v in sd.items() if "adapter" in k}
        model.image_encoder.adapter.load_state_dict(adapter_sd, strict=True)

    for p in model.parameters():
        p.requires_grad = False
    return model


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
    parser = argparse.ArgumentParser(description="Evaluate MTLoc on KITTI")
    parser.add_argument("--adapter_ckpt", required=True)
    parser.add_argument("--ckpt_path", default=str(REPO_ROOT / "checkpoints/orienternet_mgl.ckpt"))
    parser.add_argument("--yolopx_weights", default=str(REPO_ROOT / "checkpoints/epoch-195.pth"))
    parser.add_argument("--kitti_root", default=str(REPO_ROOT / "data/kitti"))
    parser.add_argument("--num_rotations", type=int, default=256)
    parser.add_argument("--adapter_type", default="fpn", choices=["simple", "fpn"])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    model = load_model(args).to(args.device).eval()

    # KITTI data
    cfg = OmegaConf.create({
        "data_dir": args.kitti_root,
        "tiles_filename": "tiles.pkl",
        "splits": {"train": "train_files.txt", "val": "test1_files.txt",
                    "test": "test2_files.txt"},
        "loading": {"val": {"batch_size": 1, "num_workers": 0}},
    })
    dm = KittiDataModule(cfg)
    dm.prepare_data()
    dm.setup("fit")
    dataloader = dm.dataloader("val")
    ppm = dm.tile_manager.ppm

    # Evaluate
    all_metrics = defaultdict(list)
    with torch.no_grad():
        for batch_ in tqdm(dataloader, desc="KITTI eval"):
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch_.items()}
            pred = model(batch)
            m = compute_metrics(pred, batch, ppm)
            for k, v in m.items():
                all_metrics[k].append(v)

    # Results
    print(f"\n{'=' * 60}")
    print(f"  MTLoc on KITTI ({len(all_metrics['xy_error'])} samples)")
    print(f"{'=' * 60}")
    for key in ["lateral_error", "longitudinal_error", "yaw_error", "xy_error"]:
        errors = torch.tensor(all_metrics[key])
        label = key.replace("_error", "")
        unit = "deg" if key == "yaw_error" else "m"
        recalls = " / ".join(f"{(errors < t).float().mean().item() * 100:.2f}%"
                             for t in [1, 3, 5])
        print(f"  {label:>12} @1/3/5{unit}: {recalls}")
        print(f"  {label:>12}   mean/med: "
              f"{errors.mean().item():.3f} / {errors.median().item():.3f}")


if __name__ == "__main__":
    main()
