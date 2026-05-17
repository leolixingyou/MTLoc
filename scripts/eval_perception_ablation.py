#!/usr/bin/env python3
"""感知子任务独立贡献消融实验.

分别屏蔽 ELANNet 的三个感知 head (Det/DA/LL) 对应的特征通道，
测量对定位性能的影响。确认哪个感知子任务对定位贡献最大。

方法：在 backbone 的 C2-C5 特征上，用 Grad-CAM 确定每个感知任务
最依赖的 channel 子集，然后 mask 该子集后重新评估定位。

Usage:
    python scripts/eval_perception_ablation.py --device cuda:0
"""
import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mtloc_model import create_mtloc_model
from maploc.data.kitti.dataset import KittiDataModule
from maploc.models.voting import TemplateSampler, argmax_xyr
from omegaconf import OmegaConf, read_write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=500)
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
    ckpt_path = REPO_ROOT / "runs/kitti_C_full/checkpoints/yolopx_loc_C_kitti-step010000-valloss10.1602.ckpt"
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    full_sd = {k.replace("model.", ""): v for k, v in sd.items() if k.startswith("model.")}
    model.load_state_dict(full_sd, strict=False)
    model = model.to(args.device).eval()
    ppm = model.conf.pixel_per_meter

    # Data
    cfg = OmegaConf.create({
        "data_dir": str(REPO_ROOT / "data/kitti"),
        "tiles_filename": "tiles.pkl",
        "splits": {"val": "test1_files.txt"},
        "loading": {"val": {"batch_size": 1, "num_workers": 0}},
        "max_num_val": args.max_samples,
    })
    dm = KittiDataModule(cfg)
    dm.prepare_data()
    dm.setup("fit")
    dataloader = dm.dataloader("val")

    # ELANNet has 4 output scales: C2(256), C3(512), C4(1024), C5(1024)
    # YOLOPX uses these for:
    #   Det (detection): primarily C4/C5 (high-level semantic)
    #   DA (drivable area): primarily C2/C3 (spatial detail)
    #   LL (lane line): primarily C2/C3 (fine structure)
    #
    # Ablation strategy: zero out specific scale features before FPN adapter

    ablation_configs = {
        "baseline": None,  # no masking
        "mask_C5": [3],    # mask deepest (Det-heavy)
        "mask_C4": [2],    # mask mid-deep
        "mask_C3": [1],    # mask mid (most important per E3)
        "mask_C2": [0],    # mask shallowest (DA/LL-heavy)
        "mask_C4C5": [2, 3],  # mask deep layers (Det)
        "mask_C2C3": [0, 1],  # mask shallow layers (DA/LL)
        "only_C3": [0, 2, 3], # keep only C3 (E3 says most important)
    }

    print("=" * 70)
    print("Perception Subtask Ablation: Impact on Localization")
    print("=" * 70)

    results = {}
    for config_name, mask_indices in ablation_configs.items():
        lateral_errors = []
        yaw_errors = []

        with torch.no_grad():
            for batch_ in tqdm(dataloader, desc=config_name, leave=False):
                batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch_.items()}
                image = batch["image"]

                # Get backbone features
                model.image_encoder.backbone.eval()
                c2, c3, c4, c5 = model.image_encoder.backbone(image)
                features = [c2, c3, c4, c5]

                # Apply mask
                if mask_indices is not None:
                    for idx in mask_indices:
                        features[idx] = torch.zeros_like(features[idx])

                # Run through FPN adapter manually
                from collections import OrderedDict
                multi_scale = OrderedDict([
                    ("c2", features[0]), ("c3", features[1]),
                    ("c4", features[2]), ("c5", features[3])
                ])
                adapted = model.image_encoder.adapter(multi_scale)

                # Replace image encoder output and run rest of forward
                # We need to manually do the rest of _forward
                f_image = adapted
                camera = batch["camera"].scale(1 / model.image_encoder.scales[0])
                camera = camera.to(f_image.device)

                scales = model.scale_classifier(f_image.moveaxis(1, -1))
                f_polar = model.projection_polar(f_image, scales, camera)
                with torch.autocast("cuda", enabled=False):
                    f_bev, valid_bev, _ = model.projection_bev(f_polar.float(), None, camera.float())

                if model.bev_net is not None:
                    pred_bev = model.bev_net({"input": f_bev})
                    f_bev_out = pred_bev["output"]
                else:
                    f_bev_out = model.feature_projection(f_bev.moveaxis(1, -1)).moveaxis(-1, 1)

                pred_map = model.map_encoder(batch)
                f_map = pred_map["map_features"][0]

                scores = model.exhaustive_voting(f_bev_out, f_map, valid_bev, None)
                scores = scores.moveaxis(1, -1)

                uvr = argmax_xyr(scores).to(scores)
                xy_gt = batch["uv"]
                yaw_gt = batch["roll_pitch_yaw"][..., -1]

                lat_err = ((uvr[..., :2] - xy_gt) / ppm).norm(dim=-1).item()
                yaw_err = abs(((uvr[..., 2] - yaw_gt + 180) % 360 - 180).item())

                lateral_errors.append(lat_err)
                yaw_errors.append(yaw_err)

        lat_arr = np.array(lateral_errors)
        yaw_arr = np.array(yaw_errors)
        r = {
            "lat@5m": (lat_arr < 5).mean() * 100,
            "lat@1m": (lat_arr < 1).mean() * 100,
            "yaw@5": (yaw_arr < 5).mean() * 100,
            "lat_med": np.median(lat_arr),
        }
        results[config_name] = r

    # Print results
    print()
    print(f"{'Config':<15} {'Lat@1m':>8} {'Lat@5m':>8} {'Yaw@5°':>8} {'Lat med':>8} {'Δ Lat@5m':>10}")
    print("-" * 60)
    baseline = results["baseline"]
    for name, r in results.items():
        delta = r["lat@5m"] - baseline["lat@5m"]
        marker = "" if name == "baseline" else f"{delta:+.2f}"
        print(f"  {name:<13} {r['lat@1m']:>8.2f} {r['lat@5m']:>8.2f} {r['yaw@5']:>8.2f} {r['lat_med']:>8.3f} {marker:>10}")

    # Save
    out_path = REPO_ROOT / "runs/p2_attribution/perception_ablation.npz"
    np.savez(str(out_path), **{f"{k}_{m}": v for k, r in results.items() for m, v in r.items()})
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
