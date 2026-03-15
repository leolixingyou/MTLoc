#!/usr/bin/env python3
"""
Evaluate MTLoc on MGL (Mapillary Geo-Localization) validation set.

Usage:
    python scripts/eval_mgl.py \
        --model mtloc --adapter_type fpn \
        --adapter_ckpt checkpoints/mtloc_445k.ckpt
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import torch
from tqdm import tqdm
from omegaconf import OmegaConf, read_write

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mtloc_model import create_mtloc_model
from maploc.data import MapillaryDataModule
from maploc.module import GenericModule
from maploc.models.voting import TemplateSampler

AVAILABLE_CITIES = [
    "avignon", "berlin", "helsinki", "lemans", "milan",
    "montrouge", "nantes", "paris", "sanfrancisco_hayes",
    "toulouse", "vilnius",
]


def load_model(args):
    """Load OrienterNet baseline or MTLocNet."""
    if args.model == "orienternet":
        cfg = OmegaConf.create({"model": {"num_rotations": args.num_rotations}})
        module = GenericModule.load_from_checkpoint(
            Path(args.ckpt_path), cfg=cfg, find_best=False,
        )
        return module.eval().to(args.device), True

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

    if args.adapter_ckpt:
        ckpt = torch.load(args.adapter_ckpt, map_location="cpu", weights_only=False)
        sd = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()
              if k.startswith("model.")}
        model.load_state_dict(sd, strict=True)

    return model.eval().to(args.device), False


def evaluate(module, dataloader, device, is_generic):
    from torchmetrics import MetricCollection
    from maploc.models.metrics import Location2DError, AngleError, LateralLongitudinalError

    model = module.model if is_generic else module
    metrics = MetricCollection(model.metrics())
    metrics["directional_error"] = LateralLongitudinalError(model.conf.pixel_per_meter)
    metrics = metrics.to(device)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="MGL eval")):
            if is_generic:
                batch = module.transfer_batch_to_device(batch, device, i)
                pred = module(batch)
            else:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                pred = model(batch)
            metrics(pred, batch)

    return {k: v.item() if v.numel() == 1 else v.tolist()
            for k, v in metrics.compute().items()}


def main():
    parser = argparse.ArgumentParser(description="Evaluate on MGL")
    parser.add_argument("--model", choices=["orienternet", "mtloc"], default="mtloc")
    parser.add_argument("--ckpt_path", default=str(REPO_ROOT / "checkpoints/orienternet_mgl.ckpt"))
    parser.add_argument("--yolopx_weights", default=str(REPO_ROOT / "checkpoints/epoch-195.pth"))
    parser.add_argument("--adapter_ckpt", default=None)
    parser.add_argument("--adapter_type", default="fpn", choices=["simple", "fpn"])
    parser.add_argument("--data_dir", default=str(REPO_ROOT / "data/MGL"))
    parser.add_argument("--num_rotations", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    module, is_generic = load_model(args)

    cfg = OmegaConf.create({
        "name": "mapillary", "data_dir": args.data_dir,
        "scenes": AVAILABLE_CITIES, "split": "splits_MGL_13loc.json",
        "tiles_filename": "tiles.pkl",
        "loading": {"val": {"batch_size": 1, "num_workers": 0}},
        "return_gps": True, "add_map_mask": True,
        "max_init_error": 32, "pixel_per_meter": 2,
        "crop_size_meters": 64, "resize_image": 512,
        "pad_to_square": True, "rectify_pitch": True,
        "num_classes": {"areas": 7, "ways": 10, "nodes": 33},
    })
    dm = MapillaryDataModule(cfg)
    dm.prepare_data()
    dm.setup("fit")

    results = evaluate(module, dm.val_dataloader(), args.device, is_generic)

    print(f"\n{'=' * 60}")
    print(f"  MGL Results ({args.model})")
    print(f"{'=' * 60}")
    for name, value in sorted(results.items()):
        if isinstance(value, list):
            print(f"  {name}: {[f'{v*100:.2f}%' for v in value]}")
        else:
            print(f"  {name}: {value*100:.2f}%")


if __name__ == "__main__":
    main()
