#!/usr/bin/env python3
"""
Train YOLOPX-Loc on nuScenes dataset (in-domain training).

Strategies (per Table 1 in paper) — same as train_kitti.py:
  A = Stage-1 (Adapter-only, ~0.51M)
  B = Stage-2 (Encoder-Decoder, ~0.6M)
  C = Stage-3-F (frozen backbone, ~12M) ⭐ paper main
  D = Stage-3-UF (all unfrozen, ~25.7M)

Usage:
    python scripts/train_nuscenes.py --gpu 1 --strategy C
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from omegaconf import OmegaConf, read_write
from torchmetrics import MeanMetric, MetricCollection

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mtloc_model import create_mtloc_model
from maploc.data.nuscenes.dataset import NuScenesDataModule
from maploc.models.voting import TemplateSampler
from maploc.models.metrics import AngleError, AngleRecall, Location2DError, Location2DRecall

# Reuse strategy logic from train_kitti.py
from scripts.train_kitti import (
    TrainingModule, AverageKeyMeter, STRATEGY_INFO, apply_strategy
)


def build_model(args):
    model = create_mtloc_model(
        orienternet_ckpt_path=args.ckpt_path,
        yolopx_weights_path=args.yolopx_weights,
        adapter_type=args.adapter_type,
        freeze_backbone=STRATEGY_INFO[args.strategy]["freeze_backbone"],
    )
    with read_write(model.conf):
        model.conf.num_rotations = 64
    model.template_sampler = TemplateSampler(
        model.projection_bev.grid_xz, model.conf.pixel_per_meter, 64,
    )
    apply_strategy(model, args.strategy)
    return model


def create_datamodule(data_dir, batch_size=4, version="v1.0-trainval"):
    """nuScenes DataModule with sensible defaults."""
    cfg = OmegaConf.create({
        "data_dir": str(Path(data_dir)),
        "tiles_dir": str(Path(data_dir) / "osm_tiles_trainval"),
        "version": version,
        "camera": "CAM_FRONT",
        "loading": {
            "train": {"batch_size": batch_size, "num_workers": batch_size},
            "val":   {"batch_size": 1, "num_workers": 2},
            "test":  {"batch_size": 1, "num_workers": 2},
        },
        "crop_size_meters": 64,
        "max_init_error": 20,
        "max_init_error_rotation": 10,
        "add_map_mask": True,
        "target_focal_length": 256,
        "num_classes": {"areas": 7, "ways": 10, "nodes": 33},
        "pixel_per_meter": 2,
        "augmentation": {
            "rot90": True, "flip": True,
            "image": {"apply": True, "brightness": 0.5, "contrast": 0.4,
                      "saturation": 0.4, "hue": 0.5 / 3.14},
        },
    })
    return NuScenesDataModule(cfg)


def main():
    parser = argparse.ArgumentParser(description="Train YOLOPX-Loc on nuScenes")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--strategy", required=True, choices=["A", "B", "C", "D"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=200000)
    parser.add_argument("--val_every", type=int, default=5000)
    parser.add_argument("--adapter_type", default="fpn", choices=["simple", "fpn"])
    parser.add_argument("--ckpt_path", default=str(REPO_ROOT / "checkpoints/orienternet_mgl.ckpt"))
    parser.add_argument("--yolopx_weights", default=str(REPO_ROOT / "checkpoints/epoch-195.pth"))
    parser.add_argument("--data_dir", default="/workspace/datasets/nuscenes")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(REPO_ROOT / f"runs/nuscenes_{args.strategy}_{ts}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")
    print(f"Dataset: nuScenes {args.version} ({args.data_dir})")

    model = build_model(args)
    lit_module = TrainingModule(model, lr=args.lr)
    dm = create_datamodule(args.data_dir, args.batch_size, args.version)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            filename=f"yolopx_loc_{args.strategy}_nuscenes-step{{step:06d}}-valloss{{loss/total/val:.4f}}",
            auto_insert_metric_name=False,
            monitor="loss/total/val", mode="min",
            save_top_k=5, save_last=True,
            every_n_train_steps=args.val_every,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        accelerator="gpu", devices=[args.gpu],
        max_steps=args.max_steps,
        val_check_interval=args.val_every,
        check_val_every_n_epoch=None,  # allow val_check_interval > batches per epoch
        limit_val_batches=500,
        log_every_n_steps=100,
        default_root_dir=str(output_dir),
        callbacks=callbacks,
    )

    trainer.fit(lit_module, datamodule=dm, ckpt_path=args.resume)
    print(f"\nBest checkpoint: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
