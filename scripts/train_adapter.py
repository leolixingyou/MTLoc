#!/usr/bin/env python3
"""
Train the MTLoc adapter on Mapillary Geo-Localization (MGL) dataset.

Strategy:
  - Freeze: YOLOPX ELANNet backbone (pretrained)
  - Freeze: OrienterNet decoder (from orienternet_mgl.ckpt)
  - Train:  FPN adapter only (~509K params)

Usage:
    python scripts/train_adapter.py --gpu 0 --adapter_type fpn

    # Resume from checkpoint
    python scripts/train_adapter.py --gpu 0 --resume checkpoints/adapter-step445000.ckpt
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

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mtloc_model import create_mtloc_model
from maploc.data.mapillary.dataset import MapillaryDataModule
from maploc.models.voting import TemplateSampler
from maploc.models.metrics import AngleError, AngleRecall, Location2DError, Location2DRecall


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------

class AverageKeyMeter(MeanMetric):
    def __init__(self, key, *args, **kwargs):
        self.key = key
        super().__init__(*args, **kwargs)

    def update(self, dict):
        value = dict[self.key]
        value = value[torch.isfinite(value)]
        return super().update(value)


class AdapterTrainingModule(pl.LightningModule):
    """Lightning module for training only the adapter."""

    def __init__(self, model, lr=1e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.save_hyperparameters(ignore=["model"])
        self.metrics_val = MetricCollection(self.model.metrics(), prefix="val/")
        self.losses_val = None

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        losses = self.model.loss(pred, batch)
        total_loss = losses["total"].mean()
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return None
        self.log_dict(
            {f"loss/{k}/train": v.mean().detach() for k, v in losses.items()},
            prog_bar=True, rank_zero_only=True,
        )
        return total_loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        losses = self.model.loss(pred, batch)
        if self.losses_val is None:
            self.losses_val = MetricCollection(
                {k: AverageKeyMeter(k).to(self.device) for k in losses},
                prefix="loss/", postfix="/val",
            )
        self.metrics_val(pred, batch)
        self.losses_val.update(losses)

    def on_validation_epoch_end(self):
        if self.losses_val is not None:
            self.log_dict(self.metrics_val.compute(), sync_dist=True)
            losses_dict = {k: v for k, v in self.losses_val.compute().items()
                           if torch.isfinite(v)}
            if losses_dict:
                self.log_dict(losses_dict, sync_dist=True, prog_bar=True)

    def on_validation_epoch_start(self):
        self.losses_val = None

    def configure_optimizers(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.Adam(trainable, lr=self.lr)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(args):
    """Build MTLocNet and freeze non-adapter parameters."""
    model = create_mtloc_model(
        orienternet_ckpt_path=args.ckpt_path,
        yolopx_weights_path=args.yolopx_weights,
        adapter_type=args.adapter_type,
        freeze_backbone=True,
    )

    # Override num_rotations for training
    with read_write(model.conf):
        model.conf.num_rotations = 64
    model.template_sampler = TemplateSampler(
        model.projection_bev.grid_xz, model.conf.pixel_per_meter, 64,
    )

    # Reset adapter to random weights
    for m in model.image_encoder.adapter.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    # Freeze everything except adapter
    trainable_params = 0
    for name, param in model.named_parameters():
        param.requires_grad = "image_encoder.adapter" in name
        if param.requires_grad:
            trainable_params += param.numel()

    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,} | Trainable: {trainable_params:,} "
          f"({trainable_params / total * 100:.1f}%)")
    return model


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def create_datamodule(data_dir, batch_size=4):
    """Create MGL DataModule."""
    mgl_root = Path(data_dir)
    scenes = [
        d.name for d in mgl_root.iterdir()
        if d.is_dir() and (d / "tiles.pkl").exists()
        and (d / "images").exists()
    ]
    if not scenes:
        raise RuntimeError(f"No MGL scenes found in {mgl_root}")
    print(f"MGL scenes: {scenes}")

    cfg = OmegaConf.create({
        "name": "mapillary",
        "data_dir": str(mgl_root),
        "local_dir": None,
        "tiles_filename": "tiles.pkl",
        "scenes": scenes,
        "split": "splits_MGL_13loc.json",
        "loading": {
            "train": {"batch_size": batch_size, "num_workers": batch_size},
            "val": {"batch_size": 1, "num_workers": 2},
        },
        "num_classes": {"areas": 7, "ways": 10, "nodes": 33},
        "pixel_per_meter": 2,
        "crop_size_meters": 64,
        "max_init_error": 48,
        "add_map_mask": True,
        "resize_image": 512,
        "pad_to_square": True,
        "rectify_pitch": True,
        "augmentation": {
            "rot90": True, "flip": True,
            "image": {"apply": True, "brightness": 0.5, "contrast": 0.4,
                      "saturation": 0.4, "hue": 0.5 / 3.14},
        },
        "filter_for": None,
        "filter_by_ground_angle": None,
        "min_num_points": 0,
    })
    return MapillaryDataModule(cfg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train MTLoc adapter")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=500000)
    parser.add_argument("--val_every", type=int, default=5000)
    parser.add_argument("--adapter_type", default="fpn", choices=["simple", "fpn"])
    parser.add_argument("--ckpt_path", default=str(REPO_ROOT / "checkpoints/orienternet_mgl.ckpt"))
    parser.add_argument("--yolopx_weights", default=str(REPO_ROOT / "checkpoints/epoch-195.pth"))
    parser.add_argument("--data_dir", default=str(REPO_ROOT / "data/MGL"))
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(REPO_ROOT / f"runs/adapter_training_{ts}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output: {output_dir}")
    model = build_model(args)
    lit_module = AdapterTrainingModule(model, lr=args.lr)
    dm = create_datamodule(args.data_dir, args.batch_size)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            filename="adapter-step{step:06d}-valloss{loss/total/val:.4f}",
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
        limit_val_batches=1000,
        log_every_n_steps=100,
        default_root_dir=str(output_dir),
        callbacks=callbacks,
    )

    trainer.fit(lit_module, datamodule=dm, ckpt_path=args.resume)
    print(f"\nBest checkpoint: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
