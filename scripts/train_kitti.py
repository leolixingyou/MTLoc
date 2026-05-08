#!/usr/bin/env python3
"""
Train YOLOPX-Loc (formerly MTLoc) on KITTI dataset (in-domain training).

Strategies (per Table 1 in paper):
  A = Stage-1: only FPN adapter (~0.51M)
  B = Stage-2: + scale classifier + BEV proj + BEV matching (~0.6M);
              backbone + map_encoder frozen
  C = Stage-3-F: all downstream + map encoder trainable (~12M);
                 backbone frozen (⭐ paper main)
  D = Stage-3-UF: all trainable including backbone (~25.7M)

Usage:
    # Strategy C (paper main, frozen backbone)
    python scripts/train_kitti.py --gpu 0 --strategy C \
        --output_dir runs/kitti_C_<ts>

    # Resume
    python scripts/train_kitti.py --gpu 0 --strategy C \
        --resume runs/kitti_C_<ts>/checkpoints/last.ckpt
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
from maploc.data.kitti.dataset import KittiDataModule
from maploc.models.voting import TemplateSampler
from maploc.models.metrics import AngleError, AngleRecall, Location2DError, Location2DRecall


# ---------------------------------------------------------------------------
# Lightning Module (same as train_adapter.py — reused via copy to avoid import cycle)
# ---------------------------------------------------------------------------

class AverageKeyMeter(MeanMetric):
    def __init__(self, key, *args, **kwargs):
        self.key = key
        super().__init__(*args, **kwargs)

    def update(self, dict):
        value = dict[self.key]
        value = value[torch.isfinite(value)]
        return super().update(value)


class TrainingModule(pl.LightningModule):
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
# Strategy-aware model construction
# ---------------------------------------------------------------------------

STRATEGY_INFO = {
    "A": {"name": "Stage-1 (Adapter-only)", "freeze_backbone": True},
    "B": {"name": "Stage-2 (Encoder-Decoder)", "freeze_backbone": True},
    "C": {"name": "Stage-3-F (All downstream + map encoder, frozen backbone)", "freeze_backbone": True},
    "D": {"name": "Stage-3-UF (All trainable, unfrozen backbone)", "freeze_backbone": False},
}


def apply_strategy(model, strategy):
    """Set parameter trainability per strategy A/B/C/D.

    A: only image_encoder.adapter trainable
    B: + scale_classifier + projection_polar + projection_bev + bev_net (if exists)
    C: + map_encoder
    D: all trainable
    """
    # Reset adapter weights for fair comparison across strategies (per Paper 1)
    for m in model.image_encoder.adapter.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    if strategy == "D":
        # All trainable (Stage-3-UF)
        for p in model.parameters():
            p.requires_grad = True
        # Note: backbone gradients flow → BN running stats drift (per Paper 1)
        # Do NOT call model.image_encoder.backbone.eval() in this strategy
    else:
        # First, freeze everything
        for p in model.parameters():
            p.requires_grad = False

        # Then unfreeze per strategy
        if strategy in ("A", "B", "C"):
            # All strategies train adapter
            for n, p in model.named_parameters():
                if "image_encoder.adapter" in n:
                    p.requires_grad = True

        if strategy in ("B", "C"):
            # B + C: scale_classifier + projection + bev_net
            for n, p in model.named_parameters():
                if any(k in n for k in [
                    "scale_classifier",
                    "feature_projection",
                    "bev_net",
                    "projection_polar",
                    "projection_bev",
                ]):
                    p.requires_grad = True

        if strategy == "C":
            # C also trains map encoder
            for n, p in model.named_parameters():
                if "map_encoder" in n:
                    p.requires_grad = True

    # Statistics
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    info = STRATEGY_INFO[strategy]
    print(f"[Strategy {strategy}] {info['name']}")
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
    print(f"  Backbone frozen: {info['freeze_backbone']}")
    return model


def build_model(args):
    model = create_mtloc_model(
        orienternet_ckpt_path=args.ckpt_path,
        yolopx_weights_path=args.yolopx_weights,
        adapter_type=args.adapter_type,
        freeze_backbone=STRATEGY_INFO[args.strategy]["freeze_backbone"],
        semantic_align_lambda=getattr(args, "semantic_align_lambda", 0.0),
    )

    with read_write(model.conf):
        model.conf.num_rotations = 64
        model.conf.semantic_align_lambda = args.semantic_align_lambda
    model.template_sampler = TemplateSampler(
        model.projection_bev.grid_xz, model.conf.pixel_per_meter, 64,
    )

    apply_strategy(model, args.strategy)
    return model


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def create_datamodule(data_dir, batch_size=4):
    """KITTI DataModule with sensible defaults from kitti/dataset.py."""
    cfg = OmegaConf.create({
        "data_dir": str(Path(data_dir)),
        "tiles_filename": "tiles.pkl",
        "loading": {
            "train": {"batch_size": batch_size, "num_workers": batch_size},
            "val":   {"batch_size": 1, "num_workers": 2},
            "test":  {"batch_size": 1, "num_workers": 2},
        },
        "max_num_val": 500,
        "skip_frames": 1,
        "camera_index": 2,
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
    return KittiDataModule(cfg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train YOLOPX-Loc on KITTI")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--strategy", required=True, choices=["A", "B", "C", "D"],
                        help="A=Stage-1, B=Stage-2, C=Stage-3-F (paper main), D=Stage-3-UF")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=200000)
    parser.add_argument("--val_every", type=int, default=5000)
    parser.add_argument("--adapter_type", default="fpn", choices=["simple", "fpn"])
    parser.add_argument("--semantic_align_lambda", type=float, default=0.0,
                        help="D.1 semantic-alignment aux loss weight (OSMLoc-B Eq.6); 0=off, paper uses 20.0")
    parser.add_argument("--ckpt_path", default=str(REPO_ROOT / "checkpoints/orienternet_mgl.ckpt"))
    parser.add_argument("--yolopx_weights", default=str(REPO_ROOT / "checkpoints/epoch-195.pth"))
    parser.add_argument("--data_dir", default="/workspace/kitti")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(REPO_ROOT / f"runs/kitti_{args.strategy}_{ts}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")
    print(f"Dataset: KITTI ({args.data_dir})")

    model = build_model(args)
    lit_module = TrainingModule(model, lr=args.lr)
    dm = create_datamodule(args.data_dir, args.batch_size)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            filename=f"yolopx_loc_{args.strategy}_kitti-step{{step:06d}}-valloss{{loss/total/val:.4f}}",
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
        check_val_every_n_epoch=None,  # allow val_check_interval > batches per epoch (KITTI ~4628)
        limit_val_batches=500,
        log_every_n_steps=100,
        default_root_dir=str(output_dir),
        callbacks=callbacks,
    )

    trainer.fit(lit_module, datamodule=dm, ckpt_path=args.resume)
    print(f"\nBest checkpoint: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
