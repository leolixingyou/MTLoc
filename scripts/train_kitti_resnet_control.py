#!/usr/bin/env python3
"""Control Experiment: ResNet-50 frozen + same FPN adapter as MTLoc.

Isolates the contribution of backbone (ELANNet vs ResNet-50) from adapter design.
If ResNet+FPN ≈ ELANNet+FPN → improvement is from adapter, not backbone.
If ResNet+FPN < ELANNet+FPN → multi-task features genuinely help.
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from collections import OrderedDict

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mtloc_model import MTLocNet, FPNAdapter
from maploc.data.kitti.dataset import KittiDataModule
from maploc.models.voting import TemplateSampler
from omegaconf import OmegaConf, read_write
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


class ResNet50FrozenEncoder(nn.Module):
    """Frozen ResNet-50 backbone + same FPN adapter as MTLoc."""

    def __init__(self, latent_dim=128, ckpt_path=None):
        super().__init__()
        import torchvision.models as models
        resnet = models.resnet50(pretrained=True)

        # Extract layers for multi-scale features (same as ELANNet C2-C5)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  # 256 ch, stride 4
        self.layer2 = resnet.layer2  # 512 ch, stride 8
        self.layer3 = resnet.layer3  # 1024 ch, stride 16
        self.layer4 = resnet.layer4  # 2048 ch, stride 32

        # Freeze backbone
        for p in self.parameters():
            p.requires_grad = False

        # Channel alignment: ResNet channels [256, 512, 1024, 2048]
        # ELANNet channels: [256, 512, 1024, 1024]
        # Align ResNet layer4 (2048) to 1024 to match ELANNet's C5
        self.channel_align = nn.Conv2d(2048, 1024, 1, bias=False)

        # Same FPN adapter as MTLoc (in_channels_list matches ResNet-50 after channel_align)
        self.adapter = FPNAdapter(in_channels_list=[256, 512, 1024, 1024], out_channels=latent_dim)

        self.scales = [4]  # same as MTLoc

    def forward(self, data):
        image = data["image"]

        # Frozen forward
        self.eval()
        with torch.no_grad():
            x = self.layer0(image)
            c2 = self.layer1(x)    # [B, 256, H/4, W/4]
            c3 = self.layer2(c2)   # [B, 512, H/8, W/8]
            c4 = self.layer3(c3)   # [B, 1024, H/16, W/16]
            c5 = self.layer4(c4)   # [B, 2048, H/32, W/32]

        # Align c5 to 1024 channels (matching ELANNet)
        c5 = self.channel_align(c5)

        multi_scale = OrderedDict([
            ("c2", c2), ("c3", c3), ("c4", c4), ("c5", c5)
        ])
        adapted = self.adapter(multi_scale)
        return {"feature_maps": [adapted]}


def create_resnet_control_model(orienternet_ckpt_path, latent_dim=128):
    """Create MTLocNet but with ResNet-50 backbone instead of ELANNet."""
    from mtloc_model import create_mtloc_model

    # First create a normal MTLoc model to get the decoder weights
    model = create_mtloc_model(
        orienternet_ckpt_path=orienternet_ckpt_path,
        yolopx_weights_path=str(REPO_ROOT / "checkpoints/epoch-195.pth"),
        adapter_type="fpn",
        freeze_backbone=True,
    )

    # Replace image encoder with ResNet-50 version
    model.image_encoder = ResNet50FrozenEncoder(latent_dim=latent_dim)

    return model


# Reuse training infrastructure from train_kitti.py
from scripts.train_kitti import TrainingModule, create_datamodule


def main():
    parser = argparse.ArgumentParser(description="Control: ResNet-50 frozen + FPN adapter")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=500000)
    parser.add_argument("--val_every", type=int, default=5000)
    parser.add_argument("--data_dir", default="/workspace/kitti")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(REPO_ROOT / "runs/control_resnet50_fpn")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create model
    model = create_resnet_control_model(
        str(REPO_ROOT / "checkpoints/orienternet_mgl.ckpt")
    )

    # Strategy C equivalent: freeze backbone, train adapter + decoder
    for p in model.parameters():
        p.requires_grad = False
    for p in model.image_encoder.adapter.parameters():
        p.requires_grad = True
    for p in model.image_encoder.channel_align.parameters():
        p.requires_grad = True
    for n, p in model.named_parameters():
        if any(k in n for k in ["scale_classifier", "bev_net", "map_encoder",
                                 "feature_projection", "projection_polar", "projection_bev"]):
            p.requires_grad = True

    with read_write(model.conf):
        model.conf.num_rotations = 64
    model.template_sampler = TemplateSampler(
        model.projection_bev.grid_xz, model.conf.pixel_per_meter, 64,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Control] ResNet-50 frozen + FPN adapter")
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    lit_module = TrainingModule(model, lr=args.lr)
    dm = create_datamodule(args.data_dir, args.batch_size)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            filename="resnet50_fpn-step{step:06d}-valloss{loss/total/val:.4f}",
            auto_insert_metric_name=False,
            monitor="loss/total/val", mode="min",
            save_top_k=5, save_last=True,
            every_n_train_steps=args.val_every,
        ),
    ]

    trainer = pl.Trainer(
        accelerator="gpu", devices=[args.gpu],
        max_steps=args.max_steps,
        val_check_interval=args.val_every,
        check_val_every_n_epoch=None,
        callbacks=callbacks,
        default_root_dir=str(output_dir),
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    trainer.fit(lit_module, dm)
    print(f"\nBest checkpoint: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
