#!/usr/bin/env python3
"""Architecture ablation: DenseNet-121 (random init) + same FPN adapter.

Purpose: Verify if the 71pp advantage of ELANNet over ResNet comes from
"multi-path architecture" or is ELANNet-specific.

If DenseNet random >> ResNet random → multi-path is the key factor
If DenseNet random ≈ ResNet random → ELANNet has unique properties

DenseNet-121 output: [256, 512, 1024, 1024] — identical to ELANNet!
→ Uses exact same FPN adapter, no channel alignment needed.
"""
import argparse
import sys
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import timm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mtloc_model import MTLocNet, FPNAdapter
from maploc.models.voting import TemplateSampler
from omegaconf import read_write
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


class DenseNetFrozenEncoder(nn.Module):
    """Frozen DenseNet-121 + same FPN adapter as MTLoc."""

    def __init__(self, latent_dim=128, pretrained=False):
        super().__init__()
        # DenseNet-121 with multi-scale output
        self.backbone = timm.create_model(
            'densenet121', pretrained=pretrained,
            features_only=True, out_indices=(1, 2, 3, 4)
        )

        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Same FPN adapter as MTLoc — channels match exactly [256,512,1024,1024]
        self.adapter = FPNAdapter(
            in_channels_list=[256, 512, 1024, 1024],
            out_channels=latent_dim
        )
        self.scales = [4]

    def forward(self, data):
        image = data["image"]
        self.backbone.eval()
        with torch.no_grad():
            features = self.backbone(image)
        multi_scale = OrderedDict([
            ("c2", features[0]), ("c3", features[1]),
            ("c4", features[2]), ("c5", features[3])
        ])
        adapted = self.adapter(multi_scale)
        return {"feature_maps": [adapted]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=500000)
    parser.add_argument("--val_every", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data_dir", default="/workspace/kitti")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--pretrained", action="store_true",
                        help="Use ImageNet pretrained DenseNet (default: random init)")
    args = parser.parse_args()

    if args.output_dir is None:
        suffix = "pretrained" if args.pretrained else "random"
        args.output_dir = str(REPO_ROOT / f"runs/control_densenet121_{suffix}")

    from mtloc_model import create_mtloc_model
    from scripts.train_kitti import TrainingModule, create_datamodule, apply_strategy

    # Create base model for decoder weights
    model = create_mtloc_model(
        orienternet_ckpt_path=str(REPO_ROOT / "checkpoints/orienternet_mgl.ckpt"),
        yolopx_weights_path=str(REPO_ROOT / "checkpoints/epoch-195.pth"),
        adapter_type="fpn",
    )

    # Replace image encoder with DenseNet
    model.image_encoder = DenseNetFrozenEncoder(
        latent_dim=128, pretrained=args.pretrained
    )

    # Strategy C: freeze backbone, train adapter + decoder + map encoder
    for p in model.parameters():
        p.requires_grad = False
    for p in model.image_encoder.adapter.parameters():
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
    print(f"[DenseNet-121 {'pretrained' if args.pretrained else 'random'}]")
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lit_module = TrainingModule(model, lr=args.lr)
    dm = create_datamodule(args.data_dir, args.batch_size)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            filename="densenet121-step{step:06d}-valloss{loss/total/val:.4f}",
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
