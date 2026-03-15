#!/usr/bin/env python3
"""
Evaluate YOLOPX perception (DA/LL/Det) with backbone from MTLoc checkpoint.

Verifies that freezing the backbone preserves multi-task perception performance.

Usage:
    # Standalone YOLOPX baseline
    python scripts/eval_bdd100k.py --mode standalone --gpu 0

    # With MTLoc backbone (should match standalone if backbone is frozen)
    python scripts/eval_bdd100k.py --mode mtloc --ckpt checkpoints/mtloc_445k.ckpt --gpu 0
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch
import torchvision.transforms as transforms

import lib.dataset as dataset
from lib.config import cfg, update_config
from lib.core.function import validate
from lib.core.loss import get_loss
from lib.models import get_net
from lib.utils import DataLoaderX
from lib.utils.utils import create_logger, select_device


def extract_backbone_from_mtloc(ckpt_path):
    """Extract ELANNet backbone weights from MTLoc checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    prefix_in = "model.image_encoder.backbone."
    prefix_out = "model.0."
    backbone_sd = {
        prefix_out + k[len(prefix_in):]: v
        for k, v in sd.items() if k.startswith(prefix_in)
    }
    print(f"  Extracted {len(backbone_sd)} backbone keys")
    return backbone_sd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["standalone", "mtloc"], required=True)
    parser.add_argument("--ckpt", default=None, help="MTLoc checkpoint (for mtloc mode)")
    parser.add_argument("--yolopx_weights", default=os.path.join(REPO_ROOT, "checkpoints/epoch-195.pth"))
    parser.add_argument("--bdd_root", default="data/bdd100k")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    if args.mode == "mtloc" and args.ckpt is None:
        parser.error("--ckpt required for mtloc mode")

    # Config
    cfg.DATASET.DATAROOT = os.path.join(args.bdd_root, "images")
    cfg.DATASET.LABELROOT = os.path.join(args.bdd_root, "bdd_det")
    cfg.DATASET.MASKROOT = os.path.join(args.bdd_root, "bdd_seg_gt")
    cfg.DATASET.LANEROOT = os.path.join(args.bdd_root, "bdd_lane_gt")
    cfg.GPUS = (args.gpu,)
    cfg.TEST.BATCH_SIZE_PER_GPU = args.batch_size
    cfg.WORKERS = 4
    cfg.LOG_DIR = "/tmp/mtloc_eval_logs"
    os.makedirs(cfg.LOG_DIR, exist_ok=True)

    class DummyArgs:
        modelDir = ""
        logDir = cfg.LOG_DIR
    update_config(cfg, DummyArgs())

    logger, final_output_dir, tb_log_dir = create_logger(cfg, cfg.LOG_DIR, "eval_bdd")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")
    model = get_net(cfg)

    # Load weights
    model_dict = model.state_dict()
    ckpt = torch.load(args.yolopx_weights, map_location="cpu")
    model_dict.update(ckpt["state_dict"])

    if args.mode == "mtloc":
        backbone_sd = extract_backbone_from_mtloc(args.ckpt)
        model_dict.update(backbone_sd)

    model.load_state_dict(model_dict)
    model = model.to(device)
    model.gr = 1.0
    model.nc = 1

    criterion = get_loss(cfg, device, model)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    valid_dataset = eval("dataset." + cfg.DATASET.DATASET)(
        cfg=cfg, is_train=False, inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
    )
    valid_loader = DataLoaderX(
        valid_dataset, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False, num_workers=cfg.WORKERS, pin_memory=False,
        collate_fn=dataset.AutoDriveDataset.collate_fn,
    )

    from tensorboardX import SummaryWriter
    writer_dict = {
        "writer": SummaryWriter(log_dir=tb_log_dir),
        "train_global_steps": 0, "valid_global_steps": 0,
    }

    da_seg, ll_seg, det, total_loss, maps, times = validate(
        0, cfg, valid_loader, valid_dataset, model, criterion,
        final_output_dir, tb_log_dir, logger, device,
    )

    label = "YOLOPX standalone" if args.mode == "standalone" else f"MTLoc backbone"
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  DA Seg:  Acc={da_seg[0]:.4f}  IoU={da_seg[1]:.4f}  mIoU={da_seg[2]:.4f}")
    print(f"  LL Seg:  Acc={ll_seg[0]:.4f}  IoU={ll_seg[1]:.4f}  mIoU={ll_seg[2]:.4f}")
    print(f"  Det:     P={det[0]:.4f}  R={det[1]:.4f}  mAP50={det[2]:.4f}  mAP50-95={det[3]:.4f}")

    writer_dict["writer"].close()


if __name__ == "__main__":
    main()
