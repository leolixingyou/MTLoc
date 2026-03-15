#!/usr/bin/env python3
"""
MTLoc: Multi-Task Localization Model

Combines a frozen YOLOPX (ELANNet) multi-task backbone with OrienterNet's
localization decoder via a lightweight FPN adapter.

Components:
    - ELANNet backbone (frozen): extracts multi-scale features C2-C5
    - FPNAdapter (~509K params): bridges YOLOPX features to OrienterNet latent space
    - OrienterNet decoder: BEV projection, map encoding, template matching

Usage:
    from mtloc_model import create_mtloc_model

    model = create_mtloc_model(
        orienternet_ckpt_path="checkpoints/orienternet_mgl.ckpt",
        yolopx_weights_path="checkpoints/epoch-195.pth",
        adapter_type="fpn",
    )
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Ensure repo root is in path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maploc.models.base import BaseModel
from maploc.models.bev_net import BEVNet
from maploc.models.bev_projection import CartesianProjection, PolarProjectionDepth
from maploc.models.map_encoder import MapEncoder
from maploc.models.voting import (
    TemplateSampler,
    argmax_xyr,
    conv2d_fft_batchwise,
    expectation_xyr,
    log_softmax_spatial,
    mask_yaw_prior,
)
from maploc.models.metrics import AngleError, AngleRecall, Location2DError, Location2DRecall


# ---------------------------------------------------------------------------
# Adapter modules
# ---------------------------------------------------------------------------

class FeatureAdapter(nn.Module):
    """Simple 2-layer Conv3x3 adapter (C2 only, 256 -> 128). ~885K params."""

    def __init__(self, in_channels: int = 256, out_channels: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.normalize(x, p=2, dim=1)


class FPNAdapter(nn.Module):
    """
    FPN-style adapter aggregating multi-scale YOLOPX features (C2-C5).

    Reuses OrienterNet's FPN architecture (top-down with lateral 1x1 convs).
    ~509K params (lighter than the simple adapter).
    """

    def __init__(self, in_channels_list=None, out_channels=128):
        super().__init__()
        if in_channels_list is None:
            in_channels_list = [256, 512, 1024, 1024]  # C2, C3, C4, C5
        from maploc.models.feature_extractor_v2 import FPN
        self.fpn = FPN(in_channels_list, out_channels)

    def forward(self, features):
        """features: OrderedDict with keys 'c2','c3','c4','c5'."""
        out = self.fpn(features)
        return F.normalize(out, p=2, dim=1)


# ---------------------------------------------------------------------------
# MTLoc image encoder
# ---------------------------------------------------------------------------

class MTLocImageEncoder(nn.Module):
    """
    Image encoder that uses frozen YOLOPX ELANNet backbone + adapter.

    Supports:
      - "simple": Uses only C2 features (FeatureAdapter)
      - "fpn":    Uses C2-C5 multi-scale features (FPNAdapter, recommended)
    """

    def __init__(
        self,
        adapter_in_channels: int = 256,
        latent_dim: int = 128,
        yolopx_weights: Optional[str] = None,
        adapter_type: str = "fpn",
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.adapter_type = adapter_type
        self.freeze_backbone = freeze_backbone

        if adapter_type == "simple":
            self.adapter = FeatureAdapter(adapter_in_channels, latent_dim)
        elif adapter_type == "fpn":
            self.adapter = FPNAdapter([256, 512, 1024, 1024], latent_dim)
        else:
            raise ValueError(f"Unknown adapter_type: {adapter_type}")

        self.backbone = None
        if yolopx_weights is not None:
            from lib.models.common import ELANNet
            self.backbone = ELANNet(use_C2=True)
            self._load_yolopx_weights(yolopx_weights)

        self.scales = [4]  # C2 features are at 1/4 resolution

    def _load_yolopx_weights(self, weights_path: str):
        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        backbone_sd = {
            k.replace("model.0.", ""): v
            for k, v in state_dict.items()
            if k.startswith("model.0.")
        }
        self.backbone.load_state_dict(backbone_sd, strict=False)
        print(f"Loaded YOLOPX backbone from {weights_path}")

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.adapter_type == "fpn":
            if self.backbone is None or "image" not in data:
                raise ValueError("FPN adapter requires backbone with 'image' input")
            image = data["image"]
            if self.freeze_backbone:
                self.backbone.eval()
                with torch.no_grad():
                    c2, c3, c4, c5 = self.backbone(image)
            else:
                c2, c3, c4, c5 = self.backbone(image)
            multi_scale = OrderedDict([
                ("c2", c2), ("c3", c3), ("c4", c4), ("c5", c5)
            ])
            adapted = self.adapter(multi_scale)
        else:
            if "yolopx_features" in data:
                features = data["yolopx_features"]
            elif self.backbone is not None and "image" in data:
                image = data["image"]
                if self.freeze_backbone:
                    self.backbone.eval()
                    with torch.no_grad():
                        c2, c3, c4, c5 = self.backbone(image)
                else:
                    c2, c3, c4, c5 = self.backbone(image)
                features = c2
            else:
                raise ValueError("Need 'yolopx_features' or 'image' with backbone")
            adapted = self.adapter(features)

        return {"feature_maps": [adapted]}


# ---------------------------------------------------------------------------
# MTLocNet (main model)
# ---------------------------------------------------------------------------

class MTLocNet(BaseModel):
    """
    MTLoc model: YOLOPX backbone + adapter + OrienterNet decoder.

    Inherits OrienterNet's architecture but replaces the image encoder
    with MTLocImageEncoder (frozen YOLOPX + lightweight adapter).
    """

    default_conf = {
        "adapter_in_channels": 256,
        "adapter_type": "fpn",
        "yolopx_weights": None,
        "freeze_backbone": True,
        "map_encoder": "???",
        "bev_net": "???",
        "latent_dim": 128,
        "matching_dim": 8,
        "scale_range": [1, 9],
        "num_scale_bins": 33,
        "z_min": None,
        "z_max": 32,
        "x_max": 32,
        "pixel_per_meter": 2,
        "num_rotations": 64,
        "add_temperature": False,
        "normalize_features": False,
        "padding_matching": "replicate",
        "apply_map_prior": True,
        "do_label_smoothing": False,
        "sigma_xy": 1,
        "sigma_r": 2,
        "depth_parameterization": "scale",
        "norm_depth_scores": False,
        "normalize_scores_by_dim": False,
        "normalize_scores_by_num_valid": True,
        "prior_renorm": True,
        "retrieval_dim": None,
    }

    def _init(self, conf):
        self.image_encoder = MTLocImageEncoder(
            adapter_in_channels=conf.adapter_in_channels,
            latent_dim=conf.latent_dim,
            yolopx_weights=conf.yolopx_weights,
            adapter_type=conf.adapter_type,
            freeze_backbone=conf.freeze_backbone,
        )
        self.map_encoder = MapEncoder(conf.map_encoder)
        self.bev_net = None if conf.bev_net is None else BEVNet(conf.bev_net)

        ppm = conf.pixel_per_meter
        self.projection_polar = PolarProjectionDepth(
            conf.z_max, ppm, conf.scale_range, conf.z_min,
        )
        self.projection_bev = CartesianProjection(
            conf.z_max, conf.x_max, ppm, conf.z_min,
        )
        self.template_sampler = TemplateSampler(
            self.projection_bev.grid_xz, ppm, conf.num_rotations,
        )
        self.scale_classifier = nn.Linear(conf.latent_dim, conf.num_scale_bins)

        if conf.bev_net is None:
            self.feature_projection = nn.Linear(conf.latent_dim, conf.matching_dim)
        if conf.add_temperature:
            self.register_parameter(
                "temperature", nn.Parameter(torch.tensor(0.0))
            )

    def exhaustive_voting(self, f_bev, f_map, valid_bev, confidence_bev=None):
        if self.conf.normalize_features:
            f_bev = F.normalize(f_bev, dim=1)
            f_map = F.normalize(f_map, dim=1)
        if confidence_bev is not None:
            f_bev = f_bev * confidence_bev.unsqueeze(1)
        f_bev = f_bev.masked_fill(~valid_bev.unsqueeze(1), 0.0)
        templates = self.template_sampler(f_bev)

        with torch.autocast("cuda", enabled=False):
            scores = conv2d_fft_batchwise(
                f_map.float(), templates.float(),
                padding_mode=self.conf.padding_matching,
            )
        if self.conf.add_temperature:
            scores = scores * torch.exp(self.temperature)

        valid_templates = self.template_sampler(valid_bev.float()[None]) > (1 - 1e-4)
        num_valid = valid_templates.float().sum((-3, -2, -1))
        scores = scores / num_valid[..., None, None]
        return scores

    def _forward(self, data):
        pred = {}
        pred_map = pred["map"] = self.map_encoder(data)
        f_map = pred_map["map_features"][0]

        level = 0
        encoder_output = self.image_encoder(data)
        f_image = encoder_output["feature_maps"][level]

        camera = data["camera"].scale(1 / self.image_encoder.scales[level])
        camera = camera.to(f_image.device, non_blocking=True)

        pred["pixel_scales"] = scales = self.scale_classifier(f_image.moveaxis(1, -1))
        f_polar = self.projection_polar(f_image, scales, camera)

        with torch.autocast("cuda", enabled=False):
            f_bev, valid_bev, _ = self.projection_bev(
                f_polar.float(), None, camera.float()
            )

        pred_bev = {}
        if self.conf.bev_net is None:
            f_bev = self.feature_projection(f_bev.moveaxis(1, -1)).moveaxis(-1, 1)
        else:
            pred_bev = pred["bev"] = self.bev_net({"input": f_bev})
            f_bev = pred_bev["output"]

        scores = self.exhaustive_voting(
            f_bev, f_map, valid_bev, pred_bev.get("confidence")
        )
        scores = scores.moveaxis(1, -1)

        if "log_prior" in pred_map and self.conf.apply_map_prior:
            scores = scores + pred_map["log_prior"][0].unsqueeze(-1)
        if "map_mask" in data:
            scores.masked_fill_(~data["map_mask"][..., None], -np.inf)
        if "yaw_prior" in data:
            mask_yaw_prior(scores, data["yaw_prior"], self.conf.num_rotations)

        log_probs = log_softmax_spatial(scores)
        with torch.no_grad():
            uvr_max = argmax_xyr(scores).to(scores)
            uvr_avg, _ = expectation_xyr(log_probs.exp())

        return {
            **pred,
            "scores": scores,
            "log_probs": log_probs,
            "uvr_max": uvr_max,
            "uv_max": uvr_max[..., :2],
            "yaw_max": uvr_max[..., 2],
            "uvr_expectation": uvr_avg,
            "uv_expectation": uvr_avg[..., :2],
            "yaw_expectation": uvr_avg[..., 2],
            "features_image": f_image,
            "features_bev": f_bev,
            "valid_bev": valid_bev.squeeze(1),
        }

    def loss(self, pred, data):
        from maploc.models.voting import nll_loss_xyr, nll_loss_xyr_smoothed

        xy_gt = data["uv"]
        yaw_gt = data["roll_pitch_yaw"][..., -1]

        if self.conf.do_label_smoothing:
            nll = nll_loss_xyr_smoothed(
                pred["log_probs"], xy_gt, yaw_gt,
                self.conf.sigma_xy / self.conf.pixel_per_meter,
                self.conf.sigma_r,
                mask=data.get("map_mask"),
            )
        else:
            nll = nll_loss_xyr(pred["log_probs"], xy_gt, yaw_gt)

        loss = {"total": nll, "nll": nll}
        if self.training and self.conf.add_temperature:
            loss["temperature"] = self.temperature.expand(len(nll))
        return loss

    def metrics(self):
        ppm = self.conf.pixel_per_meter
        return {
            "xy_max_error": Location2DError("uv_max", ppm),
            "xy_expectation_error": Location2DError("uv_expectation", ppm),
            "yaw_max_error": AngleError("yaw_max"),
            "xy_recall_2m": Location2DRecall(2.0, ppm, "uv_max"),
            "xy_recall_5m": Location2DRecall(5.0, ppm, "uv_max"),
            "yaw_recall_2": AngleRecall(2.0, "yaw_max"),
            "yaw_recall_5": AngleRecall(5.0, "yaw_max"),
        }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_orienternet_checkpoint(ckpt_path: str) -> dict:
    """Load OrienterNet checkpoint and extract component state dicts."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_config = ckpt["hyper_parameters"]["model"]
    state_dict = ckpt["state_dict"]

    map_encoder_state, bev_net_state, other_state = {}, {}, {}
    for key, value in state_dict.items():
        if key.startswith("model.map_encoder."):
            map_encoder_state[key.replace("model.map_encoder.", "")] = value
        elif key.startswith("model.bev_net."):
            bev_net_state[key.replace("model.bev_net.", "")] = value
        elif not key.startswith("model.image_encoder."):
            other_state[key] = value

    return {
        "model_config": model_config,
        "state_dict": state_dict,
        "map_encoder_state": map_encoder_state,
        "bev_net_state": bev_net_state,
        "other_state": other_state,
    }


def create_mtloc_model(
    orienternet_ckpt_path: str,
    yolopx_weights_path: Optional[str] = None,
    adapter_in_channels: int = 256,
    load_pretrained: bool = True,
    adapter_type: str = "fpn",
    freeze_backbone: bool = True,
) -> MTLocNet:
    """
    Create MTLocNet with pretrained OrienterNet weights.

    Args:
        orienternet_ckpt_path: Path to OrienterNet MGL checkpoint
        yolopx_weights_path:  Path to YOLOPX weights (epoch-195.pth)
        adapter_in_channels:  YOLOPX C2 channels (256)
        load_pretrained:      Load pretrained map_encoder/bev_net weights
        adapter_type:         "fpn" (recommended) or "simple"
        freeze_backbone:      Freeze ELANNet backbone (default: True)
    """
    from omegaconf import OmegaConf

    ckpt_data = load_orienternet_checkpoint(orienternet_ckpt_path)
    mc = ckpt_data["model_config"]

    conf = OmegaConf.create({
        "adapter_in_channels": adapter_in_channels,
        "adapter_type": adapter_type,
        "yolopx_weights": yolopx_weights_path,
        "freeze_backbone": freeze_backbone,
        "latent_dim": mc["latent_dim"],
        "matching_dim": mc["matching_dim"],
        "z_max": mc["z_max"],
        "x_max": mc["x_max"],
        "pixel_per_meter": mc["pixel_per_meter"],
        "scale_range": mc["scale_range"],
        "num_scale_bins": mc["num_scale_bins"],
        "num_rotations": mc["num_rotations"],
        "padding_matching": mc["padding_matching"],
        "normalize_features": mc["normalize_features"],
        "normalize_scores_by_num_valid": mc["normalize_scores_by_num_valid"],
        "map_encoder": mc["map_encoder"],
        "bev_net": mc["bev_net"],
    })

    model = MTLocNet(conf)

    if load_pretrained:
        model.map_encoder.load_state_dict(ckpt_data["map_encoder_state"], strict=False)
        if model.bev_net is not None:
            model.bev_net.load_state_dict(ckpt_data["bev_net_state"], strict=False)
        scale_cls_state = {
            k.replace("model.scale_classifier.", ""): v
            for k, v in ckpt_data["state_dict"].items()
            if k.startswith("model.scale_classifier.")
        }
        if scale_cls_state:
            model.scale_classifier.load_state_dict(scale_cls_state)
        print("Loaded pretrained OrienterNet weights (map_encoder, bev_net, scale_classifier)")

    return model
