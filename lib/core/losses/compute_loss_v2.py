"""
ComputeLoss for YOLOPv2
Based on YOLOPv2 paper with Hybrid Loss for lane detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.core.general import bbox_iou
from lib.core.postprocess_v1 import build_targets
from lib.core.evaluate import SegmentationMetric


def smooth_BCE(eps=0.1):
    """Label smoothing for BCE targets"""
    return 1.0 - 0.5 * eps, 0.5 * eps


class FocalLoss(nn.Module):
    """Focal loss wrapper around existing loss_fcn"""
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'
        
    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        """
        pred: (N, C, H, W) logits
        target: (N, C, H, W) one-hot encoded targets
        """
        pred = torch.sigmoid(pred)
        
        # Flatten for computation
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice


class HybridLoss(nn.Module):
    """
    Hybrid loss combining Dice loss and Focal loss
    As described in YOLOPv2 paper for lane detection
    """
    def __init__(self, gamma_focal=2.0, alpha_focal=0.25, gamma_weight=1.0):
        super(HybridLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(
            nn.BCEWithLogitsLoss(reduction='mean'),
            gamma=gamma_focal,
            alpha=alpha_focal
        )
        self.gamma_weight = gamma_weight
        
    def forward(self, pred, target):
        """
        pred: (N, C, H, W) logits
        target: (N, C, H, W) targets
        """
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred.view(-1), target.view(-1))
        
        # Hybrid loss as per paper formula
        return dice + self.gamma_weight * focal


class ComputeLossV2(nn.Module):
    """
    Compute loss for YOLOPv2 model
    Key differences from v1:
    - Hybrid loss for lane detection
    - Different loss weights optimized for YOLOPv2
    """
    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        
        # Detection losses (same as v1)
        self.BCEcls = nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor([cfg.LOSS.CLS_POS_WEIGHT])
        ).to(device)
        self.BCEobj = nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor([cfg.LOSS.OBJ_POS_WEIGHT])
        ).to(device)
        
        # Focal loss for detection
        gamma = cfg.LOSS.FL_GAMMA if hasattr(cfg.LOSS, 'FL_GAMMA') else 0
        if gamma > 0:
            self.BCEcls = FocalLoss(self.BCEcls, gamma)
            self.BCEobj = FocalLoss(self.BCEobj, gamma)
        
        # Segmentation losses
        # Drivable area: standard BCE loss
        self.BCEseg_da = nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor([cfg.LOSS.SEG_POS_WEIGHT])
        ).to(device)
        
        # Lane line: Hybrid loss (Dice + Focal) as per YOLOPv2 paper
        self.hybrid_loss_lane = HybridLoss(
            gamma_focal=2.0,  # From paper
            alpha_focal=0.25,  # From paper
            gamma_weight=getattr(cfg.LOSS, 'HYBRID_GAMMA', 1.0)  # Trade-off parameter
        )
        
        # Loss weights (lambdas)
        self.lambdas = cfg.LOSS.MULTI_HEAD_LAMBDA if hasattr(cfg.LOSS, 'MULTI_HEAD_LAMBDA') else [1.0] * 6
        
    def forward(self, predictions, targets, shapes, model, imgs=None):
        """
        Args:
            predictions: [det_outputs, da_seg_output, ll_seg_output]
            targets: [det_targets, da_seg_targets, ll_seg_targets]
            shapes: image shapes
            model: model instance
            imgs: input images (not used)
            
        Returns:
            total_loss: sum of all losses
            head_losses: tuple of individual losses
            loss_dict: dict of individual losses for gradient computation
        """
        device = self.device
        cfg = self.cfg
        
        # Initialize losses
        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)
        
        # Build targets for detection
        tcls, tbox, indices, anchors = build_targets(cfg, predictions[0], targets[0], model)
        
        # Class label smoothing
        cp, cn = smooth_BCE(eps=0.0)
        
        # Detection loss calculation (same as v1)
        nt = 0  # number of targets
        no = len(predictions[0])  # number of outputs
        balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]
        
        for i, pi in enumerate(predictions[0]):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(pi[..., 0], device=device)
            
            n = b.shape[0]
            if n:
                nt += n
                ps = pi[b, a, gj, gi]
                
                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1).to(device)
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)
                lbox += (1.0 - iou).mean()
                
                # Objectness
                gr = model.gr if hasattr(model, 'gr') else 1.0
                tobj[b, a, gj, gi] = (1.0 - gr) + gr * iou.detach().clamp(0).type(tobj.dtype)
                
                # Classification
                if model.nc > 1:
                    t = torch.full_like(ps[:, 5:], cn, device=device)
                    t[range(n), tcls[i]] = cp
                    lcls += self.BCEcls(ps[:, 5:], t)
                    
            lobj += self.BCEobj(pi[..., 4], tobj) * balance[i]
        
        # Drivable area segmentation loss (standard BCE)
        drive_area_seg_predicts = predictions[1]
        drive_area_seg_targets = targets[1]
        lseg_da = self.BCEseg_da(
            drive_area_seg_predicts.view(-1),
            drive_area_seg_targets.view(-1)
        )
        
        # Lane line segmentation loss (Hybrid loss)
        lane_line_seg_predicts = predictions[2]
        lane_line_seg_targets = targets[2]
        lseg_ll = self.hybrid_loss_lane(
            lane_line_seg_predicts,
            lane_line_seg_targets
        )
        
        # Store raw losses before gain application for conflict calculation
        lbox_raw = lbox.clone()
        lobj_raw = lobj.clone()
        lcls_raw = lcls.clone()
        lseg_da_raw = lseg_da.clone()
        lseg_ll_raw = lseg_ll.clone()
        
        # Apply gains and lambdas
        s = 3 / no  # output count scaling
        lcls *= cfg.LOSS.CLS_GAIN * s * self.lambdas[0]
        lobj *= cfg.LOSS.OBJ_GAIN * s * (1.4 if no == 4 else 1.) * self.lambdas[1]
        lbox *= cfg.LOSS.BOX_GAIN * s * self.lambdas[2]
        
        lseg_da *= cfg.LOSS.DA_SEG_GAIN * self.lambdas[3]
        lseg_ll *= cfg.LOSS.LL_SEG_GAIN * self.lambdas[4]
        
        # Handle training mode flags
        if cfg.TRAIN.DET_ONLY or cfg.TRAIN.ENC_DET_ONLY:
            lseg_da = 0 * lseg_da
            lseg_ll = 0 * lseg_ll
            
        if cfg.TRAIN.SEG_ONLY or cfg.TRAIN.ENC_SEG_ONLY:
            lcls = 0 * lcls
            lobj = 0 * lobj
            lbox = 0 * lbox
            
        if cfg.TRAIN.LANE_ONLY:
            lcls = 0 * lcls
            lobj = 0 * lobj
            lbox = 0 * lbox
            lseg_da = 0 * lseg_da
            
        if cfg.TRAIN.DRIVABLE_ONLY:
            lcls = 0 * lcls
            lobj = 0 * lobj
            lbox = 0 * lbox
            lseg_ll = 0 * lseg_ll
        
        # Total loss
        total_loss = lbox + lobj + lcls + lseg_da + lseg_ll
        
        # Loss dict for explicit gradient computation
        # Include raw losses for YOLOPv2 conflict calculation
        loss_dict = {
            'loss_det': lbox + lobj + lcls,  # After gains
            'loss_da_seg': lseg_da,  # After gains
            'loss_ll_seg': lseg_ll,  # After gains
            # Raw losses for conflict calculation
            'loss_det_raw': lbox_raw + lobj_raw + lcls_raw,
            'loss_da_seg_raw': lseg_da_raw,
            'loss_ll_seg_raw': lseg_ll_raw
        }
        
        # Return format compatible with training code
        head_losses = (
            (lbox + lobj + lcls).item(),
            lseg_da.item(),
            lseg_ll.item(),
            0.0  # No IoU loss for YOLOPv2 (included in hybrid loss)
        )
        
        return total_loss, head_losses, loss_dict