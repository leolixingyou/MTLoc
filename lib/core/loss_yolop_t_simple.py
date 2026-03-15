"""
Simplified YOLOP-T loss module with fixed gradient scaling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.core.loss import MultiHeadLoss, FocalLossSeg, TverskyLoss
from lib.models.YOLOX_Loss import YOLOX_Loss


class YOLOPTSimpleLoss(nn.Module):
    """
    YOLOP-T specific loss with fixed gradient scaling.
    
    This simplified version uses predetermined weights that ensure
    all tasks contribute meaningfully to backbone learning.
    """
    
    def __init__(self, cfg, device, model):
        super().__init__()
        
        # Base loss functions (same as original)
        self.det_loss = YOLOX_Loss(device, 1)
        self.da_seg_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor([cfg.LOSS.SEG_POS_WEIGHT]).to(device)
        )
        self.ll_seg_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor([cfg.LOSS.SEG_POS_WEIGHT]).to(device)
        )
        self.tversky_loss = TverskyLoss(alpha=0.7, beta=0.3, gamma=4.0/3).to(device)
        
        # Apply focal loss if configured
        gamma = cfg.LOSS.FL_GAMMA
        if gamma > 0.0:
            self.ll_seg_loss = FocalLossSeg(self.ll_seg_loss, gamma)
        
        # Base weights from config
        self.lambdas = cfg.LOSS.MULTI_HEAD_LAMBDA if hasattr(cfg.LOSS, 'MULTI_HEAD_LAMBDA') else [1.0] * 5
        
        # Fixed gradient-balanced weights
        # These are adjusted to ensure all tasks contribute similarly to backbone
        # Based on empirical observation that detection gradients are ~1000x larger
        self.task_weights = {
            'det': 0.002,      # Reduced from 0.02 to balance with segmentation
            'da_seg': 0.5,     # Increased from 0.2 to boost gradient contribution
            'll_seg': 0.5,     # Increased from 0.2 to boost gradient contribution
            'll_tversky': 0.5  # Increased from 0.2 to boost gradient contribution
        }
        
        self.cfg = cfg
        self.device = device
        
    def forward(self, predictions, targets, shapes, model, imgs):
        """
        Forward pass with fixed gradient scaling.
        
        Args:
            predictions: [detection_output, da_seg_output, ll_seg_output]
            targets: [det_targets, da_seg_targets, ll_seg_targets]
            shapes: Image shapes
            model: The model instance
            imgs: Input images
            
        Returns:
            total_loss: Balanced total loss
            head_losses: Individual loss values
            loss_dict: Dictionary of losses for gradient computation
        """
        
        # Calculate individual losses
        det_loss = self.det_loss(predictions[0], targets[0], imgs)
        
        # Driving area segmentation loss
        da_seg_pred = predictions[1].view(-1)
        da_seg_target = targets[1].view(-1)
        da_seg_loss = self.da_seg_loss(da_seg_pred, da_seg_target)
        
        # Lane line segmentation loss
        ll_seg_pred = predictions[2].view(-1)
        ll_seg_target = targets[2].view(-1)
        ll_seg_loss = self.ll_seg_loss(ll_seg_pred, ll_seg_target)
        
        # Tversky loss for lane lines
        ll_tversky_loss = self.tversky_loss(predictions[2], targets[2])
        
        # Apply base lambdas
        det_loss_weighted = det_loss * self.lambdas[1]
        da_seg_loss_weighted = da_seg_loss * self.lambdas[2]
        ll_seg_loss_weighted = ll_seg_loss * self.lambdas[3]
        ll_tversky_loss_weighted = ll_tversky_loss * self.lambdas[4]
        
        # Apply fixed task weights for gradient balancing
        det_loss_final = det_loss_weighted * self.task_weights['det']
        da_seg_loss_final = da_seg_loss_weighted * self.task_weights['da_seg']
        ll_seg_loss_final = ll_seg_loss_weighted * self.task_weights['ll_seg']
        ll_tversky_loss_final = ll_tversky_loss_weighted * self.task_weights['ll_tversky']
        
        # Total loss
        total_loss = det_loss_final + da_seg_loss_final + ll_seg_loss_final + ll_tversky_loss_final
        
        # Create loss dictionary for gradient computation
        loss_dict = {
            'loss_det': det_loss_final,
            'loss_da_seg': da_seg_loss_final,
            'loss_ll_seg': ll_seg_loss_final,
            'loss_ll_tversky': ll_tversky_loss_final
        }
        
        # Return values
        head_losses = (
            det_loss_final.item(),
            da_seg_loss_final.item(),
            ll_seg_loss_final.item(),
            ll_tversky_loss_final.item(),
            total_loss.item()
        )
        
        return total_loss, head_losses, loss_dict


def get_yolop_t_simple_loss(cfg, device, model):
    """
    Factory function to create simplified YOLOP-T loss module.
    
    Args:
        cfg: Configuration object
        device: Device to place loss on
        model: Model instance
        
    Returns:
        YOLOPTSimpleLoss instance
    """
    return YOLOPTSimpleLoss(cfg, device, model)