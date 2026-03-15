"""
Vanilla loss module for ablation study - using naive equal weights
用于消融实验的朴素损失模块 - 使用未优化的相等权重
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.core.loss import MultiHeadLoss, FocalLossSeg, TverskyLoss
from lib.models.YOLOX_Loss import YOLOX_Loss


class VanillaLoss(nn.Module):
    """
    Vanilla loss with naive equal weights for all tasks.
    用于消融实验的朴素损失，所有任务使用相同权重。
    
    This is intentionally NOT optimized to demonstrate architectural advantages.
    这是故意不优化的，用于展示架构本身的优势。
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
        
        # NAIVE EQUAL WEIGHTS - This is the key for ablation study
        # 朴素的相等权重 - 这是消融实验的关键
        self.task_weights = {
            'det': 1.0,        # Equal weight
            'da_seg': 1.0,     # Equal weight
            'll_seg': 1.0,     # Equal weight
            'll_tversky': 1.0  # Equal weight
        }
        
        # Note: We intentionally do NOT apply the original lambdas
        # to keep the comparison absolutely fair
        
        self.cfg = cfg
        self.device = device
        
    def forward(self, predictions, targets, shapes, model, imgs):
        """
        Forward pass with naive equal weights.
        使用朴素相等权重的前向传播。
        
        Args:
            predictions: [detection_output, da_seg_output, ll_seg_output]
            targets: [det_targets, da_seg_targets, ll_seg_targets]
            shapes: Image shapes
            model: The model instance
            imgs: Input images
            
        Returns:
            total_loss: Naively weighted total loss
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
        
        # Apply NAIVE EQUAL weights (no optimization)
        # 应用朴素的相等权重（未优化）
        det_loss_final = det_loss * self.task_weights['det']
        da_seg_loss_final = da_seg_loss * self.task_weights['da_seg']
        ll_seg_loss_final = ll_seg_loss * self.task_weights['ll_seg']
        ll_tversky_loss_final = ll_tversky_loss * self.task_weights['ll_tversky']
        
        # Total loss - simple sum with equal weights
        # 总损失 - 相等权重的简单求和
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


def get_vanilla_loss(cfg, device, model):
    """
    Factory function to create vanilla loss module for ablation study.
    为消融实验创建朴素损失模块的工厂函数。
    
    Args:
        cfg: Configuration object
        device: Device to place loss on
        model: Model instance
        
    Returns:
        VanillaLoss instance
    """
    return VanillaLoss(cfg, device, model)