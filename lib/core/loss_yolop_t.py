"""
YOLOP-T specific loss module with gradient balancing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.core.loss import MultiHeadLoss, FocalLossSeg, TverskyLoss
from lib.models.YOLOX_Loss import YOLOX_Loss


class YOLOPTLoss(nn.Module):
    """
    YOLOP-T specific loss with gradient balancing.
    
    Key features:
    1. Monitors gradient magnitudes from each task
    2. Dynamically adjusts loss weights to balance gradients
    3. Ensures all tasks contribute meaningfully to backbone learning
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
        
        # Initial weights (will be adjusted dynamically)
        self.lambdas = cfg.LOSS.MULTI_HEAD_LAMBDA if hasattr(cfg.LOSS, 'MULTI_HEAD_LAMBDA') else [1.0] * 5
        
        # Gradient balancing parameters
        self.use_gradient_balancing = True
        self.gradient_history = {
            'det': [],
            'da_seg': [],
            'll_seg': [],
            'll_tversky': []
        }
        self.history_size = 100  # Rolling window for gradient statistics
        
        # Target gradient ratio (all tasks should have similar gradient magnitudes)
        self.target_gradient_ratio = 1.0
        
        # Adaptive weight bounds (prevent extreme adjustments)
        self.min_weight = 0.01
        self.max_weight = 10.0
        
        # Current adaptive weights
        self.adaptive_weights = {
            'det': 0.02,      # Initial from original
            'da_seg': 0.2,    # Initial from original
            'll_seg': 0.2,    # Initial from original
            'll_tversky': 0.2 # Initial from original
        }
        
        self.cfg = cfg
        self.device = device
        
    def forward(self, predictions, targets, shapes, model, imgs):
        """
        Forward pass with gradient monitoring and balancing.
        
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
        
        # Apply adaptive weights for gradient balancing
        if self.use_gradient_balancing:
            det_loss_final = det_loss_weighted * self.adaptive_weights['det']
            da_seg_loss_final = da_seg_loss_weighted * self.adaptive_weights['da_seg']
            ll_seg_loss_final = ll_seg_loss_weighted * self.adaptive_weights['ll_seg']
            ll_tversky_loss_final = ll_tversky_loss_weighted * self.adaptive_weights['ll_tversky']
        else:
            # Use original fixed weights
            det_loss_final = det_loss_weighted * 0.02
            da_seg_loss_final = da_seg_loss_weighted * 0.2
            ll_seg_loss_final = ll_seg_loss_weighted * 0.2
            ll_tversky_loss_final = ll_tversky_loss_weighted * 0.2
        
        # Total loss
        total_loss = det_loss_final + da_seg_loss_final + ll_seg_loss_final + ll_tversky_loss_final
        
        # Create loss dictionary for gradient computation
        loss_dict = {
            'loss_det': det_loss_final,
            'loss_da_seg': da_seg_loss_final,
            'loss_ll_seg': ll_seg_loss_final,
            'loss_ll_tversky': ll_tversky_loss_final
        }
        
        # Monitor gradients and update weights (during training only)
        if model.training and self.use_gradient_balancing:
            self._update_gradient_weights(loss_dict, model)
        
        # Return values
        head_losses = (
            det_loss_final.item(),
            da_seg_loss_final.item(),
            ll_seg_loss_final.item(),
            ll_tversky_loss_final.item(),
            total_loss.item()
        )
        
        return total_loss, head_losses, loss_dict
    
    def _update_gradient_weights(self, loss_dict, model):
        """
        Monitor gradient magnitudes and update weights for balancing.
        
        This ensures all tasks contribute meaningfully to backbone learning.
        """
        
        # Calculate gradient norms for each task
        gradient_norms = {}
        
        for task_name, loss in loss_dict.items():
            # Clear gradients
            model.zero_grad()
            
            # Compute gradients
            loss.backward(retain_graph=True)
            
            # Calculate backbone gradient norm
            grad_norm = 0.0
            param_count = 0
            for name, param in model.named_parameters():
                if 'model.0' in name and param.grad is not None:  # Backbone parameters
                    grad_norm += param.grad.norm().item()
                    param_count += 1
            
            # Store normalized gradient magnitude
            if param_count > 0:
                task_key = task_name.replace('loss_', '')
                gradient_norms[task_key] = grad_norm / param_count
        
        # Clear gradients after measurement
        model.zero_grad()
        
        # Update gradient history
        for task_key, norm in gradient_norms.items():
            if task_key in self.gradient_history:
                self.gradient_history[task_key].append(norm)
                # Keep only recent history
                if len(self.gradient_history[task_key]) > self.history_size:
                    self.gradient_history[task_key].pop(0)
        
        # Calculate average gradient magnitudes
        avg_gradients = {}
        for task_key, history in self.gradient_history.items():
            if len(history) > 0:
                avg_gradients[task_key] = sum(history) / len(history)
        
        # Update adaptive weights if we have enough history
        if len(avg_gradients) == 4 and all(len(h) >= 10 for h in self.gradient_history.values()):
            # Calculate target gradient magnitude (geometric mean)
            all_grads = list(avg_gradients.values())
            # Ensure non-zero values for geometric mean
            all_grads = [max(g, 1e-8) for g in all_grads]
            target_grad = (all_grads[0] * all_grads[1] * all_grads[2] * all_grads[3]) ** 0.25
            
            # Update weights to balance gradients
            for task_key in self.adaptive_weights:
                if task_key in avg_gradients and avg_gradients[task_key] > 1e-8 and target_grad > 1e-8:
                    # Calculate adjustment factor
                    current_ratio = avg_gradients[task_key] / target_grad
                    
                    # Apply smooth adjustment (exponential moving average)
                    adjustment = 1.0 / current_ratio
                    smoothing = 0.95  # Higher = slower adaptation
                    new_weight = (smoothing * self.adaptive_weights[task_key] + 
                                  (1 - smoothing) * self.adaptive_weights[task_key] * adjustment)
                    
                    # Clip to bounds
                    self.adaptive_weights[task_key] = max(self.min_weight, 
                                                          min(self.max_weight, new_weight))
    
    def get_gradient_stats(self):
        """Get current gradient statistics for logging."""
        stats = {}
        for task_key, history in self.gradient_history.items():
            if len(history) > 0:
                stats[f'{task_key}_grad_mean'] = sum(history) / len(history)
                stats[f'{task_key}_grad_std'] = torch.std(torch.tensor(history)).item()
        
        # Add current weights
        for task_key, weight in self.adaptive_weights.items():
            stats[f'{task_key}_weight'] = weight
            
        return stats


def get_yolop_t_loss(cfg, device, model):
    """
    Factory function to create YOLOP-T loss module.
    
    Args:
        cfg: Configuration object
        device: Device to place loss on
        model: Model instance
        
    Returns:
        YOLOPTLoss instance
    """
    return YOLOPTLoss(cfg, device, model)