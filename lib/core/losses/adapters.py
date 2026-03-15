"""
Adapters for converting between different label formats
"""

import torch
import torch.nn as nn
from .compute_loss_v1 import ComputeLossV1


class ComputeLossWithAdapter(nn.Module):
    """
    Adapter wrapper for ComputeLossV1 to handle static format labels.
    Converts static format (B, N, 5) to dynamic format (N, 6) before passing to ComputeLossV1.
    """
    
    def __init__(self, cfg, device, model=None):
        super().__init__()
        self.compute_loss = ComputeLossV1(cfg, device)
        self.device = device
        
    def static_to_dynamic(self, static_labels):
        """
        Convert static padded format to dynamic format.
        
        Args:
            static_labels: Tensor of shape (B, N, 5) where:
                          B = batch size
                          N = max number of objects (padded)
                          5 = [class_id, cx, cy, w, h]
                          
        Returns:
            dynamic_labels: Tensor of shape (M, 6) where:
                           M = total number of valid objects
                           6 = [batch_idx, class_id, cx, cy, w, h]
        """
        if static_labels.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {static_labels.dim()}D")
            
        batch_size = static_labels.shape[0]
        dynamic_labels = []
        
        for batch_idx in range(batch_size):
            # Get labels for this image
            labels = static_labels[batch_idx]
            
            # Filter out padded entries (sum > 0 means valid object)
            valid_mask = labels.sum(dim=1) > 0
            valid_labels = labels[valid_mask]
            
            if len(valid_labels) > 0:
                # Add batch index as first column
                batch_indices = torch.full((len(valid_labels), 1), 
                                          batch_idx, 
                                          dtype=labels.dtype, 
                                          device=labels.device)
                # Concatenate batch index with labels
                batch_labels = torch.cat([batch_indices, valid_labels], dim=1)
                dynamic_labels.append(batch_labels)
        
        # Concatenate all batch labels
        if dynamic_labels:
            return torch.cat(dynamic_labels, dim=0)
        else:
            # Return empty tensor with correct shape if no valid labels
            return torch.empty((0, 6), dtype=static_labels.dtype, device=static_labels.device)
    
    def forward(self, predictions, targets, shapes, model, imgs=None):
        """
        Forward pass with format conversion.
        
        Args:
            predictions: Model outputs [det_outputs, da_seg_output, ll_seg_output]
            targets: Ground truth [det_targets, da_seg_targets, ll_seg_targets]
                    where det_targets is in static format (B, N, 5)
            shapes: Image shapes
            model: Model instance
            imgs: Input images
            
        Returns:
            Same as ComputeLossV1
        """
        # Convert detection targets from static to dynamic format
        det_targets_static = targets[0]
        det_targets_dynamic = self.static_to_dynamic(det_targets_static)
        
        # Replace detection targets with dynamic format
        adapted_targets = [det_targets_dynamic, targets[1], targets[2]]
        
        # Call original loss function with adapted targets
        return self.compute_loss(predictions, adapted_targets, shapes, model, imgs)


class ComputeLossV3WithAdapter(ComputeLossWithAdapter):
    """
    Adapter for YOLOPv3. Currently uses the same loss as v1.
    This class exists for future customization if needed.
    """
    pass


def get_adapted_loss(model_type, cfg, device, model=None):
    """
    Factory function to get the appropriate loss with adapter.
    
    Args:
        model_type: Type of model ('yolop_v1', 'yolop_v3', etc.)
        cfg: Configuration object
        device: Device to run on
        model: Model instance (optional)
        
    Returns:
        Loss module with appropriate adapter
    """
    if model_type in ['yolop_v1', 'yolop_v2']:
        return ComputeLossWithAdapter(cfg, device, model)
    elif model_type == 'yolop_v3':
        return ComputeLossV3WithAdapter(cfg, device, model)
    else:
        raise ValueError(f"Unknown model type for adapted loss: {model_type}")