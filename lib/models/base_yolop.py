"""
Base class for all YOLOP series models.
Provides unified interface for multi-task learning models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any


class BaseYOLOP(nn.Module, ABC):
    """
    Abstract base class for YOLOP series models.
    All YOLOP variants should inherit from this class.
    """
    
    def __init__(self, cfg=None):
        super(BaseYOLOP, self).__init__()
        self.cfg = cfg
        self.nc = 1  # number of classes for detection
        self.detector = None
        self.da_seg_head = None
        self.ll_seg_head = None
        
        # Feature visualization flags
        self.visualize_features = False
        self.feature_visualizations = {}
        self.stored_input_image = None
        
        # Task indices (can be overridden by subclasses)
        self.det_out_idx = 2
        self.da_seg_out_idx = 16
        self.ll_seg_out_idx = 31
        
    @abstractmethod
    def build_model(self, cfg):
        """
        Build the model architecture.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def forward(self, x, input_image=None):
        """
        Forward pass of the model.
        Must return [det_output, da_seg_output, ll_seg_output]
        
        Args:
            x: Input tensor [B, 3, H, W]
            input_image: Original input for visualization (optional)
            
        Returns:
            List of outputs: [detection, da_segmentation, ll_segmentation]
        """
        pass
    
    def fuse(self):
        """
        Fuse Conv2d() + BatchNorm2d() layers for inference speedup.
        Can be overridden by subclasses for custom fusion.
        """
        print('Fusing layers... ')
        for m in self.modules():
            if type(m) is nn.Conv2d and hasattr(m, 'bn'):
                m.conv = self.fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.fuseforward
        return self
    
    @staticmethod
    def fuse_conv_and_bn(conv, bn):
        """Fuse Conv2d and BatchNorm2d layers."""
        fusedconv = nn.Conv2d(conv.in_channels,
                              conv.out_channels,
                              kernel_size=conv.kernel_size,
                              stride=conv.stride,
                              padding=conv.padding,
                              groups=conv.groups,
                              bias=True).requires_grad_(False).to(conv.weight.device)
        
        # prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
        
        # prepare spatial bias
        b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
        
        return fusedconv
    
    def visualize_feature_map(self, feature, name, input_img=None, save_path=None):
        """
        Visualize a single feature map as heatmap overlay on input image.
        
        Args:
            feature: Feature tensor [B, C, H, W]
            name: Name for the feature map
            input_img: Original input image [B, 3, H, W]
            save_path: Optional path to save visualization
        """
        with torch.no_grad():
            if feature is None:
                return None
            
            # Use stored input image if not provided
            if input_img is None:
                input_img = self.stored_input_image
            
            if input_img is None:
                return None
            
            # Take first batch sample
            feat = feature[0] if feature.dim() == 4 else feature
            img = input_img[0] if input_img.dim() == 4 else input_img
            
            # Average across channels to get single heatmap
            if feat.dim() == 3 and feat.shape[0] > 1:
                heatmap = torch.mean(feat, dim=0)
            else:
                heatmap = feat.squeeze()
            
            # Normalize to [0, 1]
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            # Resize heatmap to match input size
            h, w = img.shape[-2:]
            heatmap = F.interpolate(
                heatmap.unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )[0, 0]
            
            # Convert to numpy and colormap
            heatmap_np = heatmap.cpu().numpy()
            heatmap_colored = cv2.applyColorMap((heatmap_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            # Convert input image to numpy
            img_np = img.cpu().permute(1, 2, 0).numpy()
            if img_np.max() <= 1:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
            
            # Ensure RGB format
            if img_np.shape[2] == 3:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # Overlay heatmap on image
            overlay = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)
            
            # Add text label
            cv2.putText(overlay, name, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Save if path provided
            if save_path:
                cv2.imwrite(save_path, overlay)
            
            # Store for later retrieval
            self.feature_visualizations[name] = overlay
            
            return overlay
    
    def visualize_intermediate_features(self, features_dict, input_img=None):
        """
        Visualize intermediate features during forward pass.
        
        Args:
            features_dict: Dictionary of {name: feature_tensor}
            input_img: Original input image
        """
        if not self.visualize_features:
            return
        
        for name, feature in features_dict.items():
            self.visualize_feature_map(feature, name, input_img)
    
    def get_all_feature_visualizations(self):
        """
        Get all stored feature visualizations.
        
        Returns:
            Dictionary of {feature_name: visualization_image}
        """
        return self.feature_visualizations.copy()
    
    def clear_feature_visualizations(self):
        """Clear stored feature visualizations to free memory."""
        self.feature_visualizations.clear()
        self.stored_input_image = None
    
    def info(self, verbose=False, img_size=640):
        """
        Print model information.
        Can be overridden by subclasses for custom info.
        """
        model_info(self, verbose, img_size)
    
    def get_task_outputs(self, outputs):
        """
        Extract task-specific outputs from model outputs.
        Provides a unified interface for all models.
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            Tuple of (det_output, da_seg_output, ll_seg_output)
        """
        if isinstance(outputs, (list, tuple)) and len(outputs) == 3:
            return outputs[0], outputs[1], outputs[2]
        else:
            # Handle different output formats from different models
            # This can be overridden in subclasses
            raise NotImplementedError("Subclass must implement output extraction")
    
    def initialize_weights(self):
        """
        Initialize model weights.
        Can be overridden by subclasses for custom initialization.
        """
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
                m.inplace = True


def model_info(model, verbose=False, img_size=640):
    """Print model information."""
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPS
        from thop import profile
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device)  # input
        flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2  # stride GFLOPS
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # expand if int/float
        fs = ', %.1f GFLOPS' % (flops * img_size[0] / stride * img_size[1] / stride)  # 640x640 GFLOPS
    except (ImportError, Exception):
        fs = ''

    print(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")