"""
YOLOPX Conflict Calculator
Model-specific parameter extraction for YOLOPX (anchor-free, YOLOX-based)
"""

import torch
from typing import List, Optional
from lib.core.conflict_calculator.base_conflict import BaseConflictCalculator


class YOLOPXConflictCalculator(BaseConflictCalculator):
    """
    YOLOPX-specific conflict calculator.
    
    YOLOPX Architecture:
    - Module 0: ELANNet (backbone)
    - Module 1: PaFPNELAN (neck)
    - Module 2+: Detection and segmentation heads
    """
    
    def _initialize_params(self):
        """
        Initialize parameters for YOLOPX architecture.
        Extracts parameters from backbone, neck, and heads.
        """
        # Get actual model (handle DataParallel wrapper)
        actual_model = self.model.module if hasattr(self.model, 'module') else self.model
        
        if hasattr(actual_model, 'model') and isinstance(actual_model.model, torch.nn.Sequential):
            # YOLOPX uses Sequential model structure
            
            # Backbone parameters (Module 0: ELANNet)
            if len(actual_model.model) > 0:
                backbone_params = []
                module = actual_model.model[0]
                for name, param in module.named_parameters():
                    if param.requires_grad:
                        backbone_params.append((f"backbone.{name}", param))
                
                if backbone_params:
                    self.params_dict['backbone'] = backbone_params
                    print(f"YOLOPX: Initialized backbone with {len(backbone_params)} parameters")
            
            # Neck parameters (Module 1: PaFPNELAN)
            if len(actual_model.model) > 1:
                neck_params = []
                module = actual_model.model[1]
                for name, param in module.named_parameters():
                    if param.requires_grad:
                        neck_params.append((f"neck.{name}", param))
                
                if neck_params:
                    self.params_dict['neck'] = neck_params
                    print(f"YOLOPX: Initialized neck with {len(neck_params)} parameters")
            
            # Head parameters (Modules 2+: Detection and segmentation heads)
            if len(actual_model.model) > 2:
                head_params = []
                for idx in range(2, len(actual_model.model)):
                    module = actual_model.model[idx]
                    for name, param in module.named_parameters():
                        if param.requires_grad:
                            head_params.append((f"head_{idx}.{name}", param))
                
                if head_params:
                    self.params_dict['heads'] = head_params
                    print(f"YOLOPX: Initialized heads with {len(head_params)} parameters")
        
        # Verify initialization
        if not self.params_dict:
            print("Warning: YOLOPX - No parameters found for conflict analysis")
            print(f"Model structure: {type(actual_model)}")
            if hasattr(actual_model, 'model'):
                print(f"Model length: {len(actual_model.model) if hasattr(actual_model.model, '__len__') else 'N/A'}")
        else:
            total_params = sum(len(params) for params in self.params_dict.values())
            print(f"YOLOPX: Total parameters tracked: {total_params}")
    
    def get_model_info(self) -> dict:
        """
        Get model-specific information for debugging/logging.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'model_type': 'YOLOPX',
            'architecture': 'Anchor-free (YOLOX-based)',
            'locations': list(self.params_dict.keys()),
            'param_counts': {loc: len(params) for loc, params in self.params_dict.items()},
            'total_params': sum(len(params) for params in self.params_dict.values())
        }
        return info