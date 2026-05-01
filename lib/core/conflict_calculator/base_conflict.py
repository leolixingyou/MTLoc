"""
Base Conflict Calculator for Multi-Task Learning

Provides base functionality for computing gradient conflicts between tasks.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod


class BaseConflictCalculator(ABC):
    """
    Base class for computing gradient conflicts in multi-task learning.

    Subclasses should implement _initialize_params() to extract
    model-specific parameters for conflict analysis.
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        """
        Initialize the conflict calculator.

        Args:
            model: The neural network model
            device: Device for computations (defaults to model's device)
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.params_dict: Dict[str, List[Tuple[str, nn.Parameter]]] = {}

        # Initialize parameters
        self._initialize_params()

    @abstractmethod
    def _initialize_params(self):
        """
        Initialize the parameters dictionary.

        Subclasses must implement this to extract parameters from the model
        and populate self.params_dict with location -> [(name, param), ...] mapping.
        """
        pass

    def compute_gradients(self) -> Dict[str, torch.Tensor]:
        """
        Extract current gradients for all tracked parameters.

        Returns:
            Dictionary mapping location names to flattened gradient tensors
        """
        gradients = {}

        for location, params in self.params_dict.items():
            grads = []
            for name, param in params:
                if param.grad is not None:
                    grads.append(param.grad.view(-1))

            if grads:
                gradients[location] = torch.cat(grads)

        return gradients

    def compute_cosine_similarity(self, grad1: torch.Tensor, grad2: torch.Tensor) -> float:
        """
        Compute cosine similarity between two gradient vectors.

        Args:
            grad1: First gradient vector
            grad2: Second gradient vector

        Returns:
            Cosine similarity value in [-1, 1]
        """
        if grad1.numel() == 0 or grad2.numel() == 0:
            return 0.0

        # Ensure same device
        if grad1.device != grad2.device:
            grad2 = grad2.to(grad1.device)

        # Ensure same size
        min_size = min(grad1.numel(), grad2.numel())
        grad1 = grad1[:min_size]
        grad2 = grad2[:min_size]

        # Compute cosine similarity
        dot_product = torch.dot(grad1.flatten(), grad2.flatten())
        norm1 = torch.norm(grad1)
        norm2 = torch.norm(grad2)

        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0

        return (dot_product / (norm1 * norm2)).item()

    def compute_conflict_metrics(self,
                                 gradients: Dict[str, torch.Tensor]) -> Dict[str, Dict]:
        """
        Compute conflict metrics between all pairs of gradient locations.

        Args:
            gradients: Dictionary of location -> gradient tensor

        Returns:
            Nested dictionary of conflict metrics between location pairs
        """
        metrics = {}
        locations = list(gradients.keys())

        for i, loc1 in enumerate(locations):
            for loc2 in locations[i+1:]:
                grad1 = gradients[loc1]
                grad2 = gradients[loc2]

                cos_sim = self.compute_cosine_similarity(grad1, grad2)

                # Conflict severity: how negative is the cosine similarity
                severity = max(0, -cos_sim)

                pair_name = f"{loc1}_vs_{loc2}"
                metrics[pair_name] = {
                    'cosine_similarity': cos_sim,
                    'conflict_severity': severity,
                    'is_conflicting': cos_sim < 0,
                    'grad1_norm': grad1.norm().item(),
                    'grad2_norm': grad2.norm().item()
                }

        return metrics

    def analyze_step(self) -> Dict:
        """
        Perform full conflict analysis for current training step.

        Returns:
            Dictionary containing all computed metrics
        """
        gradients = self.compute_gradients()

        if not gradients:
            return {'status': 'no_gradients'}

        metrics = self.compute_conflict_metrics(gradients)

        # Compute summary statistics
        if metrics:
            cos_sims = [m['cosine_similarity'] for m in metrics.values()]
            severities = [m['conflict_severity'] for m in metrics.values()]

            summary = {
                'avg_cosine_similarity': sum(cos_sims) / len(cos_sims),
                'min_cosine_similarity': min(cos_sims),
                'max_cosine_similarity': max(cos_sims),
                'avg_severity': sum(severities) / len(severities),
                'max_severity': max(severities),
                'num_conflicts': sum(1 for m in metrics.values() if m['is_conflicting'])
            }
        else:
            summary = {}

        return {
            'pairwise_metrics': metrics,
            'summary': summary,
            'num_locations': len(gradients)
        }
