"""
Detailed Conflict Analyzer for Thorough MTL Debugging

This module provides fine-grained conflict analysis at multiple levels:
1. Per-layer gradient conflict analysis
2. Per-parameter conflict heatmap generation
3. Gradient trajectory tracking
4. Feature map spatial conflict analysis

Author: Enhanced debugging system for MTL research
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import os
import json
from pathlib import Path


class DetailedConflictAnalyzer:
    """
    Fine-grained conflict analyzer for thorough debugging.

    This analyzer goes beyond standard metrics to provide:
    - Layer-wise gradient conflict decomposition
    - Parameter-level conflict identification
    - Temporal gradient trajectory tracking
    - Spatial feature conflict analysis
    """

    def __init__(self, model: nn.Module, output_dir: str,
                 track_layers: Optional[List[str]] = None,
                 save_interval: int = 1):
        """
        Initialize the detailed analyzer.

        Args:
            model: The multi-task model
            output_dir: Directory to save analysis results
            track_layers: Specific layers to track (None = all layers)
            save_interval: Save data every N steps
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.save_interval = save_interval

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'layer_conflicts').mkdir(exist_ok=True)
        (self.output_dir / 'param_heatmaps').mkdir(exist_ok=True)
        (self.output_dir / 'gradient_trajectories').mkdir(exist_ok=True)
        (self.output_dir / 'feature_conflicts').mkdir(exist_ok=True)
        (self.output_dir / 'raw_data').mkdir(exist_ok=True)

        # Get model structure
        self.model_unwrapped = model.module if hasattr(model, 'module') else model
        self.layer_names, self.layer_params = self._extract_layer_structure(track_layers)

        # History tracking
        self.gradient_history = defaultdict(list)  # task -> layer -> [gradients over time]
        self.conflict_history = defaultdict(list)  # metric_name -> [values over time]
        self.feature_history = defaultdict(list)   # layer -> [features over time]

        # Statistics
        self.step_count = 0
        self.epoch_count = 0

        print(f"[DetailedAnalyzer] Initialized with {len(self.layer_names)} layers to track")
        print(f"[DetailedAnalyzer] Output directory: {self.output_dir}")

    def _extract_layer_structure(self, track_layers: Optional[List[str]] = None) -> Tuple[List[str], Dict[str, List]]:
        """
        Extract layer structure from model.

        Returns:
            layer_names: List of layer names
            layer_params: Dict mapping layer name to list of (param_name, param) tuples
        """
        layer_params = defaultdict(list)
        layer_names = []

        for name, param in self.model_unwrapped.named_parameters():
            if not param.requires_grad:
                continue

            # Extract layer name (e.g., "model.0.conv1.weight" -> "model.0")
            parts = name.split('.')
            if len(parts) >= 2:
                layer_name = '.'.join(parts[:2])  # Take first two parts
            else:
                layer_name = parts[0]

            # Filter by track_layers if specified
            if track_layers is not None:
                if not any(track_name in layer_name for track_name in track_layers):
                    continue

            if layer_name not in layer_names:
                layer_names.append(layer_name)

            layer_params[layer_name].append((name, param))

        return layer_names, dict(layer_params)

    def analyze_step(self, losses: Dict[str, torch.Tensor],
                     optimizer: torch.optim.Optimizer,
                     step: int, epoch: int,
                     features: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        Perform detailed analysis for one training step.

        Args:
            losses: Task losses
            optimizer: Optimizer (for zeroing gradients)
            step: Current training step
            epoch: Current epoch
            features: Optional feature maps from forward pass

        Returns:
            Dictionary containing all analysis results
        """
        self.step_count = step
        self.epoch_count = epoch

        results = {}

        # 1. Layer-wise gradient conflict analysis
        layer_conflicts = self._analyze_layer_wise_conflicts(losses, optimizer)
        results['layer_conflicts'] = layer_conflicts

        # 2. Parameter-level conflict analysis
        param_conflicts = self._analyze_parameter_conflicts(losses, optimizer)
        results['param_conflicts'] = param_conflicts

        # 3. Gradient trajectory tracking
        self._track_gradient_trajectory(losses, optimizer)
        results['gradient_trajectory'] = self._get_trajectory_summary()

        # 4. Feature map conflict analysis (if features provided)
        if features is not None:
            feature_conflicts = self._analyze_feature_conflicts(features)
            results['feature_conflicts'] = feature_conflicts

        # 5. Save results periodically
        if step % self.save_interval == 0:
            self._save_results(results, step, epoch)

        return results

    def _analyze_layer_wise_conflicts(self, losses: Dict[str, torch.Tensor],
                                      optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """
        Analyze gradient conflicts at each layer.

        Uses torch.autograd.grad() to compute gradients without consuming the computation graph.

        Returns:
            Dict mapping layer names to conflict metrics
        """
        task_names = list(losses.keys())
        layer_conflicts = {}

        # Get all parameters as a flat list
        all_params = []
        param_to_layer = {}
        for layer_name, params in self.layer_params.items():
            for param_name, param in params:
                if param.requires_grad:
                    all_params.append(param)
                    param_to_layer[id(param)] = layer_name

        if not all_params:
            return layer_conflicts

        # Collect gradients per task per layer using autograd.grad (preserves graph)
        task_layer_gradients = {task: {} for task in task_names}

        for task_name, loss in losses.items():
            # Skip if loss is zero (no gradient)
            if not isinstance(loss, torch.Tensor) or loss.abs().item() < 1e-10:
                continue

            try:
                # Use autograd.grad to compute gradients without backward()
                grads = torch.autograd.grad(
                    outputs=loss,
                    inputs=all_params,
                    retain_graph=True,
                    allow_unused=True
                )

                # Organize gradients by layer
                for param, grad in zip(all_params, grads):
                    if grad is None:
                        continue
                    layer_name = param_to_layer[id(param)]
                    if layer_name not in task_layer_gradients[task_name]:
                        task_layer_gradients[task_name][layer_name] = []
                    task_layer_gradients[task_name][layer_name].append(grad.detach().flatten())

                # Concatenate gradients per layer
                for layer_name in list(task_layer_gradients[task_name].keys()):
                    grads_list = task_layer_gradients[task_name][layer_name]
                    if grads_list:
                        task_layer_gradients[task_name][layer_name] = torch.cat(grads_list)
                    else:
                        del task_layer_gradients[task_name][layer_name]

            except RuntimeError as e:
                print(f"[DetailedAnalyzer] Warning: Could not compute gradient for {task_name}: {e}")
                continue

        # Use only tasks that actually have gradients (handles DET_ONLY, SEG_ONLY, etc.)
        active_tasks = [t for t in task_names if task_layer_gradients.get(t)]

        # Compute pairwise conflicts for each layer
        for layer_name in self.layer_names:
            layer_metrics = {}

            # Get gradients for this layer from active tasks only
            grads = {}
            for task_name in active_tasks:
                if layer_name in task_layer_gradients[task_name]:
                    grads[task_name] = task_layer_gradients[task_name][layer_name]

            if len(grads) < 2:
                continue

            # Compute all pairwise conflicts using active tasks
            for i, task_i in enumerate(active_tasks):
                for j, task_j in enumerate(active_tasks):
                    if i >= j or task_i not in grads or task_j not in grads:
                        continue

                    grad_i = grads[task_i]
                    grad_j = grads[task_j]

                    # Cosine similarity
                    cos_sim = torch.nn.functional.cosine_similarity(
                        grad_i.unsqueeze(0), grad_j.unsqueeze(0)
                    ).item()

                    # Gradient norms
                    norm_i = torch.norm(grad_i).item()
                    norm_j = torch.norm(grad_j).item()

                    # Conflict metrics
                    conflict_key = f"{task_i}_vs_{task_j}"
                    layer_metrics[conflict_key] = {
                        'cosine_similarity': cos_sim,
                        'is_conflict': cos_sim < 0,
                        'norm_i': norm_i,
                        'norm_j': norm_j,
                        'magnitude_product': norm_i * norm_j,
                        'conflict_severity': (1 - cos_sim) * np.sqrt(norm_i * norm_j) if norm_i > 0 and norm_j > 0 else 0
                    }

            layer_conflicts[layer_name] = layer_metrics

        return layer_conflicts

    def _analyze_parameter_conflicts(self, losses: Dict[str, torch.Tensor],
                                     optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """
        Analyze conflicts at individual parameter level.

        Uses torch.autograd.grad() to compute gradients without consuming the computation graph.

        Returns:
            Dict containing parameter-level conflict statistics
        """
        task_names = list(losses.keys())
        param_conflicts = defaultdict(dict)

        # Get all trainable parameters with names
        params_with_names = [(name, param) for name, param in self.model_unwrapped.named_parameters()
                            if param.requires_grad]

        if not params_with_names:
            return {'per_parameter': param_conflicts, 'summary': {}}

        all_params = [p for _, p in params_with_names]
        param_names = [n for n, _ in params_with_names]

        # Collect gradients per task using autograd.grad
        task_gradients = {}
        for task_name, loss in losses.items():
            # Skip if loss is zero
            if not isinstance(loss, torch.Tensor) or loss.abs().item() < 1e-10:
                continue

            try:
                grads = torch.autograd.grad(
                    outputs=loss,
                    inputs=all_params,
                    retain_graph=True,
                    allow_unused=True
                )

                task_gradients[task_name] = {}
                for name, grad in zip(param_names, grads):
                    if grad is not None:
                        task_gradients[task_name][name] = grad.detach().clone()

            except RuntimeError as e:
                print(f"[DetailedAnalyzer] Warning: Could not compute param gradient for {task_name}: {e}")
                continue

        # If no gradients were computed, return empty
        if not task_gradients:
            return {'per_parameter': param_conflicts, 'summary': {}}

        # Find first task with gradients
        first_task_with_grads = None
        for task_name in task_names:
            if task_name in task_gradients and task_gradients[task_name]:
                first_task_with_grads = task_name
                break

        if first_task_with_grads is None:
            return {'per_parameter': param_conflicts, 'summary': {}}

        # Use only tasks that actually have gradients (handles DET_ONLY, SEG_ONLY, etc.)
        active_tasks = list(task_gradients.keys())

        # Analyze each parameter
        for param_name in task_gradients[first_task_with_grads].keys():
            param_metrics = {}

            # Get gradients for this parameter from active tasks only
            grads = {}
            for task in active_tasks:
                if param_name in task_gradients[task]:
                    grads[task] = task_gradients[task].get(param_name)

            if len(grads) < 2:
                continue

            # Compute statistics using active tasks only
            for i, task_i in enumerate(active_tasks):
                for j, task_j in enumerate(active_tasks):
                    if i >= j or task_i not in grads or task_j not in grads:
                        continue

                    grad_i = grads[task_i].flatten()
                    grad_j = grads[task_j].flatten()

                    # Element-wise sign agreement
                    sign_agreement = (torch.sign(grad_i) == torch.sign(grad_j)).float().mean().item()

                    # Cosine similarity
                    cos_sim = torch.nn.functional.cosine_similarity(
                        grad_i.unsqueeze(0), grad_j.unsqueeze(0)
                    ).item()

                    conflict_key = f"{task_i}_vs_{task_j}"
                    param_metrics[conflict_key] = {
                        'cosine_similarity': cos_sim,
                        'sign_agreement': sign_agreement,
                        'conflict_ratio': 1.0 - sign_agreement,
                        'is_conflict': cos_sim < 0
                    }

            param_conflicts[param_name] = param_metrics

        # Generate summary statistics
        summary = self._generate_param_conflict_summary(param_conflicts)

        return {
            'per_parameter': param_conflicts,
            'summary': summary
        }

    def _generate_param_conflict_summary(self, param_conflicts: Dict) -> Dict[str, Any]:
        """Generate summary statistics for parameter conflicts."""
        summary = {
            'total_params_analyzed': len(param_conflicts),
            'most_conflicted_params': [],
            'least_conflicted_params': [],
            'avg_conflict_ratio': 0.0,
            'conflict_distribution': defaultdict(int)
        }

        # Calculate average conflict for each parameter
        param_avg_conflicts = {}
        for param_name, conflicts in param_conflicts.items():
            if conflicts:
                avg_conflict = np.mean([
                    metrics['conflict_ratio']
                    for metrics in conflicts.values()
                ])
                param_avg_conflicts[param_name] = avg_conflict

        if param_avg_conflicts:
            # Sort by conflict level
            sorted_params = sorted(param_avg_conflicts.items(),
                                 key=lambda x: x[1], reverse=True)

            summary['most_conflicted_params'] = sorted_params[:10]  # Top 10
            summary['least_conflicted_params'] = sorted_params[-10:]  # Bottom 10
            summary['avg_conflict_ratio'] = np.mean(list(param_avg_conflicts.values()))

            # Distribution
            for conflict_ratio in param_avg_conflicts.values():
                bucket = int(conflict_ratio * 10) / 10  # 0.0, 0.1, 0.2, ...
                summary['conflict_distribution'][f"{bucket:.1f}"] += 1

        return summary

    def _track_gradient_trajectory(self, losses: Dict[str, torch.Tensor],
                                   optimizer: torch.optim.Optimizer):
        """Track gradient direction changes over time.

        Uses torch.autograd.grad() to compute gradients without consuming the computation graph.
        """
        # Get all parameters as a flat list
        all_params = []
        param_to_layer = {}
        for layer_name, params in self.layer_params.items():
            for param_name, param in params:
                if param.requires_grad:
                    all_params.append(param)
                    param_to_layer[id(param)] = layer_name

        if not all_params:
            return

        for task_name, loss in losses.items():
            # Skip if loss is zero
            if not isinstance(loss, torch.Tensor) or loss.abs().item() < 1e-10:
                continue

            try:
                # Use autograd.grad to compute gradients
                grads = torch.autograd.grad(
                    outputs=loss,
                    inputs=all_params,
                    retain_graph=True,
                    allow_unused=True
                )

                # Organize gradients by layer
                layer_grads_dict = {}
                for param, grad in zip(all_params, grads):
                    if grad is None:
                        continue
                    layer_name = param_to_layer[id(param)]
                    if layer_name not in layer_grads_dict:
                        layer_grads_dict[layer_name] = []
                    layer_grads_dict[layer_name].append(grad.detach().flatten())

                # Store trajectories for each layer
                for layer_name, layer_grads in layer_grads_dict.items():
                    if not layer_grads:
                        continue

                    # Concatenate and normalize
                    grad_vec = torch.cat(layer_grads)
                    grad_norm = torch.norm(grad_vec)

                    if grad_norm > 1e-8:
                        grad_vec = grad_vec / grad_norm  # Normalize to unit vector

                    # Store in history
                    key = f"{task_name}_{layer_name}"
                    self.gradient_history[key].append({
                        'step': self.step_count,
                        'epoch': self.epoch_count,
                        'gradient': grad_vec.cpu().numpy(),
                        'norm': grad_norm.item()
                    })

            except RuntimeError as e:
                print(f"[DetailedAnalyzer] Warning: Could not track trajectory for {task_name}: {e}")
                continue

    def _get_trajectory_summary(self) -> Dict[str, Any]:
        """Get summary of gradient trajectories."""
        summary = {}

        for key, history in self.gradient_history.items():
            if len(history) < 2:
                continue

            # Compute angle changes between consecutive steps
            angles = []
            for i in range(1, len(history)):
                prev_grad = torch.from_numpy(history[i-1]['gradient'])
                curr_grad = torch.from_numpy(history[i]['gradient'])

                cos_angle = torch.dot(prev_grad, curr_grad).item()
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # Radians
                angles.append(np.degrees(angle))  # Convert to degrees

            if angles:
                summary[key] = {
                    'avg_angle_change': np.mean(angles),
                    'max_angle_change': np.max(angles),
                    'std_angle_change': np.std(angles),
                    'total_steps': len(history),
                    'is_oscillating': np.std(angles) > 45.0  # High variance indicates oscillation
                }

        return summary

    def _analyze_feature_conflicts(self, features: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Analyze spatial conflicts in feature maps.

        Args:
            features: Dict mapping layer names to feature tensors [B, C, H, W]
        """
        feature_conflicts = {}

        # Assuming features contain outputs from different task heads
        # This is a placeholder - actual implementation depends on your feature format
        for layer_name, feature_map in features.items():
            if feature_map.dim() != 4:  # Expect [B, C, H, W]
                continue

            # Compute spatial statistics
            spatial_stats = {
                'mean': feature_map.mean().item(),
                'std': feature_map.std().item(),
                'max': feature_map.max().item(),
                'min': feature_map.min().item(),
                'shape': list(feature_map.shape)
            }

            feature_conflicts[layer_name] = spatial_stats

            # Store for later analysis
            self.feature_history[layer_name].append({
                'step': self.step_count,
                'epoch': self.epoch_count,
                'stats': spatial_stats
            })

        return feature_conflicts

    def _save_results(self, results: Dict[str, Any], step: int, epoch: int):
        """Save analysis results to disk."""
        # Save as JSON
        output_file = self.output_dir / 'raw_data' / f'step_{step:06d}_epoch_{epoch:03d}.json'

        # Convert to serializable format
        serializable_results = self._make_serializable(results)

        with open(output_file, 'w') as f:
            json.dump({
                'step': step,
                'epoch': epoch,
                'results': serializable_results
            }, f, indent=2)

        print(f"[DetailedAnalyzer] Saved analysis to {output_file}")

    def _make_serializable(self, obj):
        """Convert numpy/torch objects to Python native types."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist() if obj.size < 1000 else f"<array shape={obj.shape}>"
        elif isinstance(obj, torch.Tensor):
            return obj.tolist() if obj.numel() < 1000 else f"<tensor shape={tuple(obj.shape)}>"
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj

    def generate_summary_report(self, output_file: Optional[str] = None):
        """Generate comprehensive summary report."""
        if output_file is None:
            output_file = self.output_dir / 'summary_report.json'

        report = {
            'total_steps_analyzed': self.step_count,
            'total_epochs': self.epoch_count,
            'layers_tracked': self.layer_names,
            'gradient_trajectory_summary': self._get_trajectory_summary(),
            'conflict_history_summary': {
                key: {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                } for key, values in self.conflict_history.items() if values
            }
        }

        with open(output_file, 'w') as f:
            json.dump(self._make_serializable(report), f, indent=2)

        print(f"[DetailedAnalyzer] Summary report saved to {output_file}")
        return report
