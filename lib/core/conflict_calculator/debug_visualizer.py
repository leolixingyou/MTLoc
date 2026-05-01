"""
Advanced Visualization Tools for Detailed Conflict Analysis

This module provides visualization utilities for:
1. Parameter-level conflict heatmaps
2. Gradient trajectory plots (3D and 2D)
3. Layer-wise conflict evolution
4. Feature map spatial analysis
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
from collections import defaultdict


class DebugVisualizer:
    """
    Advanced visualizer for detailed conflict analysis.

    Generates publication-quality plots and heatmaps for understanding
    multi-task learning conflicts at multiple granularities.
    """

    def __init__(self, output_dir: str, dpi: int = 150):
        """
        Initialize the visualizer.

        Args:
            output_dir: Directory to save visualization outputs
            dpi: Resolution for saved figures
        """
        self.output_dir = Path(output_dir)
        self.dpi = dpi

        # Create subdirectories
        (self.output_dir / 'heatmaps').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'trajectories').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'layer_evolution').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'summary_plots').mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)

        print(f"[DebugVisualizer] Initialized at {self.output_dir}")

    def plot_parameter_conflict_heatmap(self, param_conflicts: Dict[str, Dict],
                                       step: int, epoch: int,
                                       top_n: int = 50):
        """
        Generate heatmap of parameter-level conflicts.

        Args:
            param_conflicts: Output from DetailedAnalyzer._analyze_parameter_conflicts
            step: Current step
            epoch: Current epoch
            top_n: Number of most conflicted parameters to show
        """
        # Extract conflict ratios
        param_scores = {}
        for param_name, conflicts in param_conflicts.items():
            if conflicts:
                avg_conflict = np.mean([
                    metrics['conflict_ratio']
                    for metrics in conflicts.values()
                ])
                param_scores[param_name] = avg_conflict

        if not param_scores:
            print("[DebugVisualizer] No parameter conflicts to visualize")
            return

        # Sort and select top N
        sorted_params = sorted(param_scores.items(), key=lambda x: x[1], reverse=True)
        top_params = sorted_params[:top_n]

        # Create heatmap data
        param_names = [name for name, _ in top_params]
        conflict_values = [score for _, score in top_params]

        # Plot
        fig, ax = plt.subplots(figsize=(14, max(8, len(param_names) * 0.3)))

        # Create horizontal bar chart (better for long parameter names)
        y_pos = np.arange(len(param_names))
        colors = plt.cm.RdYlGn_r(np.array(conflict_values))

        ax.barh(y_pos, conflict_values, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([self._shorten_param_name(name) for name in param_names],
                          fontsize=8)
        ax.set_xlabel('Conflict Ratio', fontsize=12)
        ax.set_title(f'Top {top_n} Most Conflicted Parameters\nStep {step}, Epoch {epoch}',
                    fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1.0)
        ax.grid(axis='x', alpha=0.3)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('Conflict Severity', fontsize=10)

        plt.tight_layout()

        # Save
        output_file = self.output_dir / 'heatmaps' / f'param_conflict_step{step:06d}.png'
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"[DebugVisualizer] Saved parameter heatmap to {output_file}")

    def plot_layer_conflict_evolution(self, layer_conflicts_history: List[Dict],
                                      save_name: str = 'layer_evolution'):
        """
        Plot how layer conflicts evolve over training.

        Args:
            layer_conflicts_history: List of layer_conflicts dicts over time
            save_name: Name for saved figure
        """
        if not layer_conflicts_history:
            return

        # Extract data
        layers = set()
        for lc in layer_conflicts_history:
            layers.update(lc.keys())
        layers = sorted(list(layers))

        # Prepare data for each task pair
        task_pairs = set()
        for lc in layer_conflicts_history:
            for layer_data in lc.values():
                task_pairs.update(layer_data.keys())
        task_pairs = sorted(list(task_pairs))

        # Create subplots for each task pair
        n_pairs = len(task_pairs)
        fig, axes = plt.subplots(n_pairs, 1, figsize=(14, 5 * n_pairs), sharex=True)
        if n_pairs == 1:
            axes = [axes]

        for ax, task_pair in zip(axes, task_pairs):
            # Extract cosine similarities over time for each layer
            for layer_name in layers:
                cos_sims = []
                steps = []

                for idx, lc in enumerate(layer_conflicts_history):
                    if layer_name in lc and task_pair in lc[layer_name]:
                        cos_sims.append(lc[layer_name][task_pair]['cosine_similarity'])
                        steps.append(idx)

                if cos_sims:
                    ax.plot(steps, cos_sims, marker='o', label=self._shorten_layer_name(layer_name),
                           alpha=0.7, linewidth=2, markersize=4)

            # Styling
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Conflict Threshold')
            ax.set_ylabel('Cosine Similarity', fontsize=11)
            ax.set_title(f'Task Pair: {task_pair}', fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Training Step', fontsize=12)
        fig.suptitle('Layer-wise Gradient Conflict Evolution', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])

        # Save
        output_file = self.output_dir / 'layer_evolution' / f'{save_name}.png'
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"[DebugVisualizer] Saved layer evolution to {output_file}")

    def plot_gradient_trajectory_2d(self, gradient_history: Dict[str, List],
                                    step_range: Optional[Tuple[int, int]] = None,
                                    projection_dims: Tuple[int, int] = (0, 1)):
        """
        Plot 2D projection of gradient trajectories.

        Args:
            gradient_history: Dict from DetailedAnalyzer.gradient_history
            step_range: (start_step, end_step) to plot, None for all
            projection_dims: Which two dimensions to project onto
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()

        # Group by task
        tasks = defaultdict(list)
        for key in gradient_history.keys():
            task_name = key.split('_')[0]  # Assuming format: taskname_layername
            tasks[task_name].append(key)

        task_names = sorted(tasks.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(task_names)))

        for ax_idx, (ax, task_name) in enumerate(zip(axes, task_names)):
            task_color = colors[ax_idx]

            for key in tasks[task_name]:
                history = gradient_history[key]

                if step_range:
                    history = [h for h in history if step_range[0] <= h['step'] <= step_range[1]]

                if len(history) < 2:
                    continue

                # Extract gradients and project to 2D
                gradients = np.array([h['gradient'] for h in history])
                steps = [h['step'] for h in history]

                # Simple projection: take first two dimensions
                if gradients.shape[1] > max(projection_dims):
                    proj_x = gradients[:, projection_dims[0]]
                    proj_y = gradients[:, projection_dims[1]]

                    # Plot trajectory
                    ax.plot(proj_x, proj_y, alpha=0.6, linewidth=1.5,
                           color=task_color, label=self._shorten_layer_name(key))

                    # Mark start and end
                    ax.scatter(proj_x[0], proj_y[0], marker='o', s=100,
                              color=task_color, edgecolors='black', linewidth=2, zorder=5)
                    ax.scatter(proj_x[-1], proj_y[-1], marker='s', s=100,
                              color=task_color, edgecolors='black', linewidth=2, zorder=5)

            ax.set_xlabel(f'Gradient Dim {projection_dims[0]}', fontsize=11)
            ax.set_ylabel(f'Gradient Dim {projection_dims[1]}', fontsize=11)
            ax.set_title(f'Task: {task_name}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
            ax.axvline(x=0, color='k', linestyle='-', alpha=0.2)

        plt.suptitle('Gradient Trajectory Evolution (2D Projection)', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        # Save
        output_file = self.output_dir / 'trajectories' / 'gradient_trajectory_2d.png'
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"[DebugVisualizer] Saved 2D trajectory to {output_file}")

    def plot_conflict_summary_dashboard(self, analysis_data: Dict[str, Any],
                                       step: int, epoch: int):
        """
        Generate comprehensive dashboard with multiple conflict metrics.

        Args:
            analysis_data: Complete output from DetailedAnalyzer.analyze_step
            step: Current step
            epoch: Current epoch
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Layer-wise conflict matrix (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_layer_conflict_matrix(ax1, analysis_data.get('layer_conflicts', {}))

        # 2. Parameter conflict distribution (top-middle)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_param_conflict_distribution(ax2, analysis_data.get('param_conflicts', {}))

        # 3. Gradient trajectory angles (top-right)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_trajectory_angles(ax3, analysis_data.get('gradient_trajectory', {}))

        # 4. Conflict severity by layer (middle-left)
        ax4 = fig.add_subplot(gs[1, :])
        self._plot_conflict_severity_by_layer(ax4, analysis_data.get('layer_conflicts', {}))

        # 5. Top conflicted parameters table (bottom)
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_top_conflicts_table(ax5, analysis_data.get('param_conflicts', {}))

        # Overall title
        fig.suptitle(f'Conflict Analysis Dashboard - Step {step}, Epoch {epoch}',
                    fontsize=18, fontweight='bold')

        # Save
        output_file = self.output_dir / 'summary_plots' / f'dashboard_step{step:06d}.png'
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"[DebugVisualizer] Saved dashboard to {output_file}")

    def _plot_layer_conflict_matrix(self, ax, layer_conflicts: Dict):
        """Helper: Plot layer conflict matrix."""
        if not layer_conflicts:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
            ax.set_title('Layer Conflict Matrix')
            return

        # Extract average cosine similarity per layer
        layers = list(layer_conflicts.keys())
        n_layers = len(layers)

        matrix = np.zeros((n_layers, n_layers))
        for i, layer in enumerate(layers):
            for conflict_key, metrics in layer_conflicts[layer].items():
                matrix[i, i] = metrics.get('cosine_similarity', 0)

        sns.heatmap(matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                   xticklabels=[self._shorten_layer_name(l) for l in layers],
                   yticklabels=[self._shorten_layer_name(l) for l in layers],
                   ax=ax, cbar_kws={'label': 'Cos Similarity'})
        ax.set_title('Layer Conflict Matrix', fontweight='bold')

    def _plot_param_conflict_distribution(self, ax, param_conflicts: Dict):
        """Helper: Plot parameter conflict distribution."""
        summary = param_conflicts.get('summary', {})
        distribution = summary.get('conflict_distribution', {})

        if not distribution:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
            ax.set_title('Parameter Conflict Distribution')
            return

        bins = sorted(distribution.keys())
        counts = [distribution[b] for b in bins]

        ax.bar(bins, counts, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Conflict Ratio Range', fontsize=11)
        ax.set_ylabel('Number of Parameters', fontsize=11)
        ax.set_title('Parameter Conflict Distribution', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    def _plot_trajectory_angles(self, ax, trajectory_summary: Dict):
        """Helper: Plot gradient trajectory angle changes."""
        if not trajectory_summary:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
            ax.set_title('Gradient Trajectory Angles')
            return

        keys = list(trajectory_summary.keys())[:10]  # Top 10
        avg_angles = [trajectory_summary[k]['avg_angle_change'] for k in keys]
        max_angles = [trajectory_summary[k]['max_angle_change'] for k in keys]

        x = np.arange(len(keys))
        width = 0.35

        ax.bar(x - width/2, avg_angles, width, label='Avg Angle', alpha=0.7)
        ax.bar(x + width/2, max_angles, width, label='Max Angle', alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels([self._shorten_layer_name(k) for k in keys], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Angle (degrees)', fontsize=11)
        ax.set_title('Gradient Direction Changes', fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    def _plot_conflict_severity_by_layer(self, ax, layer_conflicts: Dict):
        """Helper: Plot conflict severity across layers."""
        if not layer_conflicts:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
            ax.set_title('Conflict Severity by Layer')
            return

        layers = sorted(layer_conflicts.keys())
        severities = []

        for layer in layers:
            layer_severities = [
                metrics.get('conflict_severity', 0)
                for metrics in layer_conflicts[layer].values()
            ]
            severities.append(np.mean(layer_severities) if layer_severities else 0)

        colors = plt.cm.RdYlGn_r(np.array(severities) / (max(severities) + 1e-8))
        ax.bar(range(len(layers)), severities, color=colors, edgecolor='black', alpha=0.8)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([self._shorten_layer_name(l) for l in layers], rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Average Conflict Severity', fontsize=12)
        ax.set_title('Conflict Severity by Layer', fontweight='bold', fontsize=13)
        ax.grid(axis='y', alpha=0.3)

    def _plot_top_conflicts_table(self, ax, param_conflicts: Dict):
        """Helper: Display table of most conflicted parameters."""
        summary = param_conflicts.get('summary', {})
        top_conflicts = summary.get('most_conflicted_params', [])[:10]

        if not top_conflicts:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
            ax.set_title('Top Conflicted Parameters')
            ax.axis('off')
            return

        # Prepare table data
        table_data = []
        for idx, (param_name, conflict_ratio) in enumerate(top_conflicts, 1):
            table_data.append([
                str(idx),
                self._shorten_param_name(param_name, max_len=40),
                f'{conflict_ratio:.4f}'
            ])

        table = ax.table(cellText=table_data,
                        colLabels=['Rank', 'Parameter Name', 'Conflict Ratio'],
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.1, 0.7, 0.2])

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Color cells by conflict ratio
        for i in range(1, len(table_data) + 1):
            conflict_val = float(table_data[i-1][2])
            color = plt.cm.RdYlGn_r(conflict_val)
            table[(i, 2)].set_facecolor(color)

        ax.set_title('Top 10 Most Conflicted Parameters', fontweight='bold', fontsize=13, pad=20)
        ax.axis('off')

    def _shorten_param_name(self, name: str, max_len: int = 30) -> str:
        """Shorten parameter name for display."""
        if len(name) <= max_len:
            return name
        parts = name.split('.')
        if len(parts) > 3:
            return f"{parts[0]}...{'.'.join(parts[-2:])}"
        return name[:max_len-3] + '...'

    def _shorten_layer_name(self, name: str, max_len: int = 15) -> str:
        """Shorten layer name for display."""
        if len(name) <= max_len:
            return name
        return name[:max_len-3] + '...'
