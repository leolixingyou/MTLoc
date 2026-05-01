"""
Visualization Integration Module

Provides seamless integration of the new visualization system into training.
Outputs are organized into four categories:
1. network_structure/ - ONNX for Netron
2. spatial_attention/ - Grad-CAM heatmaps
3. feature_distribution/ - TensorBoard Projector embeddings
4. layer_contribution/ - Captum-style attribution
"""

import torch
import os
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def init_visualization(cfg, model, final_output_dir, train_logger) -> Optional[Dict[str, Any]]:
    """
    Initialize visualization system if enabled in config.

    Args:
        cfg: Configuration object
        model: The model being trained
        final_output_dir: Output directory for this run
        train_logger: Logger instance

    Returns:
        Dictionary containing visualization components, or None if disabled
    """
    # Check if visualization is enabled
    if not hasattr(cfg.TRAIN, 'VISUALIZATION'):
        return None

    viz_cfg = cfg.TRAIN.VISUALIZATION
    if not getattr(viz_cfg, 'ENABLED', False):
        return None

    train_logger.info("=" * 80)
    train_logger.info("MTL VISUALIZATION SYSTEM ENABLED")
    train_logger.info("=" * 80)

    # Import visualization module
    try:
        from lib.core.visualization import MTLVisualizer
    except ImportError as e:
        train_logger.error(f"Failed to import visualization module: {e}")
        return None

    # Create visualization output directory
    viz_output_dir = Path(final_output_dir)

    # Extract configuration
    analysis_interval = getattr(viz_cfg, 'ANALYSIS_INTERVAL', 100)
    enable_network = getattr(viz_cfg, 'NETWORK_STRUCTURE', True)
    enable_attention = getattr(viz_cfg, 'SPATIAL_ATTENTION', True)
    enable_features = getattr(viz_cfg, 'FEATURE_DISTRIBUTION', True)
    enable_contribution = getattr(viz_cfg, 'LAYER_CONTRIBUTION', True)

    train_logger.info(f"Visualization output directory: {viz_output_dir}")
    train_logger.info(f"  - Network structure: {enable_network}")
    train_logger.info(f"  - Spatial attention (Grad-CAM): {enable_attention}")
    train_logger.info(f"  - Feature distribution: {enable_features}")
    train_logger.info(f"  - Layer contribution: {enable_contribution}")
    train_logger.info(f"  - Analysis interval: {analysis_interval} steps")

    # Initialize MTLVisualizer
    try:
        visualizer = MTLVisualizer(
            model=model,
            output_dir=str(viz_output_dir),
            cfg=cfg,
            enable_network=enable_network,
            enable_attention=enable_attention,
            enable_features=enable_features,
            enable_contribution=enable_contribution,
            analysis_interval=analysis_interval
        )

        # Export network structure at initialization
        if enable_network:
            input_shape = (1, 3, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
            visualizer.export_network_structure(input_shape)
            train_logger.info("  - Network structure exported (open with Netron)")

    except Exception as e:
        train_logger.error(f"Failed to initialize visualizer: {e}")
        import traceback
        train_logger.error(traceback.format_exc())
        return None

    # Create components dictionary
    viz_components = {
        'visualizer': visualizer,
        'config': viz_cfg,
        'output_dir': viz_output_dir,
        'step_counter': 0
    }

    train_logger.info("=" * 80)
    train_logger.info("Visualization system initialized successfully")
    train_logger.info("Output folders:")
    train_logger.info(f"  - {viz_output_dir}/network_structure/")
    train_logger.info(f"  - {viz_output_dir}/spatial_attention/")
    train_logger.info(f"  - {viz_output_dir}/feature_distribution/")
    train_logger.info(f"  - {viz_output_dir}/layer_contribution/")
    train_logger.info("=" * 80)

    return viz_components


def run_visualization_analysis(viz_components: Optional[Dict],
                               images: torch.Tensor,
                               step: int,
                               epoch: int,
                               train_logger=None) -> Optional[Dict[str, Any]]:
    """
    Run visualization analysis for current training step.

    This function is SAFE to call during training - it will not modify
    the model state, input tensors, or interfere with the training loop.

    Args:
        viz_components: Visualization components from init_visualization
        images: Input images tensor
        step: Current step number
        epoch: Current epoch number
        train_logger: Logger instance

    Returns:
        Analysis results dictionary, or None if visualization disabled
    """
    if viz_components is None:
        return None

    visualizer = viz_components['visualizer']

    try:
        # Run step analysis (safe - will restore model state)
        visualizer.analyze_step(images, epoch, step)

        # Periodically log summary (less frequent to avoid overhead)
        cfg = viz_components['config']
        log_interval = getattr(cfg, 'LOG_INTERVAL', 500)

        if step % log_interval == 0 and train_logger:
            # Create a detached clone for summary to avoid any side effects
            images_for_summary = images.clone().detach()
            summary = visualizer.get_analysis_summary(images_for_summary)
            if summary:
                _log_visualization_summary(summary, step, epoch, train_logger)

        return {'status': 'success'}

    except Exception as e:
        if train_logger:
            train_logger.warning(f"Visualization analysis failed at step {step}: {e}")
        # Return None but don't crash - training should continue
        return None


def finalize_visualization(viz_components: Optional[Dict],
                           epoch: int,
                           train_logger=None):
    """
    Finalize epoch and save summary.

    Args:
        viz_components: Visualization components
        epoch: Current epoch
        train_logger: Logger instance
    """
    if viz_components is None:
        return

    visualizer = viz_components['visualizer']
    output_dir = viz_components['output_dir']

    try:
        # Save epoch summary
        visualizer.save_epoch_summary(epoch)

        if train_logger:
            train_logger.info(f"Epoch {epoch} visualization summary saved")

    except Exception as e:
        if train_logger:
            train_logger.warning(f"Visualization finalization failed: {e}")


def cleanup_visualization(viz_components: Optional[Dict], train_logger=None):
    """
    Cleanup visualization resources.

    Args:
        viz_components: Visualization components
        train_logger: Logger instance
    """
    if viz_components is None:
        return

    try:
        visualizer = viz_components['visualizer']
        visualizer.cleanup()

        if train_logger:
            train_logger.info("Visualization system cleaned up successfully")

    except Exception as e:
        if train_logger:
            train_logger.warning(f"Visualization cleanup failed: {e}")


def _log_visualization_summary(summary: Dict, step: int, epoch: int, train_logger):
    """Log summary of visualization analysis."""
    train_logger.info(f"\n{'='*60}")
    train_logger.info(f"VISUALIZATION SUMMARY - Step {step}, Epoch {epoch}")
    train_logger.info(f"{'='*60}")

    # Attention overlap
    if 'attention_overlap' in summary:
        train_logger.info("\n[Attention Overlap]")
        for key, value in summary['attention_overlap'].items():
            train_logger.info(f"  {key}: {value:.4f}")

    # Layer sharing
    if 'layer_sharing' in summary:
        train_logger.info("\n[Layer Sharing Scores]")
        for layer, score in list(summary['layer_sharing'].items())[:5]:
            status = "shared" if score > 0.7 else "specialized" if score < 0.3 else "mixed"
            train_logger.info(f"  {layer}: {score:.4f} ({status})")

    train_logger.info(f"{'='*60}\n")
