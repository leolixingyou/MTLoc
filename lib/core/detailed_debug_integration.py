"""
Integration module for thorough debugging system

This module provides seamless integration of DetailedConflictAnalyzer
into the main training loop without modifying core training logic.
"""

import torch
import os
from pathlib import Path
from typing import Optional, Dict, Any
from lib.core.conflict_calculator import DetailedConflictAnalyzer, DebugVisualizer


def init_detailed_debug(cfg, model, final_output_dir, logger) -> Optional[Dict[str, Any]]:
    """
    Initialize detailed debugging system if enabled in config.

    Args:
        cfg: Configuration object
        model: The model being trained
        final_output_dir: Output directory for this run
        logger: Logger instance

    Returns:
        Dictionary containing debug components, or None if disabled
    """
    # Check if detailed debugging is enabled
    if not hasattr(cfg.TRAIN, 'DETAILED_DEBUG'):
        return None

    detailed_cfg = cfg.TRAIN.DETAILED_DEBUG
    if not getattr(detailed_cfg, 'ENABLED', False):
        return None

    logger.info("=" * 80)
    logger.info("DETAILED DEBUGGING MODE ENABLED")
    logger.info("=" * 80)

    # Create debug output directory
    debug_output_dir = Path(final_output_dir) / 'detailed_debug'
    debug_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Debug output directory: {debug_output_dir}")

    # Extract configuration
    save_interval = getattr(detailed_cfg, 'SAVE_INTERVAL', 1)
    track_layers = getattr(detailed_cfg, 'TRACK_LAYERS', None)

    # Initialize DetailedConflictAnalyzer
    logger.info("Initializing DetailedConflictAnalyzer...")
    analyzer = DetailedConflictAnalyzer(
        model=model,
        output_dir=str(debug_output_dir),
        track_layers=track_layers,
        save_interval=save_interval
    )

    # Initialize DebugVisualizer
    logger.info("Initializing DebugVisualizer...")
    visualizer = DebugVisualizer(
        output_dir=str(debug_output_dir),
        dpi=150
    )

    # Create components dictionary
    debug_components = {
        'analyzer': analyzer,
        'visualizer': visualizer,
        'config': detailed_cfg,
        'output_dir': debug_output_dir,
        'history': {
            'layer_conflicts': [],
            'param_conflicts': [],
            'gradient_trajectories': [],
            'feature_conflicts': []
        }
    }

    logger.info("Detailed debugging system initialized successfully")
    logger.info(f"  - Layer-wise analysis: {getattr(detailed_cfg, 'LAYER_WISE_ANALYSIS', False)}")
    logger.info(f"  - Parameter-level analysis: {getattr(detailed_cfg, 'PARAM_LEVEL_ANALYSIS', False)}")
    logger.info(f"  - Gradient trajectory tracking: {getattr(detailed_cfg, 'TRACK_GRADIENT_TRAJECTORY', False)}")
    logger.info(f"  - Feature map saving: {getattr(detailed_cfg, 'SAVE_FEATURE_MAPS', False)}")
    logger.info("=" * 80)

    return debug_components


def run_detailed_analysis(debug_components: Optional[Dict],
                         losses: Dict[str, torch.Tensor],
                         optimizer: torch.optim.Optimizer,
                         step: int,
                         epoch: int,
                         features: Optional[Dict] = None,
                         logger = None) -> Optional[Dict[str, Any]]:
    """
    Run detailed conflict analysis for current training step.

    Args:
        debug_components: Debug components from init_detailed_debug
        losses: Task losses dictionary
        optimizer: Optimizer instance
        step: Current step number
        epoch: Current epoch number
        features: Optional feature maps
        logger: Logger instance

    Returns:
        Analysis results dictionary, or None if debugging disabled
    """
    if debug_components is None:
        return None

    analyzer = debug_components['analyzer']
    visualizer = debug_components['visualizer']
    cfg = debug_components['config']
    history = debug_components['history']

    try:
        # Run comprehensive analysis
        if logger and step == 0:
            logger.info(f"Running detailed analysis at step {step}, epoch {epoch}...")

        results = analyzer.analyze_step(
            losses=losses,
            optimizer=optimizer,
            step=step,
            epoch=epoch,
            features=features
        )

        # Store in history
        if 'layer_conflicts' in results:
            history['layer_conflicts'].append(results['layer_conflicts'])
        if 'param_conflicts' in results:
            history['param_conflicts'].append(results['param_conflicts'])
        if 'gradient_trajectory' in results:
            history['gradient_trajectories'].append(results['gradient_trajectory'])
        if 'feature_conflicts' in results:
            history['feature_conflicts'].append(results['feature_conflicts'])

        # Generate visualizations based on config
        viz_interval = getattr(cfg, 'VISUALIZE_EVERY_N_STEPS', 1)

        if step % viz_interval == 0:
            # Generate dashboard
            if getattr(cfg, 'GENERATE_DASHBOARD', False):
                visualizer.plot_conflict_summary_dashboard(
                    analysis_data=results,
                    step=step,
                    epoch=epoch
                )

            # Generate parameter heatmaps
            if getattr(cfg, 'GENERATE_HEATMAPS', False) and 'param_conflicts' in results:
                top_n = getattr(cfg, 'PARAM_HEATMAP_TOP_N', 50)
                param_data = results['param_conflicts'].get('per_parameter', {})
                if param_data:
                    visualizer.plot_parameter_conflict_heatmap(
                        param_conflicts=param_data,
                        step=step,
                        epoch=epoch,
                        top_n=top_n
                    )

            # Generate trajectory plots
            if getattr(cfg, 'GENERATE_TRAJECTORIES', False) and len(history['layer_conflicts']) > 1:
                visualizer.plot_layer_conflict_evolution(
                    layer_conflicts_history=history['layer_conflicts'],
                    save_name=f'evolution_epoch{epoch}'
                )

                # 2D trajectory plot
                if len(analyzer.gradient_history) > 0:
                    visualizer.plot_gradient_trajectory_2d(
                        gradient_history=analyzer.gradient_history
                    )

        # Log summary
        if logger and step % cfg.SAVE_INTERVAL == 0:
            _log_analysis_summary(results, step, epoch, logger)

        return results

    except Exception as e:
        if logger:
            logger.error(f"Detailed analysis failed at step {step}: {e}")
            import traceback
            logger.error(traceback.format_exc())
        return None


def finalize_detailed_debug(debug_components: Optional[Dict], logger = None):
    """
    Generate final reports and cleanup.

    Args:
        debug_components: Debug components from init_detailed_debug
        logger: Logger instance
    """
    if debug_components is None:
        return

    analyzer = debug_components['analyzer']
    visualizer = debug_components['visualizer']
    output_dir = debug_components['output_dir']
    history = debug_components['history']

    if logger:
        logger.info("=" * 80)
        logger.info("Finalizing detailed debugging analysis...")

    try:
        # Generate summary report
        summary_file = output_dir / 'final_summary_report.json'
        analyzer.generate_summary_report(str(summary_file))

        if logger:
            logger.info(f"  ✓ Summary report saved to {summary_file}")

        # Generate final visualizations
        if len(history['layer_conflicts']) > 1:
            visualizer.plot_layer_conflict_evolution(
                layer_conflicts_history=history['layer_conflicts'],
                save_name='final_evolution_all_epochs'
            )
            if logger:
                logger.info(f"  ✓ Final layer evolution plot generated")

        if len(analyzer.gradient_history) > 0:
            visualizer.plot_gradient_trajectory_2d(
                gradient_history=analyzer.gradient_history
            )
            if logger:
                logger.info(f"  ✓ Final gradient trajectory plot generated")

        # Save history
        import json
        history_file = output_dir / 'analysis_history.json'
        with open(history_file, 'w') as f:
            # Convert to serializable format
            serializable_history = {
                'total_steps': len(history['layer_conflicts']),
                'layer_conflicts_count': len(history['layer_conflicts']),
                'param_conflicts_count': len(history['param_conflicts']),
                'trajectory_count': len(history['gradient_trajectories'])
            }
            json.dump(serializable_history, f, indent=2)

        if logger:
            logger.info(f"  ✓ Analysis history saved to {history_file}")
            logger.info("=" * 80)
            logger.info("DETAILED DEBUGGING FINALIZED")
            logger.info(f"All results saved to: {output_dir}")
            logger.info("=" * 80)

    except Exception as e:
        if logger:
            logger.error(f"Error during finalization: {e}")
            import traceback
            logger.error(traceback.format_exc())


def _log_analysis_summary(results: Dict[str, Any], step: int, epoch: int, logger):
    """Log summary of analysis results."""
    logger.info(f"\n{'='*80}")
    logger.info(f"DETAILED ANALYSIS SUMMARY - Step {step}, Epoch {epoch}")
    logger.info(f"{'='*80}")

    # Layer conflicts summary
    if 'layer_conflicts' in results and results['layer_conflicts']:
        logger.info(f"\n[Layer-wise Conflicts]")
        for layer_name, conflicts in list(results['layer_conflicts'].items())[:5]:  # Top 5
            logger.info(f"  {layer_name}:")
            for pair, metrics in conflicts.items():
                cos_sim = metrics.get('cosine_similarity', 0)
                severity = metrics.get('conflict_severity', 0)
                logger.info(f"    {pair}: cos_sim={cos_sim:.4f}, severity={severity:.4f}")

    # Parameter conflicts summary
    if 'param_conflicts' in results:
        summary = results['param_conflicts'].get('summary', {})
        if summary:
            logger.info(f"\n[Parameter-level Conflicts]")
            logger.info(f"  Total parameters analyzed: {summary.get('total_params_analyzed', 0)}")
            logger.info(f"  Average conflict ratio: {summary.get('avg_conflict_ratio', 0):.4f}")

            most_conflicted = summary.get('most_conflicted_params', [])[:3]
            if most_conflicted:
                logger.info(f"  Top 3 most conflicted:")
                for param_name, conflict_ratio in most_conflicted:
                    logger.info(f"    - {param_name}: {conflict_ratio:.4f}")

    # Gradient trajectory summary
    if 'gradient_trajectory' in results and results['gradient_trajectory']:
        logger.info(f"\n[Gradient Trajectories]")
        for key, metrics in list(results['gradient_trajectory'].items())[:3]:  # Top 3
            avg_angle = metrics.get('avg_angle_change', 0)
            is_osc = metrics.get('is_oscillating', False)
            logger.info(f"  {key}: avg_angle={avg_angle:.2f}°, oscillating={is_osc}")

    logger.info(f"{'='*80}\n")
