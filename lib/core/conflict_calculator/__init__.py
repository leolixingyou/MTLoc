"""
Conflict Calculator Module for Multi-Task Learning Analysis

This module provides conflict analysis for YOLOPX model training.
Includes both standard conflict metrics and thorough debugging capabilities.
"""

# Model-specific calculator (YOLOPX only)
from .yolopx_conflict import YOLOPXConflictCalculator

# Thorough debugging modules
from .detailed_analyzer import DetailedConflictAnalyzer
from .debug_visualizer import DebugVisualizer

__all__ = [
    'YOLOPXConflictCalculator',
    'DetailedConflictAnalyzer',
    'DebugVisualizer'
]
