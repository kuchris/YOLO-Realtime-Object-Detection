"""
YOLO Detection Package

This package provides modules for real-time object detection using YOLO
with support for multiple input sources and GPU acceleration.

Modules:
    yolo_detector: Core YOLO detection functionality
    input_manager: Multi-source input handling
    visualizer: Detection visualization and drawing
    config_manager: Configuration management
    utils: Utility functions

Author: [Your Name]
License: MIT
"""

__version__ = "1.0.0"
__author__ = "[Your Name]"

from .config_manager import ConfigManager
from .input_manager import InputManager
from .visualizer import Visualizer
from .yolo_detector import YOLODetector

__all__ = [
    'ConfigManager',
    'InputManager',
    'Visualizer',
    'YOLODetector'
]