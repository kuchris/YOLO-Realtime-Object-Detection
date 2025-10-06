"""
Configuration Manager Module - Settings and Configuration Handling

This module provides the ConfigManager class for loading and managing
application configuration from YAML files.

Author: [Your Name]
License: MIT
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import yaml


class ConfigManager:
    """Manages application configuration with YAML files.

    This class handles loading, validating, and providing access to configuration
    settings with proper error handling and default values.

    Attributes:
        config: Dictionary containing configuration values
        config_path: Path to configuration file
        logger: Logger instance for debugging

    Example:
        >>> config_manager = ConfigManager("config/config.yaml")
        >>> detection_config = config_manager.get_detection_config()
        >>> confidence = config_manager.get('detection.confidence', 0.5)
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize configuration manager.

        Args:
            config_path: Path to configuration file (optional, uses default if None)

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If configuration file is invalid
        """
        self.logger = logging.getLogger(__name__)

        if config_path is None:
            config_path = self._get_default_config_path()

        self.config_path = Path(config_path)
        self.config = self._load_config()

        self.logger.info(f"Configuration loaded from {self.config_path}")

    def _get_default_config_path(self) -> Path:
        """Get default configuration file path.

        Returns:
            Path to default configuration file
        """
        # Try multiple possible locations
        possible_paths = [
            Path("config/config.yaml"),
            Path("../config/config.yaml"),
            Path(__file__).parent.parent / "config" / "config.yaml"
        ]

        for path in possible_paths:
            if path.exists():
                return path

        # If no config file found, create default
        default_path = Path("config/config.yaml")
        self.logger.warning(f"No config file found, creating default at {default_path}")
        self._create_default_config(default_path)
        return default_path

    def _create_default_config(self, config_path: Path) -> None:
        """Create default configuration file.

        Args:
            config_path: Path where to create default config
        """
        default_config = {
            'detection': {
                'model': 'models/yolov8n.pt',
                'confidence': 0.6,
                'iou_threshold': 0.45,
                'input_size': 640,
                'device': 'cuda',
                'half_precision': True
            },
            'input': {
                'source': 'webcam',
                'webcam_id': 0,
                'youtube_url': '',
                'video_file': '',
                'resolution': [1280, 720],
                'fps': 30
            },
            'display': {
                'show_fps': True,
                'show_confidence': True,
                'box_thickness': 2,
                'font_scale': 0.6,
                'save_detections': False,
                'output_dir': 'outputs'
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/app.log',
                'max_size': '10MB',
                'backup_count': 5
            }
        }

        # Create directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Write default config
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)

        self.logger.info(f"Default configuration created at {config_path}")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            if not isinstance(config, dict):
                raise ValueError("Configuration file must contain a dictionary")

            return config

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {str(e)}") from e
        except Exception as e:
            raise ValueError(f"Error loading configuration: {str(e)}") from e

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.

        Args:
            key: Configuration key (supports dot notation like 'detection.confidence')
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            >>> confidence = config.get('detection.confidence', 0.5)
            >>> device = config.get('detection.device', 'cpu')
        """
        try:
            keys = key.split('.')
            value = self.config

            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default

            return value

        except Exception:
            return default

    def get_detection_config(self) -> Dict[str, Any]:
        """Get detection configuration section.

        Returns:
            Detection configuration dictionary
        """
        return self.get('detection', {})

    def get_input_config(self) -> Dict[str, Any]:
        """Get input configuration section.

        Returns:
            Input configuration dictionary
        """
        return self.get('input', {})

    def get_display_config(self) -> Dict[str, Any]:
        """Get display configuration section.

        Returns:
            Display configuration dictionary
        """
        return self.get('display', {})

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration section.

        Returns:
            Logging configuration dictionary
        """
        return self.get('logging', {})

    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set

        Example:
            >>> config.set('detection.confidence', 0.7)
            >>> config.set('input.source', 'youtube')
        """
        keys = key.split('.')
        config_section = self.config

        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config_section:
                config_section[k] = {}
            config_section = config_section[k]

        # Set the final value
        config_section[keys[-1]] = value

        self.logger.debug(f"Configuration updated: {key} = {value}")

    def save(self, config_path: Optional[str] = None) -> None:
        """Save current configuration to file.

        Args:
            config_path: Path to save configuration (uses current path if None)
        """
        if config_path is None:
            config_path = self.config_path
        else:
            config_path = Path(config_path)

        try:
            # Create directory if it doesn't exist
            config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)

            self.logger.info(f"Configuration saved to {config_path}")

        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")
            raise

    def validate(self) -> bool:
        """Validate configuration values.

        Returns:
            True if configuration is valid, False otherwise
        """
        valid = True

        # Validate detection configuration
        detection_config = self.get_detection_config()
        if not self._validate_detection_config(detection_config):
            valid = False

        # Validate input configuration
        input_config = self.get_input_config()
        if not self._validate_input_config(input_config):
            valid = False

        # Validate display configuration
        display_config = self.get_display_config()
        if not self._validate_display_config(display_config):
            valid = False

        return valid

    def _validate_detection_config(self, config: Dict[str, Any]) -> bool:
        """Validate detection configuration section.

        Args:
            config: Detection configuration dictionary

        Returns:
            True if valid, False otherwise
        """
        valid = True

        # Check confidence range
        confidence = config.get('confidence', 0.6)
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            self.logger.error("Invalid confidence value: must be between 0 and 1")
            valid = False

        # Check device
        device = config.get('device', 'cuda')
        if device not in ['cuda', 'cpu']:
            self.logger.error("Invalid device: must be 'cuda' or 'cpu'")
            valid = False

        # Check input size
        input_size = config.get('input_size', 640)
        if not isinstance(input_size, int) or input_size <= 0:
            self.logger.error("Invalid input_size: must be positive integer")
            valid = False

        return valid

    def _validate_input_config(self, config: Dict[str, Any]) -> bool:
        """Validate input configuration section.

        Args:
            config: Input configuration dictionary

        Returns:
            True if valid, False otherwise
        """
        valid = True

        # Check source type
        source = config.get('source', 'webcam')
        if source not in ['webcam', 'youtube', 'video_file']:
            self.logger.error("Invalid source: must be 'webcam', 'youtube', or 'video_file'")
            valid = False

        # Check webcam_id
        webcam_id = config.get('webcam_id', 0)
        if not isinstance(webcam_id, int) or webcam_id < 0:
            self.logger.error("Invalid webcam_id: must be non-negative integer")
            valid = False

        # Check resolution
        resolution = config.get('resolution', [1280, 720])
        if not isinstance(resolution, list) or len(resolution) != 2:
            self.logger.error("Invalid resolution: must be list of two integers [width, height]")
            valid = False
        elif not all(isinstance(x, int) and x > 0 for x in resolution):
            self.logger.error("Invalid resolution values: must be positive integers")
            valid = False

        return valid

    def _validate_display_config(self, config: Dict[str, Any]) -> bool:
        """Validate display configuration section.

        Args:
            config: Display configuration dictionary

        Returns:
            True if valid, False otherwise
        """
        valid = True

        # Check boolean values
        for key in ['show_fps', 'show_confidence', 'save_detections']:
            value = config.get(key)
            if value is not None and not isinstance(value, bool):
                self.logger.error(f"Invalid {key}: must be boolean")
                valid = False

        # Check numeric values
        for key in ['box_thickness', 'font_scale']:
            value = config.get(key)
            if value is not None and not isinstance(value, (int, float)) or (isinstance(value, (int, float)) and value <= 0):
                self.logger.error(f"Invalid {key}: must be positive number")
                valid = False

        return valid

    def __str__(self) -> str:
        """String representation of configuration.

        Returns:
            Configuration summary
        """
        return f"ConfigManager(config_path={self.config_path}, sections={list(self.config.keys())})"


# Test function for development
def test_config_manager():
    """Test the configuration manager."""
    import sys

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    try:
        # Test configuration manager
        config_manager = ConfigManager()

        print("Configuration Manager Test:")
        print(f"Config path: {config_manager.config_path}")
        print(f"Valid: {config_manager.validate()}")

        # Test getting values
        print(f"\nDetection config: {config_manager.get_detection_config()}")
        print(f"Input config: {config_manager.get_input_config()}")
        print(f"Display config: {config_manager.get_display_config()}")

        # Test dot notation
        confidence = config_manager.get('detection.confidence')
        print(f"\nConfidence from dot notation: {confidence}")

        # Test setting values
        config_manager.set('test.value', 'hello world')
        print(f"Test value: {config_manager.get('test.value')}")

        print("\nConfiguration manager test completed successfully!")

    except Exception as e:
        print(f"Configuration manager test failed: {str(e)}")


if __name__ == "__main__":
    test_config_manager()