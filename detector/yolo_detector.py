"""
YOLO Real-time Object Detection - Core Detector Module

This module provides the main YOLODetector class for object detection
with GPU acceleration support and comprehensive error handling.

Author: [Your Name]
License: MIT
"""

import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class YOLODetector:
    """YOLO object detector with GPU acceleration support.

    This class provides a clean interface for YOLO object detection with support
    for different input sources and real-time processing.

    Attributes:
        model: Loaded YOLO model instance
        device: Processing device (cuda/cpu)
        class_names: List of COCO class names
        logger: Logger instance for debugging
        input_size: Model input resolution
        half_precision: Whether to use FP16 inference

    Example:
        >>> detector = YOLODetector(model_path="models/yolov8n.pt", device="cuda")
        >>> results = detector.detect(image_frame)
        >>> for box, score, class_id in results:
        ...     print(f"Detected {detector.class_names[class_id]}: {score:.2f}")
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        confidence: float = 0.6,
        input_size: int = 640,
        half_precision: bool = True
    ) -> None:
        """Initialize YOLO detector with specified model and device.

        Args:
            model_path: Path to YOLO model weights file
            device: Processing device ('cuda' or 'cpu')
            confidence: Default confidence threshold for detections
            input_size: Model input resolution
            half_precision: Use FP16 for faster inference

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If GPU unavailable when cuda specified
            ValueError: If invalid device specified
        """
        self.logger = logging.getLogger(__name__)
        self.model_path = Path(model_path)
        self.device = self._validate_device(device)
        self.confidence = confidence
        self.input_size = input_size
        self.half_precision = half_precision

        # Load model and classes
        self.model = self._load_model()
        self.class_names = self._load_class_names()

        self.logger.info(f"YOLO detector initialized: {self.model_path}")
        self.logger.info(f"Device: {self.device}, Half precision: {self.half_precision}")

    def _validate_device(self, device: str) -> str:
        """Validate and return appropriate device.

        Args:
            device: Desired device ('cuda' or 'cpu')

        Returns:
            Validated device string

        Raises:
            ValueError: If invalid device specified
            RuntimeError: If CUDA requested but unavailable
        """
        device = device.lower()
        if device not in ['cuda', 'cpu']:
            raise ValueError(f"Invalid device '{device}'. Use 'cuda' or 'cpu'.")

        if device == 'cuda' and not torch.cuda.is_available():
            self.logger.warning("CUDA requested but not available. Falling back to CPU.")
            return 'cpu'

        return device

    def _load_model(self) -> YOLO:
        """Load YOLO model from file.

        Returns:
            Loaded YOLO model

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            self.logger.info(f"Loading YOLO model from {self.model_path}")
            model = YOLO(str(self.model_path))

            # Move model to specified device
            model.to(self.device)

            # Enable half precision if requested and supported
            if self.half_precision and self.device == 'cuda':
                model.fuse()
                self.logger.info("Half precision (FP16) enabled")

            return model

        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}") from e

    def _load_class_names(self) -> List[str]:
        """Load COCO class names from file.

        Returns:
            List of class names

        Raises:
            FileNotFoundError: If class names file doesn't exist
        """
        try:
            classes_path = Path(__file__).parent.parent / "config" / "coco_classes.txt"

            if not classes_path.exists():
                # Fallback to default COCO classes if file doesn't exist
                self.logger.warning(f"Class names file not found: {classes_path}")
                return self._get_default_coco_classes()

            with open(classes_path, 'r', encoding='utf-8') as f:
                class_names = [line.strip() for line in f.readlines()]

            self.logger.info(f"Loaded {len(class_names)} class names")
            return class_names

        except Exception as e:
            self.logger.error(f"Failed to load class names: {str(e)}")
            return self._get_default_coco_classes()

    def _get_default_coco_classes(self) -> List[str]:
        """Get default COCO class names.

        Returns:
            Default COCO class names list
        """
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
            'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
        ]

    def detect(
        self,
        image: np.ndarray,
        confidence: Optional[float] = None
    ) -> List[Tuple[Tuple[int, int, int, int], float, int]]:
        """Detect objects in the input image.

        Args:
            image: Input image as numpy array (BGR format)
            confidence: Override default confidence threshold

        Returns:
            List of detection results as ((x1, y1, x2, y2), confidence, class_id) tuples

        Raises:
            ValueError: If input image is invalid
            RuntimeError: If detection fails
        """
        if confidence is None:
            confidence = self.confidence

        try:
            # Validate input
            if image is None or image.size == 0:
                raise ValueError("Invalid input image")

            start_time = time.perf_counter()

            # Run inference
            results = self.model(
                image,
                conf=confidence,
                imgsz=self.input_size,
                device=self.device,
                half=self.half_precision,
                verbose=False
            )

            # Process results
            detections = self._process_results(results[0])

            inference_time = time.perf_counter() - start_time
            self.logger.debug(f"Detection completed in {inference_time:.3f}s, found {len(detections)} objects")

            return detections

        except Exception as e:
            self.logger.error(f"Detection failed: {str(e)}")
            raise RuntimeError(f"Detection failed: {str(e)}") from e

    def _process_results(self, result) -> List[Tuple[Tuple[int, int, int, int], float, int]]:
        """Process YOLO results into standard format.

        Args:
            result: YOLO result object

        Returns:
            List of detections in ((x1, y1, x2, y2), confidence, class_id) format
        """
        detections = []

        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box
                detections.append(((x1, y1, x2, y2), float(score), int(cls)))

        return detections

    def get_device_info(self) -> Dict[str, Any]:
        """Get device information.

        Returns:
            Dictionary containing device information
        """
        info = {
            'device': self.device,
            'model_path': str(self.model_path),
            'input_size': self.input_size,
            'half_precision': self.half_precision,
            'class_count': len(self.class_names)
        }

        if self.device == 'cuda' and torch.cuda.is_available():
            info.update({
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory': torch.cuda.get_device_properties(0).total_memory / 1e9,  # GB
                'cuda_version': torch.version.cuda
            })

        return info

    def __del__(self):
        """Cleanup when detector is destroyed."""
        try:
            if hasattr(self, 'model'):
                # Clear GPU memory if using CUDA
                if self.device == 'cuda' and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.logger.debug("YOLO detector cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")


# Test function for development
def test_detector():
    """Test the YOLO detector with a sample image."""
    import os

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create a test image
    test_image = np.zeros((640, 640, 3), dtype=np.uint8)

    try:
        # Initialize detector
        detector = YOLODetector(
            model_path="models/yolov8n.pt",
            device="cuda",
            confidence=0.5
        )

        # Test detection
        results = detector.detect(test_image)
        print(f"Test completed. Found {len(results)} objects")

        # Print device info
        device_info = detector.get_device_info()
        print("Device Info:", device_info)

    except Exception as e:
        print(f"Test failed: {str(e)}")


if __name__ == "__main__":
    test_detector()