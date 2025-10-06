"""
Utility Functions Module

This module provides helper functions for common operations throughout
the YOLO detection application.

Author: [Your Name]
License: MIT
"""

import logging
import time
from typing import Tuple, List, Optional

import cv2
import numpy as np


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path

    Returns:
        Configured logger instance
    """
    import sys
    from pathlib import Path

    # Create logs directory if log file specified
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    level = getattr(logging, log_level.upper(), logging.INFO)

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

    return logging.getLogger(__name__)


def check_camera_availability(camera_id: int = 0) -> Tuple[bool, dict]:
    """Check if a camera is available and get its properties.

    Args:
        camera_id: Camera device ID to check

    Returns:
        Tuple of (is_available, camera_info)
    """
    cap = None
    try:
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            return False, {}

        # Get camera properties
        info = {
            'camera_id': camera_id,
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'backend': cap.getBackendName()
        }

        # Try to get supported resolutions
        supported_resolutions = []
        test_resolutions = [
            (1920, 1080), (1280, 720), (640, 480), (320, 240)
        ]

        for width, height in test_resolutions:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if actual_width == width and actual_height == height:
                supported_resolutions.append((width, height))

        info['supported_resolutions'] = supported_resolutions
        return True, info

    except Exception as e:
        logging.getLogger(__name__).error(f"Error checking camera {camera_id}: {str(e)}")
        return False, {}
    finally:
        if cap is not None:
            cap.release()


def find_available_cameras(max_cameras: int = 5) -> List[dict]:
    """Find all available cameras on the system.

    Args:
        max_cameras: Maximum number of cameras to check

    Returns:
        List of available camera information dictionaries
    """
    available_cameras = []
    logger = logging.getLogger(__name__)

    for camera_id in range(max_cameras):
        is_available, info = check_camera_availability(camera_id)
        if is_available:
            available_cameras.append(info)
            logger.info(f"Found camera {camera_id}: {info}")

    return available_cameras


def format_time_duration(seconds: float) -> str:
    """Format time duration in human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.0f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        return f"{hours}h {remaining_minutes}m"


def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1: First bounding box (x1, y1, x2, y2)
        box2: Second bounding box (x1, y1, x2, y2)

    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def resize_with_aspect_ratio(
    image: np.ndarray,
    target_size: Tuple[int, int],
    pad_color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize image while maintaining aspect ratio.

    Args:
        image: Input image
        target_size: Target size (width, height)
        pad_color: Padding color (BGR)

    Returns:
        Tuple of (resized_image, scale_factor, padding)
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Calculate scale factor
    scale = min(target_w / w, target_h / h)

    # Calculate new size
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Calculate padding
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Add padding
    padded = cv2.copyMakeBorder(
        resized,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=pad_color
    )

    return padded, scale, (pad_left, pad_top)


def create_fps_tracker(window_size: int = 30) -> dict:
    """Create FPS tracker for performance monitoring.

    Args:
        window_size: Number of frames to average over

    Returns:
        FPS tracker dictionary
    """
    return {
        'frame_times': [],
        'window_size': window_size,
        'last_time': time.perf_counter()
    }


def update_fps_tracker(tracker: dict) -> float:
    """Update FPS tracker and return current FPS.

    Args:
        tracker: FPS tracker dictionary

    Returns:
        Current FPS
    """
    current_time = time.perf_counter()

    if tracker['last_time'] > 0:
        frame_time = current_time - tracker['last_time']
        tracker['frame_times'].append(frame_time)

        # Keep only recent frame times
        if len(tracker['frame_times']) > tracker['window_size']:
            tracker['frame_times'].pop(0)

    tracker['last_time'] = current_time

    if len(tracker['frame_times']) > 0:
        avg_frame_time = sum(tracker['frame_times']) / len(tracker['frame_times'])
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

    return 0.0


def validate_youtube_url(url: str) -> bool:
    """Validate YouTube URL format.

    Args:
        url: YouTube URL to validate

    Returns:
        True if URL appears to be valid YouTube URL
    """
    import re

    youtube_patterns = [
        r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
        r'https?://(?:www\.)?youtube\.com/embed/[\w-]+',
        r'https?://youtu\.be/[\w-]+',
        r'https?://(?:www\.)?youtube\.com/shorts/[\w-]+'
    ]

    return any(re.match(pattern, url) for pattern in youtube_patterns)


def get_system_info() -> dict:
    """Get system information for debugging.

    Returns:
        Dictionary containing system information
    """
    import platform
    import torch

    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'opencv_version': cv2.__version__,
        'numpy_version': np.__version__,
    }

    if torch.cuda.is_available():
        info.update({
            'cuda_available': True,
            'cuda_version': torch.version.cuda,
            'gpu_count': torch.cuda.device_count(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None,
            'gpu_memory': torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.device_count() > 0 else None
        })
    else:
        info['cuda_available'] = False

    return info


# Test functions
def test_utils():
    """Test utility functions."""
    import sys

    # Setup logging
    logger = setup_logging("INFO")

    print("Testing utility functions...")

    # Test camera detection
    print("\n1. Testing camera detection:")
    cameras = find_available_cameras(3)
    print(f"Found {len(cameras)} cameras")

    # Test time formatting
    print("\n2. Testing time formatting:")
    print(f"45 seconds: {format_time_duration(45)}")
    print(f"125 seconds: {format_time_duration(125)}")
    print(f"3665 seconds: {format_time_duration(3665)}")

    # Test IoU calculation
    print("\n3. Testing IoU calculation:")
    box1 = (100, 100, 200, 200)
    box2 = (150, 150, 250, 250)
    iou = calculate_iou(box1, box2)
    print(f"IoU between boxes: {iou:.3f}")

    # Test FPS tracker
    print("\n4. Testing FPS tracker:")
    fps_tracker = create_fps_tracker()
    for i in range(5):
        time.sleep(0.1)  # Simulate processing time
        fps = update_fps_tracker(fps_tracker)
        print(f"Frame {i+1}: {fps:.1f} FPS")

    # Test YouTube URL validation
    print("\n5. Testing YouTube URL validation:")
    valid_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ"
    ]
    for url in valid_urls:
        is_valid = validate_youtube_url(url)
        print(f"URL {url}: {'Valid' if is_valid else 'Invalid'}")

    # Test system info
    print("\n6. System information:")
    sys_info = get_system_info()
    for key, value in sys_info.items():
        print(f"{key}: {value}")

    print("\nUtility functions test completed!")


if __name__ == "__main__":
    test_utils()