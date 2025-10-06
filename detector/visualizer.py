"""
Visualization Module - Drawing and Display Utilities

This module provides the Visualizer class for drawing detection results
and performance information on video frames.

Author: [Your Name]
License: MIT
"""

import logging
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np


class Visualizer:
    """Handles visualization of detection results and performance metrics.

    This class provides methods to draw bounding boxes, labels, confidence scores,
    and performance information on video frames with customizable styling.

    Attributes:
        class_names: List of COCO class names
        colors: List of BGR colors for different classes
        logger: Logger instance for debugging

    Example:
        >>> visualizer = Visualizer(class_names=coco_classes)
        >>> annotated_frame = visualizer.draw_detections(frame, detections)
        >>> visualizer.display_frame(annotated_frame)
    """

    def __init__(
        self,
        class_names: List[str],
        box_thickness: int = 2,
        font_scale: float = 0.6,
        show_confidence: bool = True,
        show_fps: bool = True
    ) -> None:
        """Initialize visualizer with styling parameters.

        Args:
            class_names: List of class names for labels
            box_thickness: Thickness of bounding box lines
            font_scale: Scale factor for text
            show_confidence: Whether to show confidence scores
            show_fps: Whether to show FPS counter
        """
        self.logger = logging.getLogger(__name__)
        self.class_names = class_names
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.show_confidence = show_confidence
        self.show_fps = show_fps

        # Generate colors for each class
        self.colors = self._generate_colors(len(class_names))

        # Performance tracking
        self.fps_history = []
        self.max_history_length = 30

    def _generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for each class.

        Args:
            num_classes: Number of classes to generate colors for

        Returns:
            List of BGR color tuples
        """
        colors = []
        for i in range(num_classes):
            # Generate colors using HSV color space for better distribution
            hue = (i * 180 // num_classes) % 180
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, color)))
        return colors

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Tuple[Tuple[int, int, int, int], float, int]],
        fps: Optional[float] = None
    ) -> np.ndarray:
        """Draw detection results on frame.

        Args:
            frame: Input video frame
            detections: List of ((x1, y1, x2, y2), confidence, class_id) tuples
            fps: Current FPS value

        Returns:
            Frame with drawn detections
        """
        annotated_frame = frame.copy()

        # Draw FPS counter if enabled and provided
        if self.show_fps and fps is not None:
            self._draw_fps(annotated_frame, fps)

        # Draw each detection
        for bbox, confidence, class_id in detections:
            if 0 <= class_id < len(self.class_names):
                color = self.colors[class_id]
                class_name = self.class_names[class_id]
                self._draw_single_detection(annotated_frame, bbox, confidence, class_id, class_name, color)

        # Draw statistics
        self._draw_statistics(annotated_frame, len(detections))

        return annotated_frame

    def _draw_single_detection(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        confidence: float,
        class_id: int,
        class_name: str,
        color: Tuple[int, int, int]
    ) -> None:
        """Draw a single detection bounding box and label.

        Args:
            frame: Frame to draw on
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            confidence: Detection confidence score
            class_id: Class ID
            class_name: Class name
            color: BGR color tuple
        """
        x1, y1, x2, y2 = bbox

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)

        # Prepare label text
        if self.show_confidence:
            label = f"{class_name}: {confidence:.2f}"
        else:
            label = class_name

        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1)[0]
        label_y = max(y1, label_size[1] + 10)

        # Draw filled rectangle for label background
        cv2.rectangle(
            frame,
            (x1, label_y - label_size[1] - 10),
            (x1 + label_size[0], label_y),
            color,
            -1
        )

        # Draw label text
        cv2.putText(
            frame,
            label,
            (x1, label_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    def _draw_fps(self, frame: np.ndarray, fps: float) -> None:
        """Draw FPS counter on frame.

        Args:
            frame: Frame to draw on
            fps: Current FPS value
        """
        fps_text = f"FPS: {fps:.1f}"
        text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1)[0]

        # Draw background for FPS text
        cv2.rectangle(
            frame,
            (10, 10),
            (text_size[0] + 20, text_size[1] + 20),
            (0, 0, 0),
            -1
        )

        # Draw FPS text
        cv2.putText(
            frame,
            fps_text,
            (15, text_size[1] + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    def _draw_statistics(self, frame: np.ndarray, detection_count: int) -> None:
        """Draw detection statistics on frame.

        Args:
            frame: Frame to draw on
            detection_count: Number of detections
        """
        # Calculate position for statistics (bottom left)
        height, width = frame.shape[:2]
        stats_text = f"Objects: {detection_count}"
        text_size = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1)[0]

        # Draw background for statistics
        y_pos = height - 10
        cv2.rectangle(
            frame,
            (10, y_pos - text_size[1] - 10),
            (text_size[0] + 20, y_pos),
            (0, 0, 0),
            -1
        )

        # Draw statistics text
        cv2.putText(
            frame,
            stats_text,
            (15, y_pos - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    def create_info_overlay(
        self,
        frame: np.ndarray,
        device_info: Dict[str, Any],
        source_info: Dict[str, Any]
    ) -> np.ndarray:
        """Create information overlay with device and source details.

        Args:
            frame: Input frame
            device_info: Device information dictionary
            source_info: Source information dictionary

        Returns:
            Frame with information overlay
        """
        # Prepare info text (shorter, more concise)
        info_lines = [
            f"Device: {device_info.get('device', 'Unknown')}",
            f"Source: {source_info.get('source_type', 'Unknown')}",
        ]

        if source_info.get('source_type') != 'webcam':
            resolution = f"{source_info.get('width', 0)}x{source_info.get('height', 0)}"
            info_lines.append(f"Res: {resolution}")

        if device_info.get('device') == 'cuda':
            gpu_name = device_info.get('gpu_name', 'Unknown')
            # Shorten GPU name for display
            if 'NVIDIA' in gpu_name:
                gpu_name = gpu_name.replace('NVIDIA ', '').replace(' Laptop GPU', '')
            info_lines.append(f"GPU: {gpu_name}")

        # Create semi-transparent overlay
        overlay = frame.copy()
        alpha = 0.7  # Transparency level

        # Calculate text area size
        max_text_width = max([len(line) for line in info_lines]) * 12
        text_height = 25
        background_height = len(info_lines) * text_height + 15

        # Position in top-right corner with margin
        margin = 15
        x_pos = frame.shape[1] - max_text_width - margin
        y_pos = margin

        # Draw semi-transparent background
        cv2.rectangle(
            overlay,
            (x_pos - 8, y_pos - 8),
            (x_pos + max_text_width + 8, y_pos + background_height),
            (40, 40, 40),  # Dark gray background
            -1
        )

        # Apply transparency
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Draw info text with better readability
        for i, line in enumerate(info_lines):
            text_y = y_pos + 20 + i * text_height
            cv2.putText(
                frame,
                line,
                (x_pos, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # White text
                1,
                cv2.LINE_AA
            )

        return frame

    def update_fps(self, new_fps: float) -> float:
        """Update FPS history and calculate smoothed FPS.

        Args:
            new_fps: New FPS measurement

        Returns:
            Smoothed FPS value
        """
        self.fps_history.append(new_fps)
        if len(self.fps_history) > self.max_history_length:
            self.fps_history.pop(0)

        return sum(self.fps_history) / len(self.fps_history)

    def create_help_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Create help overlay with keyboard shortcuts.

        Args:
            frame: Input frame

        Returns:
            Frame with help overlay
        """
        help_text = [
            "Keyboard Shortcuts:",
            "Q - Quit",
            "S - Save frame",
            "SPACE - Pause/Resume",
            "R - Reset video",
            "H - Toggle help",
            "I - Toggle info overlay"
        ]

        # Create semi-transparent overlay
        overlay = frame.copy()
        alpha = 0.8

        # Calculate text area
        max_text_width = max([len(line) for line in help_text]) * 15
        text_height = 25
        total_height = len(help_text) * text_height + 20

        # Position in center-right
        x_pos = frame.shape[1] - max_text_width - 20
        y_pos = (frame.shape[0] - total_height) // 2

        # Draw background
        cv2.rectangle(
            overlay,
            (x_pos - 10, y_pos - 10),
            (x_pos + max_text_width + 10, y_pos + total_height),
            (0, 0, 0),
            -1
        )

        # Apply transparency
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, overlay)

        # Draw text
        for i, line in enumerate(help_text):
            cv2.putText(
                overlay,
                line,
                (x_pos, y_pos + 20 + i * text_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        return overlay

    def display_frame(self, frame: np.ndarray, window_name: str = "YOLO Detection") -> None:
        """Display frame in OpenCV window.

        Args:
            frame: Frame to display
            window_name: Window name
        """
        cv2.imshow(window_name, frame)

    def save_frame(self, frame: np.ndarray, filename: Optional[str] = None) -> str:
        """Save frame to file.

        Args:
            frame: Frame to save
            filename: Optional filename, auto-generated if None

        Returns:
            Path to saved file
        """
        import time
        from pathlib import Path

        if filename is None:
            timestamp = int(time.time())
            filename = f"detection_{timestamp}.jpg"

        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        filepath = output_dir / filename
        cv2.imwrite(str(filepath), frame)

        self.logger.info(f"Frame saved: {filepath}")
        return str(filepath)


# Test function for development
def test_visualizer():
    """Test the visualizer with sample detections."""
    import sys

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    try:
        # Create sample frame
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Create visualizer
        class_names = ['person', 'car', 'bicycle']
        visualizer = Visualizer(class_names)

        # Sample detections
        detections = [
            ((100, 100, 200, 300), 0.85, 0),  # person
            ((300, 200, 500, 400), 0.72, 1),  # car
            ((600, 300, 700, 500), 0.93, 2),  # bicycle
        ]

        # Draw detections
        annotated_frame = visualizer.draw_detections(frame, detections, fps=30.5)

        # Add info overlay
        device_info = {'device': 'cuda', 'gpu_name': 'RTX 4080'}
        source_info = {'source_type': 'webcam', 'fps': 30}
        annotated_frame = visualizer.create_info_overlay(annotated_frame, device_info, source_info)

        print("Visualizer test completed successfully!")
        print(f"Frame shape: {annotated_frame.shape}")
        print(f"Detections drawn: {len(detections)}")

    except Exception as e:
        print(f"Visualizer test failed: {str(e)}")


if __name__ == "__main__":
    test_visualizer()