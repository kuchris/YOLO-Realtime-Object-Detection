"""
YOLO Real-time Object Detection - Main Application

This is the main entry point for the YOLO real-time object detection system.
It integrates the detector, input manager, visualizer, and configuration manager
to provide a complete real-time detection solution.

Author: [Your Name]
License: MIT
"""

import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Set OpenCV to headless mode if HEADLESS environment variable is set
if os.getenv('HEADLESS', 'false').lower() == 'true':
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    cv2.setNumThreads(0)  # Disable threading for headless mode

from detector.config_manager import ConfigManager
from detector.input_manager import InputManager
from detector.visualizer import Visualizer
from detector.yolo_detector import YOLODetector


class YOLODetectionApp:
    """Main application class for YOLO real-time object detection.

    This class orchestrates the entire detection process, handling user input,
    managing different input sources, and coordinating between all components.

    Attributes:
        config_manager: Configuration manager instance
        detector: YOLO detector instance
        input_manager: Input manager instance
        visualizer: Visualizer instance
        running: Application running state
        paused: Pause state for video playback
        show_help: Help overlay visibility

    Example:
        >>> app = YOLODetectionApp()
        >>> app.run()
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize the YOLO detection application.

        Args:
            config_path: Path to configuration file (optional)

        Raises:
            RuntimeError: If application initialization fails
        """
        self.logger = self._setup_logging()
        self.running = False
        self.paused = False
        self.show_help = False
        self.show_info = True
        self.frame_count = 0
        self.start_time = time.perf_counter()
        self.recording = False
        self.video_writer = None
        self.output_path = None

        # Check if running in headless mode (no display)
        self.headless = os.getenv('HEADLESS', 'false').lower() == 'true'
        if self.headless:
            self.logger.info("Running in HEADLESS mode - output will be saved automatically")

        try:
            # Initialize components
            self.config_manager = ConfigManager(config_path)
            self._validate_configuration()

            self.detector = self._initialize_detector()
            self.input_manager = InputManager()
            self.visualizer = self._initialize_visualizer()

            # Setup input source
            self._setup_input_source()

            # Setup signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)

            self.logger.info("YOLO Detection Application initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize application: {str(e)}")
            raise RuntimeError(f"Application initialization failed: {str(e)}") from e

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration.

        Returns:
            Configured logger instance
        """
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/app.log', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )

        return logging.getLogger(__name__)

    def _validate_configuration(self) -> None:
        """Validate application configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.config_manager.validate():
            raise ValueError("Invalid configuration detected")

        self.logger.info("Configuration validated successfully")

    def _initialize_detector(self) -> YOLODetector:
        """Initialize YOLO detector with configuration.

        Returns:
            Initialized YOLO detector

        Raises:
            RuntimeError: If detector initialization fails
        """
        detection_config = self.config_manager.get_detection_config()

        try:
            detector = YOLODetector(
                model_path=detection_config['model'],
                device=detection_config['device'],
                confidence=detection_config['confidence'],
                input_size=detection_config['input_size'],
                half_precision=detection_config['half_precision']
            )

            self.logger.info(f"YOLO detector initialized: {detector.get_device_info()}")
            return detector

        except Exception as e:
            self.logger.error(f"Failed to initialize detector: {str(e)}")
            raise RuntimeError(f"Detector initialization failed: {str(e)}") from e

    def _initialize_visualizer(self) -> Visualizer:
        """Initialize visualizer with configuration.

        Returns:
            Initialized visualizer
        """
        display_config = self.config_manager.get_display_config()

        visualizer = Visualizer(
            class_names=self.detector.class_names,
            box_thickness=display_config['box_thickness'],
            font_scale=display_config['font_scale'],
            show_confidence=display_config['show_confidence'],
            show_fps=display_config['show_fps']
        )

        self.logger.info("Visualizer initialized")
        return visualizer

    def _setup_input_source(self) -> None:
        """Setup input source based on configuration."""
        input_config = self.config_manager.get_input_config()
        source_type = input_config['source']

        success = False

        if source_type == 'webcam':
            success = self.input_manager.set_webcam_source(input_config['webcam_id'])
        elif source_type == 'youtube':
            youtube_url = input_config.get('youtube_url', '')
            if youtube_url:
                success = self.input_manager.set_youtube_source(youtube_url)
            else:
                self.logger.warning("YouTube source specified but no URL provided")
        elif source_type == 'video_file':
            video_file = input_config.get('video_file', '')
            if video_file:
                success = self.input_manager.set_video_file_source(video_file)
            else:
                self.logger.warning("Video file source specified but no path provided")

        if not success:
            self.logger.warning(f"Failed to setup {source_type} source, falling back to webcam")
            success = self.input_manager.set_webcam_source(0)

        if success:
            source_info = self.input_manager.get_source_info()
            self.logger.info(f"Input source setup: {source_info}")
        else:
            raise RuntimeError("Failed to setup any input source")

    def run(self) -> None:
        """Run the main application loop."""
        self.running = True
        self.logger.info("Starting YOLO Detection Application...")

        try:
            # In headless mode, start recording automatically
            if self.headless:
                self._start_recording()

            # Main processing loop
            while self.running:
                if not self.paused:
                    success = self._process_frame()
                    if not success:
                        self.logger.info("No more frames available, video ended")
                        if self.headless:
                            # In headless mode, stop automatically when video ends
                            self.running = False
                        else:
                            print("\n⏹️  Video ended. Press 'Q' to quit or 'R' to restart...")
                            # Pause at the end instead of quitting
                            self.paused = True
                else:
                    # When paused, just wait for key events
                    time.sleep(0.1)
                    key = cv2.waitKey(1) & 0xFF
                    self._handle_key_press(key)

        except KeyboardInterrupt:
            self.logger.info("Application interrupted by user")
        except Exception as e:
            self.logger.error(f"Application error: {str(e)}")
        finally:
            self._cleanup()

    def _process_frame(self) -> bool:
        """Process a single frame from the input source.

        Returns:
            True if frame processed successfully, False if no more frames
        """
        # Get frame from input source
        success, frame = self.input_manager.get_frame()
        if not success or frame is None:
            return False

        # Run detection
        detection_config = self.config_manager.get_detection_config()
        detections = self.detector.detect(frame, confidence=detection_config['confidence'])

        # Calculate FPS
        current_time = time.perf_counter()
        elapsed_time = current_time - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        smoothed_fps = self.visualizer.update_fps(fps)

        # Visualize results
        annotated_frame = self.visualizer.draw_detections(frame, detections, smoothed_fps)

        # Add info overlay if enabled
        if self.show_info:
            device_info = self.detector.get_device_info()
            source_info = self.input_manager.get_source_info()
            annotated_frame = self.visualizer.create_info_overlay(
                annotated_frame, device_info, source_info
            )

        # Add help overlay if enabled
        if self.show_help:
            annotated_frame = self.visualizer.create_help_overlay(annotated_frame)

        # Add recording indicator if recording
        if self.recording:
            self._draw_recording_indicator(annotated_frame)

        # Record frame if recording is enabled
        if self.recording and self.video_writer is not None:
            self.video_writer.write(annotated_frame)

        # Store frame for saving later
        self.last_frame = annotated_frame

        # Display frame (skip in headless mode)
        if not self.headless:
            self.visualizer.display_frame(annotated_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            self._handle_key_press(key)
        else:
            # In headless mode, just log progress periodically
            if self.frame_count % 30 == 0:
                self.logger.info(f"Processed {self.frame_count} frames, FPS: {smoothed_fps:.1f}")

        self.frame_count += 1
        return True

    def _handle_key_press(self, key: int) -> None:
        """Handle keyboard input.

        Args:
            key: Key code from cv2.waitKey
        """
        if key == ord('q') or key == 27:  # 'q' or ESC
            self.logger.info("Quit requested by user")
            self.running = False
        elif key == ord(' '):  # Spacebar
            self.paused = not self.paused
            self.logger.info(f"Application {'paused' if self.paused else 'resumed'}")
        elif key == ord('s'):  # Save frame
            if hasattr(self, 'last_frame') and self.last_frame is not None:
                output_path = self.visualizer.save_frame(self.last_frame)
                self.logger.info(f"Frame saved: {output_path}")
                print(f"\n📸 Frame saved: {output_path}")
        elif key == ord('r'):  # Reset video
            if self.input_manager.source_type != 'webcam':
                self.input_manager.reset_source()
                self.logger.info("Video reset to beginning")
        elif key == ord('h'):  # Toggle help
            self.show_help = not self.show_help
            self.logger.info(f"Help overlay {'enabled' if self.show_help else 'disabled'}")
        elif key == ord('i'):  # Toggle info overlay
            self.show_info = not self.show_info
            self.logger.info(f"Info overlay {'enabled' if self.show_info else 'disabled'}")
        elif key == ord('c'):  # Toggle recording
            if self.recording:
                self._stop_recording()
            else:
                self._start_recording()

    def _draw_recording_indicator(self, frame: np.ndarray) -> None:
        """Draw a recording indicator on the frame.

        Args:
            frame: Frame to draw on (modified in place)
        """
        height, width = frame.shape[:2]

        # Draw red circle (recording dot) in top-right corner
        cv2.circle(frame, (width - 80, 30), 12, (0, 0, 255), -1)

        # Draw "REC" text next to the dot
        cv2.putText(
            frame,
            "REC",
            (width - 60, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    def _start_recording(self) -> None:
        """Start recording the detection output."""
        try:
            # Create outputs directory
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)

            # Generate output filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            source_info = self.input_manager.get_source_info()
            self.output_path = output_dir / f"detection_{timestamp}.mp4"

            # Get video properties
            width = source_info['width']
            height = source_info['height']
            fps = source_info['fps']

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                str(self.output_path),
                fourcc,
                fps,
                (width, height)
            )

            if self.video_writer.isOpened():
                self.recording = True
                self.logger.info(f"Started recording to: {self.output_path}")
                print(f"\n🔴 Recording started: {self.output_path}")
            else:
                self.logger.error("Failed to open video writer")
                self.video_writer = None

        except Exception as e:
            self.logger.error(f"Error starting recording: {str(e)}")
            self.recording = False
            self.video_writer = None

    def _stop_recording(self) -> None:
        """Stop recording the detection output."""
        try:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
                self.recording = False
                self.logger.info(f"Stopped recording: {self.output_path}")
                print(f"\n⏹️  Recording saved: {self.output_path}")
        except Exception as e:
            self.logger.error(f"Error stopping recording: {str(e)}")

    def _signal_handler(self, signum, frame) -> None:
        """Handle system signals.

        Args:
            signum: Signal number
            frame: Current frame
        """
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def _cleanup(self) -> None:
        """Cleanup resources before exiting."""
        self.logger.info("Cleaning up resources...")

        try:
            # Stop recording if active
            if self.recording:
                self._stop_recording()

            if hasattr(self, 'input_manager'):
                self.input_manager._cleanup_current_source()
            if hasattr(self, 'detector') and self.detector is not None:
                del self.detector
                self.detector = None
            cv2.destroyAllWindows()
            self.logger.info("Application cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

    def get_statistics(self) -> dict:
        """Get application statistics.

        Returns:
            Dictionary containing application statistics
        """
        elapsed_time = time.perf_counter() - self.start_time
        avg_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0

        return {
            'runtime_seconds': elapsed_time,
            'frames_processed': self.frame_count,
            'average_fps': avg_fps,
            'current_fps': self.visualizer.fps_history[-1] if self.visualizer.fps_history else 0,
            'source_type': self.input_manager.source_type if hasattr(self, 'input_manager') else 'unknown',
            'device': self.detector.device if self.detector is not None else 'unknown'
        }


def main():
    """Main entry point for the application."""
    print("=" * 60)
    print("YOLO Real-time Object Detection")
    print("=" * 60)

    try:
        # Parse command line arguments
        config_path = None
        if len(sys.argv) > 1:
            config_path = sys.argv[1]

        # Create and run application
        app = YOLODetectionApp(config_path)
        app.run()

        # Print statistics
        stats = app.get_statistics()
        print("\n" + "=" * 60)
        print("Application Statistics:")
        print(f"Runtime: {stats['runtime_seconds']:.1f} seconds")
        print(f"Frames processed: {stats['frames_processed']}")
        print(f"Average FPS: {stats['average_fps']:.1f}")
        print(f"Source: {stats['source_type']}")
        print(f"Device: {stats['device']}")
        print("=" * 60)

    except Exception as e:
        print(f"Application failed: {str(e)}")
        logging.error(f"Application failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()