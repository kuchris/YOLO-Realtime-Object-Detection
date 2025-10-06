"""
System Test Script

This script demonstrates the complete YOLO detection system with different input sources.
Run this to verify everything is working correctly.

Author: [Your Name]
License: MIT
"""

import cv2
import numpy as np
from detector.config_manager import ConfigManager
from detector.input_manager import InputManager
from detector.visualizer import Visualizer
from detector.yolo_detector import YOLODetector
from detector.utils import setup_logging, get_system_info


def test_webcam_detection():
    """Test YOLO detection with webcam input."""
    print("=" * 60)
    print("Testing Webcam Detection")
    print("=" * 60)

    try:
        # Initialize components
        config_manager = ConfigManager()
        detector = YOLODetector(
            model_path="models/yolov8n.pt",
            device="cuda",
            confidence=0.6
        )
        input_manager = InputManager()
        visualizer = Visualizer(
            class_names=detector.class_names,
            show_fps=True,
            show_confidence=True
        )

        # Setup webcam
        if not input_manager.set_webcam_source(0):
            print("Failed to open webcam!")
            return False

        print("Webcam opened successfully!")
        print("Press 'q' to quit, 's' to save frame")
        print("Processing 30 frames for testing...")

        # Process some frames
        frame_count = 0
        total_detections = 0

        while frame_count < 30:
            success, frame = input_manager.get_frame()
            if not success:
                print("Failed to read frame!")
                break

            # Run detection
            detections = detector.detect(frame)

            # Visualize
            annotated_frame = visualizer.draw_detections(frame, detections, fps=30.0)

            # Display
            cv2.imshow("YOLO Detection Test", annotated_frame)

            total_detections += len(detections)
            frame_count += 1

            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()
        print(f"Processed {frame_count} frames")
        print(f"Total detections: {total_detections}")
        print(f"Average detections per frame: {total_detections/frame_count:.1f}")

        return True

    except Exception as e:
        print(f"Webcam test failed: {str(e)}")
        return False


def test_youtube_detection():
    """Test YOLO detection with YouTube video input."""
    print("\n" + "=" * 60)
    print("Testing YouTube Detection")
    print("=" * 60)

    # Example YouTube URL (short tech video for testing)
    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Short video

    try:
        # Initialize components
        config_manager = ConfigManager()
        detector = YOLODetector(
            model_path="models/yolov8n.pt",
            device="cuda",
            confidence=0.6
        )
        input_manager = InputManager()
        visualizer = Visualizer(
            class_names=detector.class_names,
            show_fps=True,
            show_confidence=True
        )

        # Setup YouTube video
        print(f"Loading YouTube video: {youtube_url}")
        if not input_manager.set_youtube_source(youtube_url, quality="720p"):
            print("Failed to load YouTube video!")
            print("This might be due to network issues or video restrictions.")
            return False

        print("YouTube video loaded successfully!")
        print("Press 'q' to quit")
        print("Processing 50 frames for testing...")

        # Process some frames
        frame_count = 0
        total_detections = 0

        while frame_count < 50:
            success, frame = input_manager.get_frame()
            if not success:
                print("End of video or failed to read frame!")
                break

            # Run detection
            detections = detector.detect(frame)

            # Visualize
            annotated_frame = visualizer.draw_detections(frame, detections, fps=30.0)

            # Display
            cv2.imshow("YOLO YouTube Detection Test", annotated_frame)

            total_detections += len(detections)
            frame_count += 1

            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()
        print(f"Processed {frame_count} frames")
        print(f"Total detections: {total_detections}")
        print(f"Average detections per frame: {total_detections/frame_count:.1f}")

        return True

    except Exception as e:
        print(f"YouTube test failed: {str(e)}")
        return False


def test_static_image():
    """Test YOLO detection with a static test image."""
    print("\n" + "=" * 60)
    print("Testing Static Image Detection")
    print("=" * 60)

    try:
        # Create a test image with some simple objects
        # This creates a blank image - in real usage you'd load an actual image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Add some simple shapes to simulate objects
        cv2.rectangle(test_image, (100, 100), (200, 300), (0, 255, 0), -1)  # Green rectangle
        cv2.circle(test_image, (400, 200), 50, (0, 0, 255), -1)  # Red circle

        # Initialize detector
        detector = YOLODetector(
            model_path="models/yolov8n.pt",
            device="cuda",
            confidence=0.6
        )

        print("Running detection on test image...")
        detections = detector.detect(test_image)

        print(f"Found {len(detections)} objects in test image")

        # Visualizer
        visualizer = Visualizer(
            class_names=detector.class_names,
            show_fps=True,
            show_confidence=True
        )

        # Draw detections
        annotated_image = visualizer.draw_detections(test_image, detections, fps=30.0)

        # Save result
        output_path = visualizer.save_frame(annotated_image, "test_result.jpg")
        print(f"Result saved to: {output_path}")

        # Display briefly
        cv2.imshow("Static Image Test", annotated_image)
        cv2.waitKey(2000)  # Show for 2 seconds
        cv2.destroyAllWindows()

        return True

    except Exception as e:
        print(f"Static image test failed: {str(e)}")
        return False


def main():
    """Run all system tests."""
    print("YOLO Real-time Object Detection - System Test")
    print("=" * 60)

    # Setup logging
    setup_logging("INFO")

    # Show system information
    print("System Information:")
    sys_info = get_system_info()
    for key, value in sys_info.items():
        print(f"  {key}: {value}")

    print("\nRunning system tests...")

    # Test results
    tests = [
        ("Static Image Detection", test_static_image),
        ("Webcam Detection", test_webcam_detection),
        ("YouTube Detection", test_youtube_detection),
    ]

    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"Test failed with exception: {str(e)}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")

    total_tests = len(results)
    passed_tests = sum(results.values())

    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 All tests passed! System is ready for use.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")

    print("\nTo run the full application:")
    print("  python main.py")


if __name__ == "__main__":
    main()