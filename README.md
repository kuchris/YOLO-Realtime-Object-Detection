# YOLO Real-time Object Detection

A comprehensive, production-ready real-time object detection system powered by YOLOv8, designed for high-performance visual recognition tasks with GPU acceleration. This project provides a complete end-to-end solution for detecting and tracking 80 different object classes from the COCO dataset in real-time video streams.

## Project Description

This system leverages the cutting-edge YOLOv8 (You Only Look Once version 8) neural network architecture to perform real-time object detection with exceptional speed and accuracy. Built with a modular, extensible architecture, it seamlessly handles multiple input sources including live webcam feeds, YouTube videos, and local video files, making it ideal for a wide range of applications from security monitoring to research demonstrations.

**Key Capabilities:**

- **Real-time Performance**: Achieves 30-60+ FPS on modern NVIDIA GPUs, enabling smooth, lag-free object detection even on high-resolution video streams
- **Multi-source Flexibility**: Seamlessly switch between webcam, YouTube videos, and local files through simple configuration changes
- **Professional Recording**: Built-in screen recording with visual indicators to capture detection demonstrations for presentations, research papers, or training materials
- **GPU Acceleration**: Full CUDA support with automatic mixed precision (FP16) for maximum throughput on NVIDIA GPUs
- **80 Object Classes**: Pre-trained on the COCO dataset to recognize common objects including people, vehicles, animals, furniture, and everyday items
- **Production Ready**: Comprehensive logging, error handling, and configuration management make it suitable for both research and deployment scenarios

**Use Cases:**

- Computer vision research and experimentation
- Security and surveillance system prototypes
- Educational demonstrations and tutorials
- Automated video analysis and content tagging
- Traffic monitoring and vehicle counting
- Retail analytics and customer behavior analysis
- Wildlife monitoring and species detection
- Quality control and manufacturing inspection

Whether you're a researcher exploring computer vision techniques, a developer building intelligent applications, or an educator creating demonstration materials, this system provides a robust foundation for real-time object detection tasks.

## Features

- 🚀 **Real-time Detection**: 30+ FPS with YOLOv8nano on GPU
- 📹 **Multiple Input Sources**: Webcam, YouTube videos, local files
- 🎥 **Recording Capability**: Record detection output with visual indicator
- 🎯 **High Accuracy**: Configurable confidence thresholds
- 💻 **GPU Acceleration**: CUDA support for NVIDIA GPUs
- 🎨 **Rich Visualization**: Bounding boxes, labels, confidence scores
- ⚙️ **Configuration-driven**: YAML-based settings
- 📊 **Performance Monitoring**: FPS counter and statistics
- 🛠️ **Modular Design**: Clean, maintainable codebase

## System Requirements

- **Python 3.8+** (Python 3.12 recommended)
- **NVIDIA GPU with CUDA 12.1+** (recommended: RTX series)
- **8GB+ RAM**
- **Webcam** (optional, for live detection)
- **Internet connection** (for YouTube video downloads)

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd yolo

# Create virtual environment (recommended)
python -m venv yolo-env

# Activate virtual environment
# On Windows:
yolo-env\Scripts\activate
# On Linux/Mac:
source yolo-env/bin/activate

# Install PyTorch with CUDA support (for GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install opencv-python numpy ultralytics PyYAML matplotlib yt-dlp

# Verify CUDA installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. Basic Usage

```bash
# Run with default settings (reads from config/config.yaml)
python main.py

# Run with custom configuration
python main.py config/custom_config.yaml
```

### 3. Configuration

Edit `config/config.yaml` to customize settings:

```yaml
detection:
  model: "models/yolov8n.pt"    # YOLOv8nano model
  confidence: 0.6               # Detection confidence (0.0-1.0)
  device: "cuda"                # Use GPU acceleration (cuda/cpu)
  half_precision: true          # Use FP16 for faster inference

input:
  source: "youtube"             # Input source (webcam/youtube/video_file)
  webcam_id: 0                  # Webcam device ID
  youtube_url: "https://www.youtube.com/watch?v=example"
  video_file: ""                # Local video file path

display:
  show_fps: true                # Show FPS counter
  show_confidence: true         # Show confidence scores
  box_thickness: 2              # Bounding box line thickness
  font_scale: 0.6               # Text font size
```

## Project Structure

```
yolo/
├── main.py                     # Main application entry point
├── detector/                   # Core detection modules
│   ├── yolo_detector.py        # YOLO detection engine
│   ├── input_manager.py        # Multi-source input handling
│   ├── visualizer.py           # Detection visualization
│   ├── config_manager.py       # Configuration management
│   └── utils.py                # Utility functions
├── config/                     # Configuration files
│   ├── config.yaml             # Main configuration
│   └── coco_classes.txt        # COCO class names (80 classes)
├── models/                     # Downloaded model weights
│   └── yolov8n.pt              # YOLOv8nano model (auto-downloaded)
├── videos/                     # Downloaded YouTube videos
│   └── youtube_*.mp4           # Cached video files
├── outputs/                    # Recorded detection videos
│   └── detection_*.mp4         # Timestamped recordings
├── logs/                       # Application logs
│   └── app.log                 # Runtime logs
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Usage Guide

### Keyboard Controls

- **Q / ESC**: Quit application
- **SPACE**: Pause/Resume video
- **C**: Start/Stop recording (saves to `outputs/` folder)
- **S**: Save current frame as image
- **R**: Reset video to beginning (non-webcam sources)
- **H**: Toggle help overlay
- **I**: Toggle info overlay (device/source information)

### Recording Detection Output

1. Run the application: `python main.py`
2. Press **C** to start recording
   - A red dot ● and "REC" indicator appears in top-right corner
   - Recording saves to `outputs/detection_YYYYMMDD_HHMMSS.mp4`
3. Press **C** again to stop recording
4. Recording automatically stops when you quit (press Q)

### Input Sources

#### Webcam (Default)
```yaml
input:
  source: "webcam"
  webcam_id: 0  # Change if you have multiple cameras
```

#### YouTube Video
```yaml
input:
  source: "youtube"
  youtube_url: "https://www.youtube.com/watch?v=example"
```
- Videos are downloaded to `videos/` folder
- Cached for reuse (no re-download if already exists)
- Supports yt-dlp for reliable downloading

#### Local Video File
```yaml
input:
  source: "video_file"
  video_file: "path/to/your/video.mp4"
```

### Detection Settings

- **Confidence Threshold**: `0.6` (default) - Higher = fewer false positives
- **IOU Threshold**: `0.45` (default) - Non-maximum suppression threshold
- **Input Size**: `640` (default) - Model input resolution (higher = slower but more accurate)
- **Half Precision**: `true` (default) - Use FP16 for 2x faster inference on compatible GPUs

## Performance

Expected performance on **RTX 4080 Laptop GPU**:
- **Webcam (720p)**: 45-60 FPS
- **YouTube (720p)**: 40-55 FPS
- **Video File (1080p)**: 35-50 FPS
- **Memory Usage**: ~2GB VRAM
- **Model**: YOLOv8nano (3.2M parameters)

## Troubleshooting

### CUDA Not Available

```bash
# Check if NVIDIA GPU is detected
nvidia-smi

# Install correct PyTorch version with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

**If CUDA still not available:**
- Update NVIDIA drivers
- Ensure GPU supports CUDA (GTX 900 series or newer)
- For CPU-only: Change `device: "cpu"` in config.yaml

### Webcam Issues

```bash
# Check available cameras
python -c "import cv2; print([i for i in range(5) if cv2.VideoCapture(i).isOpened()])"

# If camera not detected:
# - Check camera permissions in OS settings
# - Close other apps using the camera
# - Try different webcam_id in config.yaml
```

### YouTube Download Issues

The application uses **yt-dlp** (reliable alternative to pytube) for downloading YouTube videos.

**If download fails:**
- Check internet connection
- Verify video URL format (full URL or youtu.be short link)
- Update yt-dlp: `pip install --upgrade yt-dlp`
- Some videos may be region-restricted or private

**Slow downloads:**
- Download speed depends on your internet connection
- Videos are cached in `videos/` folder for reuse
- Consider downloading smaller quality (360p/480p) in config

### Recording Issues

**If recording doesn't work:**
- Ensure `outputs/` folder exists (created automatically)
- Check disk space for video files
- Recording uses mp4v codec (widely compatible)
- Press 'C' to toggle recording on/off

## Advanced Usage

### Custom Model Training

```python
from detector.yolo_detector import YOLODetector

# Load custom YOLOv8 model
detector = YOLODetector(
    model_path="models/custom_model.pt",
    device="cuda",
    confidence=0.7
)

# Run detection on frame
detections = detector.detect(frame)
```

### Batch Processing

```python
from detector.input_manager import InputManager
from detector.yolo_detector import YOLODetector

input_manager = InputManager()
detector = YOLODetector("models/yolov8n.pt")

input_manager.set_video_file("video.mp4")

for success, frame in input_manager.get_frames():
    if success:
        detections = detector.detect(frame)
        # Process detections...
```

### Using Different YOLO Models

```yaml
detection:
  # YOLOv8 model variants (speed vs accuracy tradeoff)
  model: "models/yolov8n.pt"   # Nano - Fastest (3.2M params)
  # model: "models/yolov8s.pt"  # Small - Balanced (11.2M params)
  # model: "models/yolov8m.pt"  # Medium - More accurate (25.9M params)
  # model: "models/yolov8l.pt"  # Large - High accuracy (43.7M params)
  # model: "models/yolov8x.pt"  # Extra large - Best accuracy (68.2M params)
```

Models are automatically downloaded on first use.

## Development

### Code Quality

- **PEP 8 compliant** formatting with type hints
- **Comprehensive logging** to `logs/app.log`
- **Error handling** with graceful degradation
- **Modular architecture** for easy extension
- **Configuration-driven** design

### Testing Modules Individually

```bash
# Test YOLO detector
python detector/yolo_detector.py

# Test input manager
python detector/input_manager.py

# Test configuration manager
python detector/config_manager.py

# Test visualizer
python detector/visualizer.py
```

## Detected Object Classes

The YOLOv8 model is trained on the **COCO dataset** and can detect **80 object classes**:

Person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush.

## File Outputs

- **Downloaded YouTube videos**: `videos/youtube_<video_id>.mp4`
- **Recorded detections**: `outputs/detection_<timestamp>.mp4`
- **Application logs**: `logs/app.log`
- **Model weights**: `models/yolov8*.pt` (auto-downloaded)

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Submit a pull request

## Support

For issues and questions:
1. Check the **Troubleshooting** section above
2. Review application logs in `logs/app.log`
3. Verify system requirements and dependencies
4. Check GPU/CUDA setup with `nvidia-smi` and PyTorch

## Acknowledgments

- **YOLOv8** by Ultralytics
- **PyTorch** for deep learning framework
- **OpenCV** for computer vision operations
- **yt-dlp** for reliable YouTube video downloading

---

**Happy detecting! 🚀**

*Built with ❤️ using YOLOv8 and Python*
