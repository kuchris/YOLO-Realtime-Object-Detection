# YOLO Real-time Object Detection

A comprehensive real-time object detection system using YOLOv8 with GPU acceleration. Supports multiple input sources including webcam, YouTube videos, and local video files.

## Features

- 🚀 **Real-time Detection**: 30+ FPS with YOLOv8nano on GPU
- 📹 **Multiple Input Sources**: Webcam, YouTube videos, local files
- 🎯 **High Accuracy**: Configurable confidence thresholds
- 💻 **GPU Acceleration**: CUDA support for NVIDIA GPUs
- 🎨 **Rich Visualization**: Bounding boxes, labels, confidence scores
- ⚙️ **Configuration-driven**: YAML-based settings
- 📊 **Performance Monitoring**: FPS counter and statistics
- 🛠️ **Modular Design**: Clean, maintainable codebase

## System Requirements

- **Python 3.8+**
- **NVIDIA GPU with CUDA 11.8+** (recommended: RTX series)
- **8GB+ RAM**
- **Webcam** (for live detection)

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd yolo

# Install dependencies
pip install -r requirements.txt

# Verify CUDA installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. Basic Usage

```bash
# Run with default webcam
python main.py

# Run with custom configuration
python main.py config/custom_config.yaml
```

### 3. Configuration

Edit `config/config.yaml` to customize settings:

```yaml
detection:
  model: "models/yolov8n.pt"    # YOLOv8nano model
  confidence: 0.6               # Detection confidence
  device: "cuda"                # Use GPU acceleration

input:
  source: "webcam"              # Input source
  webcam_id: 0                  # Webcam device ID
  # youtube_url: "https://youtube.com/watch?v=example"
  # video_file: "path/to/video.mp4"

display:
  show_fps: true                # Show FPS counter
  show_confidence: true         # Show confidence scores
  save_detections: false        # Save detection results
```

## Project Structure

```
yolo/
├── main.py                     # Main application
├── detector/                   # Core detection modules
│   ├── yolo_detector.py        # YOLO detection engine
│   ├── input_manager.py        # Multi-source input handling
│   ├── visualizer.py           # Detection visualization
│   ├── config_manager.py       # Configuration management
│   └── utils.py               # Utility functions
├── config/                     # Configuration files
│   ├── config.yaml            # Main configuration
│   └── coco_classes.txt       # COCO class names
├── models/                     # Downloaded model weights
│   └── yolov8n.pt             # YOLOv8nano model
├── logs/                       # Application logs
├── outputs/                    # Saved detection results
├── tests/                      # Test scripts
├── examples/                   # Example usage
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## Usage Guide

### Keyboard Controls

- **Q / ESC**: Quit application
- **SPACE**: Pause/Resume video
- **S**: Save current frame
- **R**: Reset video to beginning (non-webcam sources)
- **H**: Toggle help overlay

### Input Sources

#### Webcam (Default)
```yaml
input:
  source: "webcam"
  webcam_id: 0
```

#### YouTube Video
```yaml
input:
  source: "youtube"
  youtube_url: "https://www.youtube.com/watch?v=example"
```

#### Local Video File
```yaml
input:
  source: "video_file"
  video_file: "path/to/your/video.mp4"
```

### Detection Settings

- **Confidence Threshold**: 0.6 (default) - Adjust for accuracy vs. noise
- **IOU Threshold**: 0.45 (default) - Non-maximum suppression
- **Input Size**: 640 (default) - Model input resolution

## Testing

Run the comprehensive test suite:

```bash
python test_system.py
```

This will test:
- Static image detection
- Webcam detection
- YouTube video detection
- Configuration loading
- GPU acceleration

## Performance

Expected performance on RTX 4080:
- **Webcam (720p)**: 45-60 FPS
- **YouTube (720p)**: 40-55 FPS
- **Memory Usage**: ~2GB VRAM

## Troubleshooting

### CUDA Not Available
```bash
# Check CUDA installation
nvidia-smi

# Install correct PyTorch version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Webcam Issues
```bash
# Check available cameras
python -c "import cv2; print([i for i in range(5) if cv2.VideoCapture(i).isOpened()])"
```

### YouTube Download Issues
- Check internet connection
- Verify video URL format
- Some videos may be region-restricted

## Advanced Usage

### Custom Model Training
```python
from detector.yolo_detector import YOLODetector

# Load custom model
detector = YOLODetector(
    model_path="models/custom_model.pt",
    device="cuda",
    confidence=0.7
)
```

### Batch Processing
```python
from detector.input_manager import InputManager

input_manager = InputManager()
input_manager.set_video_file("video.mp4")

for success, frame in input_manager.get_frames():
    if success:
        detections = detector.detect(frame)
        # Process detections...
```

## Development

### Code Quality
- **PEP 8 compliant** formatting
- **Type hints** throughout
- **Comprehensive logging**
- **Error handling** with graceful degradation
- **Modular architecture** for easy extension

### Testing
```bash
# Test individual modules
python detector/yolo_detector.py
python detector/config_manager.py
python detector/input_manager.py
python detector/visualizer.py
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs in `logs/app.log`
3. Test with `python test_system.py`
4. Check system requirements

---

**Happy detecting! 🚀**