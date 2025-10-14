# ⚡ Quick Start Guide - Docker Edition

Get YOLO running in **under 5 minutes** with Docker!

## 🎯 Three Methods to Run

### Method 1: Using Makefile (Easiest)
```bash
# Clone and enter directory
git clone https://github.com/kuchris/YOLO-Realtime-Object-Detection
cd YOLO-Realtime-Object-Detection

# Check prerequisites
make check

# Run with GPU (recommended)
make run-gpu

# OR run with CPU only
make run-cpu
```

### Method 2: Using Helper Scripts
```bash
# Clone and enter directory
git clone https://github.com/kuchris/YOLO-Realtime-Object-Detection
cd YOLO-Realtime-Object-Detection

# Run with GPU
chmod +x docker-run-gpu.sh
./docker-run-gpu.sh

# OR run with CPU only
chmod +x docker-run-cpu.sh
./docker-run-cpu.sh
```

### Method 3: Using Docker Compose
```bash
# Clone and enter directory
git clone https://github.com/kuchris/YOLO-Realtime-Object-Detection
cd YOLO-Realtime-Object-Detection

# Run with GPU
docker-compose up yolo-detector-gpu

# OR run with CPU only
docker-compose --profile cpu up yolo-detector-cpu
```

---

## 🎮 Controls

Once the application starts:
- **Q** or **ESC** → Quit
- **SPACE** → Pause/Resume
- **C** → Start/Stop Recording
- **S** → Save Current Frame
- **H** → Toggle Help

---

## ⚙️ Change Input Source

Edit `config/config.yaml`:

### Use Webcam (Default)
```yaml
input:
  source: "webcam"
  webcam_id: 0
```

### Use YouTube Video
```yaml
input:
  source: "youtube"
  youtube_url: "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

### Use Local Video
```yaml
input:
  source: "video_file"
  video_file: "/app/videos/my-video.mp4"
```

Then restart the container!

---

## 📊 Performance

| Mode | Hardware | Expected FPS |
|------|----------|--------------|
| GPU | RTX 4080 | 45-60 FPS |
| GPU | RTX 3060 | 30-45 FPS |
| GPU | GTX 1660 | 20-30 FPS |
| CPU | Modern i7/i9 | 8-12 FPS |
| CPU | Older CPU | 4-8 FPS |

---

## 🐛 Quick Troubleshooting

### Display Issues
```bash
xhost +local:docker
```

### Can't Access Webcam
```bash
# Check available webcams
ls -l /dev/video*

# Give permissions
sudo chmod 666 /dev/video0
```

### GPU Not Working
```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Slow Performance
- Use GPU version if available
- Lower confidence threshold
- Use smaller model (yolov8n.pt)
- Reduce video resolution

---

## 📝 Useful Commands

```bash
# Stop container
make stop
# OR
docker stop yolo-detector-gpu

# View logs
make logs
# OR
cat logs/app.log

# Enter container
make shell-gpu
# OR
docker exec -it yolo-detector-gpu bash

# Clean up
make clean
# OR
docker rm -f yolo-detector-gpu

# Rebuild image
make rebuild-gpu
# OR
docker build --no-cache -t yolo-realtime-detection:gpu .
```

---

## 🎯 What You Get

After running, you'll have:
- ✅ Real-time object detection display window
- ✅ 80 object classes detected (COCO dataset)
- ✅ Bounding boxes with labels and confidence scores
- ✅ FPS counter showing performance
- ✅ Recording capability (press 'C')
- ✅ Outputs saved in `outputs/` directory
- ✅ Models cached in `models/` directory

---

## 🆘 Need Help?

1. Read [README-DOCKER.md](README-DOCKER.md) for detailed documentation
2. Check `logs/app.log` for error messages
3. Run `make check` to verify prerequisites
4. Open an issue on GitHub with your problem

---

## 🚀 Next Steps

1. ✅ Get it running (you're here!)
2. 🎨 Try different input sources (YouTube, video files)
3. 🎯 Adjust confidence threshold for accuracy
4. 📹 Record some detection videos
5. 🔧 Experiment with different YOLO models
6. 🌟 Star the repository if you find it useful!

---

**That's it! You're ready to detect objects in real-time! 🎉**
