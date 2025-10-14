# 🐳 Docker Setup for YOLO Real-time Object Detection

Run YOLO object detection with **zero manual dependency installation**! Everything runs inside Docker containers.

## 📋 Table of Contents
- [Quick Start](#-quick-start)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Performance](#-performance)

---

## 🚀 Quick Start

### GPU Version (Recommended - Fast)
```bash
# Clone repository
git clone https://github.com/kuchris/YOLO-Realtime-Object-Detection
cd YOLO-Realtime-Object-Detection

# Make script executable
chmod +x docker-run-gpu.sh

# Run!
./docker-run-gpu.sh
```

### CPU Version (No GPU Required - Slower)
```bash
# Make script executable
chmod +x docker-run-cpu.sh

# Run!
./docker-run-cpu.sh
```

---

## 📦 Prerequisites

### For GPU Version:
1. **Docker** (version 20.10+)
2. **NVIDIA GPU** with CUDA support
3. **NVIDIA Docker Runtime**
4. **X11** (for display - usually pre-installed on Linux)

### For CPU Version:
1. **Docker** (version 20.10+)
2. **X11** (for display)

---

## 🔧 Installation

### Step 1: Install Docker

**Ubuntu/Debian:**
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group (optional, to run without sudo)
sudo usermod -aG docker $USER
newgrp docker
```

**Other Systems:**
- Windows/Mac: Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### Step 2: Install NVIDIA Docker Runtime (GPU Only)

**Ubuntu/Debian:**
```bash
# Add NVIDIA package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

**Verify GPU Access:**
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Step 3: Clone Repository
```bash
git clone https://github.com/kuchris/YOLO-Realtime-Object-Detection
cd YOLO-Realtime-Object-Detection
```

---

## 💻 Usage

### Method 1: Helper Scripts (Easiest)

**GPU Version:**
```bash
chmod +x docker-run-gpu.sh
./docker-run-gpu.sh
```

**CPU Version:**
```bash
chmod +x docker-run-cpu.sh
./docker-run-cpu.sh
```

### Method 2: Docker Compose

**GPU Version:**
```bash
docker-compose up yolo-detector-gpu
```

**CPU Version:**
```bash
docker-compose --profile cpu up yolo-detector-cpu
```

### Method 3: Manual Docker Run

**GPU Version:**
```bash
docker build -t yolo-realtime-detection:gpu .

docker run -it --rm \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $(pwd)/config:/app/config:rw \
    -v $(pwd)/models:/app/models:rw \
    -v $(pwd)/videos:/app/videos:rw \
    -v $(pwd)/outputs:/app/outputs:rw \
    -v $(pwd)/logs:/app/logs:rw \
    --device /dev/video0:/dev/video0 \
    --network host \
    yolo-realtime-detection:gpu
```

**CPU Version:**
```bash
docker build -t yolo-realtime-detection:cpu -f Dockerfile.cpu .

docker run -it --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $(pwd)/config:/app/config:rw \
    -v $(pwd)/models:/app/models:rw \
    -v $(pwd)/videos:/app/videos:rw \
    -v $(pwd)/outputs:/app/outputs:rw \
    -v $(pwd)/logs:/app/logs:rw \
    --device /dev/video0:/dev/video0 \
    --network host \
    yolo-realtime-detection:cpu
```

---

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

```yaml
detection:
  model: "models/yolov8n.pt"
  confidence: 0.6
  device: "cuda"  # Use "cpu" for CPU version
  half_precision: true

input:
  source: "webcam"  # Options: webcam, youtube, video_file
  webcam_id: 0
  youtube_url: ""
  video_file: ""
```

### Input Sources

**1. Webcam (Default)**
```yaml
input:
  source: "webcam"
  webcam_id: 0
```

**2. YouTube Video**
```yaml
input:
  source: "youtube"
  youtube_url: "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
```

**3. Local Video File**
```yaml
input:
  source: "video_file"
  video_file: "/app/videos/your_video.mp4"
```

---

## 🎮 Controls

Once running:
- **Q** or **ESC** - Quit application
- **SPACE** - Pause/Resume video
- **C** - Start/Stop recording
- **S** - Save current frame
- **R** - Reset video (non-webcam)
- **H** - Toggle help overlay
- **I** - Toggle info overlay

---

## 🗂️ Directory Structure

```
YOLO-Realtime-Object-Detection/
├── config/          # Configuration files (editable)
├── models/          # Downloaded model weights (persistent)
├── videos/          # Cached YouTube videos (persistent)
├── outputs/         # Recorded detection videos (persistent)
├── logs/            # Application logs (persistent)
├── Dockerfile       # GPU Docker build
├── Dockerfile.cpu   # CPU Docker build
├── docker-compose.yml
├── docker-run-gpu.sh
├── docker-run-cpu.sh
└── requirements.txt
```

All directories with persistent data are mounted as Docker volumes, so your data persists between container runs.

---

## 🐛 Troubleshooting

### Issue: "Cannot connect to X server"

**Solution:**
```bash
# Allow Docker to access X11
xhost +local:docker

# Run the application
./docker-run-gpu.sh

# After done, restore permissions
xhost -local:docker
```

### Issue: "Could not open video device"

**Solutions:**
1. Check if webcam is available:
   ```bash
   ls -l /dev/video*
   ```

2. Find correct device number:
   ```bash
   v4l2-ctl --list-devices
   ```

3. Update `docker-run-gpu.sh` with correct device:
   ```bash
   --device /dev/video1:/dev/video0  # If your webcam is video1
   ```

4. Check permissions:
   ```bash
   sudo chmod 666 /dev/video0
   ```

### Issue: "CUDA not available" (GPU version)

**Check NVIDIA Docker:**
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

If this fails:
1. Reinstall NVIDIA Docker Runtime
2. Ensure NVIDIA drivers are installed: `nvidia-smi`
3. Restart Docker: `sudo systemctl restart docker`

### Issue: Slow performance

**For GPU:**
- Check GPU usage: `nvidia-smi`
- Reduce resolution in config
- Use smaller model (yolov8n.pt)

**For CPU:**
- CPU mode is inherently slower (5-15 FPS)
- Close other applications
- Use lower resolution
- Consider using GPU version

### Issue: "Permission denied" errors

**Solution:**
```bash
# Make scripts executable
chmod +x docker-run-gpu.sh docker-run-cpu.sh

# Fix directory permissions
sudo chown -R $USER:$USER config models videos outputs logs
```

### Issue: Docker build fails

**Solutions:**
1. Check internet connection
2. Increase Docker memory limit (Docker Desktop settings)
3. Clean Docker cache:
   ```bash
   docker system prune -a
   ```
4. Try building with no cache:
   ```bash
   docker build --no-cache -t yolo-realtime-detection:gpu .
   ```

### Issue: YouTube download fails

**Solutions:**
1. Update yt-dlp in container:
   ```bash
   docker exec -it yolo-detector-gpu pip install --upgrade yt-dlp
   ```
2. Check video availability (region restrictions)
3. Try different video URL

---

## 📊 Performance Comparison

| Mode | Hardware | FPS | Accuracy | Use Case |
|------|----------|-----|----------|----------|
| GPU | RTX 4080 | 45-60 | High | Real-time applications |
| GPU | RTX 3060 | 30-45 | High | Real-time applications |
| GPU | GTX 1660 | 20-30 | High | Near real-time |
| CPU | i9-12900K | 8-12 | High | Testing/development |
| CPU | i5-10400 | 4-8 | High | Testing/development |

---

## 🔄 Updating

To update to the latest version:

```bash
# Pull latest code
git pull origin main

# Rebuild Docker images
docker-compose build --no-cache

# Or rebuild manually
docker build --no-cache -t yolo-realtime-detection:gpu .
docker build --no-cache -t yolo-realtime-detection:cpu -f Dockerfile.cpu .
```

---

## 🧹 Cleanup

**Remove containers:**
```bash
docker-compose down
```

**Remove images:**
```bash
docker rmi yolo-realtime-detection:gpu
docker rmi yolo-realtime-detection:cpu
```

**Remove all (including volumes):**
```bash
docker-compose down -v
rm -rf models/* videos/* outputs/* logs/*
```

---

## 💡 Tips

1. **First run takes longer** - Models are downloaded automatically
2. **Models are cached** - Subsequent runs are faster
3. **GPU memory** - YOLOv8n uses ~2GB VRAM
4. **Multiple webcams** - Change `webcam_id` in config
5. **Recording** - Press 'C' to start/stop, files saved in `outputs/`
6. **Custom models** - Place in `models/` directory and update config
7. **Logs** - Check `logs/app.log` for debugging

---

## 🔐 Security Notes

- X11 socket is mounted for display (required for GUI)
- Webcam device is passed through to container
- All volumes use read-write permissions
- Run container without `--privileged` flag (more secure)

---

## 🌐 Using with Remote Servers

If running on a remote server without display:

### Option 1: X11 Forwarding over SSH
```bash
# Connect with X11 forwarding
ssh -X user@remote-server

# Run the container
./docker-run-gpu.sh
```

### Option 2: VNC Server (Headless)
```bash
# Install VNC in container (modify Dockerfile to add):
RUN apt-get install -y x11vnc xvfb

# Start virtual display
Xvfb :99 -screen 0 1920x1080x24 &
export DISPLAY=:99
x11vnc -display :99 -forever &

# Connect with VNC client to view
```

### Option 3: Save Video Only (No Display)
Modify `main.py` to not show windows, only record to file.

---

## 🐍 Python Version Note

Docker images use **Python 3.11** which is compatible with all dependencies. If you need a different version, modify the Dockerfile:

```dockerfile
# Change this line:
python3.11 \

# To your desired version:
python3.10 \
```

---

## 📦 Model Variants

The default model is YOLOv8n (nano - fastest). You can use larger models:

| Model | Size | Speed | Accuracy | Config Setting |
|-------|------|-------|----------|----------------|
| yolov8n.pt | 6 MB | Fastest | Good | `model: "models/yolov8n.pt"` |
| yolov8s.pt | 22 MB | Fast | Better | `model: "models/yolov8s.pt"` |
| yolov8m.pt | 52 MB | Medium | High | `model: "models/yolov8m.pt"` |
| yolov8l.pt | 87 MB | Slow | Very High | `model: "models/yolov8l.pt"` |
| yolov8x.pt | 136 MB | Slowest | Best | `model: "models/yolov8x.pt"` |

Models are auto-downloaded on first use.

---

## 🎯 Use Cases

1. **Security Monitoring** - Detect people/vehicles in real-time
2. **Traffic Analysis** - Count cars, track movement patterns
3. **Retail Analytics** - Customer flow and behavior analysis
4. **Wildlife Monitoring** - Detect animals in camera feeds
5. **Quality Control** - Manufacturing defect detection
6. **Research** - Computer vision experiments
7. **Education** - Teaching object detection concepts

---

## 🤝 Contributing

To contribute improvements to the Docker setup:

1. Test your changes with both GPU and CPU versions
2. Update this README with any new features
3. Ensure backward compatibility
4. Document breaking changes

---

## 📝 License

Same license as the main project. See main README.md for details.

---

## 🆘 Getting Help

1. **Check logs**: `cat logs/app.log`
2. **Verify GPU**: `nvidia-smi` (host) and inside container
3. **Test webcam**: `ls -l /dev/video*`
4. **Docker info**: `docker info`
5. **Container logs**: `docker logs yolo-detector-gpu`

**Common Commands:**
```bash
# Enter running container
docker exec -it yolo-detector-gpu bash

# Check GPU inside container
docker exec yolo-detector-gpu nvidia-smi

# View logs
docker logs -f yolo-detector-gpu

# Stop container
docker stop yolo-detector-gpu

# Remove container
docker rm yolo-detector-gpu
```

---

## 🚀 Advanced Usage

### Custom Configuration File
```bash
# Create custom config
cp config/config.yaml config/my-config.yaml

# Edit settings
nano config/my-config.yaml

# Run with custom config
docker run ... yolo-realtime-detection:gpu python3 main.py config/my-config.yaml
```

### Run in Background (Detached Mode)
```bash
docker run -d \
    --name yolo-detector-gpu \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    ... (other options) ...
    yolo-realtime-detection:gpu
```

### Multiple Cameras Simultaneously
```bash
# Camera 1
docker run --name yolo-cam1 --device /dev/video0:/dev/video0 ...

# Camera 2
docker run --name yolo-cam2 --device /dev/video1:/dev/video1 ...
```

### Resource Limits
```bash
docker run \
    --cpus="4.0" \
    --memory="8g" \
    --gpus '"device=0"' \
    ... (other options) ...
    yolo-realtime-detection:gpu
```

---

## 📸 Screenshots

After running, you'll see:
- Real-time video with bounding boxes
- Object labels and confidence scores
- FPS counter
- Recording indicator (when active)

Output recordings saved to `outputs/detection_YYYYMMDD_HHMMSS.mp4`

---

## ✅ Checklist for First Run

- [ ] Docker installed and running
- [ ] NVIDIA drivers installed (GPU version)
- [ ] NVIDIA Docker runtime installed (GPU version)
- [ ] Repository cloned
- [ ] Scripts made executable (`chmod +x *.sh`)
- [ ] Webcam connected (if using webcam mode)
- [ ] X11 permissions granted (`xhost +local:docker`)
- [ ] Config file reviewed and updated
- [ ] Internet connection (for model downloads)

---

## 🎉 Success!

If you see the YOLO detection window with real-time object detection, congratulations! You're now running YOLO with Docker.

**Next steps:**
- Try different input sources (YouTube, video files)
- Experiment with different models
- Adjust confidence thresholds
- Record some detection videos

---

## 📧 Support

For issues specific to Docker setup, please open an issue on GitHub with:
- Your system info (OS, Docker version, GPU model)
- Error messages or logs
- Steps to reproduce the problem
- Whether you're using GPU or CPU version

Happy detecting! 🎯