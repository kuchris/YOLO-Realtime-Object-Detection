#!/bin/bash

# YOLO Real-time Object Detection - GPU Docker Run Script
# This script handles all the setup needed to run the application with GPU support

set -e

echo "🚀 YOLO Real-time Object Detection - GPU Docker Setup"
echo "======================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed!${NC}"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if nvidia-docker is installed
if ! docker info | grep -q "nvidia"; then
    echo -e "${YELLOW}⚠️  NVIDIA Docker runtime not detected!${NC}"
    echo "Installing NVIDIA Docker support..."
    echo "You may need to install it manually:"
    echo "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi

# Check if NVIDIA GPU is available
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ NVIDIA GPU or drivers not detected!${NC}"
    echo "This script requires an NVIDIA GPU with drivers installed."
    echo "Use docker-run-cpu.sh for CPU-only mode."
    exit 1
fi

echo -e "${GREEN}✅ NVIDIA GPU detected:${NC}"
nvidia-smi --query-gpu=name --format=csv,noheader

# Change to project root directory
cd "$(dirname "$0")/.."

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p config models videos outputs logs

# Allow X11 connections from Docker
echo "🖥️  Setting up display access..."
xhost +local:docker > /dev/null 2>&1 || echo "Note: xhost command not found, display may not work"

# Build Docker image if it doesn't exist
if [[ "$(docker images -q yolo-realtime-detection:gpu 2> /dev/null)" == "" ]]; then
    echo "🔨 Building Docker image (this may take several minutes)..."
    docker build -t yolo-realtime-detection:gpu -f docker/Dockerfile .
else
    echo -e "${GREEN}✅ Docker image already exists${NC}"
fi

# Stop any existing container
if [ "$(docker ps -aq -f name=yolo-detector-gpu)" ]; then
    echo "🧹 Cleaning up existing container..."
    docker rm -f yolo-detector-gpu > /dev/null 2>&1
fi

# Run the container
echo "🎬 Starting YOLO detector with GPU support..."
echo ""
echo "Controls:"
echo "  Q/ESC  - Quit"
echo "  SPACE  - Pause/Resume"
echo "  C      - Start/Stop recording"
echo "  S      - Save frame"
echo "  H      - Toggle help"
echo ""

docker run -it --rm \
    --name yolo-detector-gpu \
    --gpus all \
    --runtime=nvidia \
    -e DISPLAY=$DISPLAY \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $(pwd)/config:/app/config:rw \
    -v $(pwd)/models:/app/models:rw \
    -v $(pwd)/videos:/app/videos:rw \
    -v $(pwd)/outputs:/app/outputs:rw \
    -v $(pwd)/logs:/app/logs:rw \
    --device /dev/video0:/dev/video0 \
    --network host \
    --ipc host \
    yolo-realtime-detection:gpu "$@"

# Cleanup X11 permissions
echo ""
echo "🧹 Cleaning up..."
xhost -local:docker > /dev/null 2>&1 || true

echo -e "${GREEN}✅ Done!${NC}"
