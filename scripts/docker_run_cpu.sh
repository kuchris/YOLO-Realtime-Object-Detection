#!/bin/bash

# YOLO Real-time Object Detection - CPU Docker Run Script
# This script runs the application in CPU-only mode (no GPU required)

set -e

echo "🚀 YOLO Real-time Object Detection - CPU Docker Setup"
echo "====================================================="

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

echo -e "${GREEN}✅ Docker detected${NC}"

# Change to project root directory
cd "$(dirname "$0")/.."

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p config models videos outputs logs

# Check if config.yaml exists and update device to CPU
if [ -f "config/config.yaml" ]; then
    echo "⚙️  Updating config to use CPU..."
    # Use sed to change device to cpu if it's set to cuda
    sed -i.bak 's/device: "cuda"/device: "cpu"/' config/config.yaml 2>/dev/null || \
    sed -i '' 's/device: "cuda"/device: "cpu"/' config/config.yaml 2>/dev/null || \
    echo "Note: Could not auto-update config. Please manually set device: 'cpu' in config/config.yaml"
else
    echo -e "${YELLOW}⚠️  config/config.yaml not found. Make sure device is set to 'cpu'${NC}"
fi

# Allow X11 connections from Docker
echo "🖥️  Setting up display access..."
xhost +local:docker > /dev/null 2>&1 || echo "Note: xhost command not found, display may not work"

# Build Docker image if it doesn't exist
if [[ "$(docker images -q yolo-realtime-detection:cpu 2> /dev/null)" == "" ]]; then
    echo "🔨 Building Docker image (this may take several minutes)..."
    docker build -t yolo-realtime-detection:cpu -f docker/Dockerfile.cpu .
else
    echo -e "${GREEN}✅ Docker image already exists${NC}"
fi

# Stop any existing container
if [ "$(docker ps -aq -f name=yolo-detector-cpu)" ]; then
    echo "🧹 Cleaning up existing container..."
    docker rm -f yolo-detector-cpu > /dev/null 2>&1
fi

# Run the container
echo "🎬 Starting YOLO detector with CPU (slower performance)..."
echo ""
echo -e "${YELLOW}Note: CPU mode is slower than GPU mode (5-15 FPS vs 30-60 FPS)${NC}"
echo ""
echo "Controls:"
echo "  Q/ESC  - Quit"
echo "  SPACE  - Pause/Resume"
echo "  C      - Start/Stop recording"
echo "  S      - Save frame"
echo "  H      - Toggle help"
echo ""

docker run -it --rm \
    --name yolo-detector-cpu \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $(pwd)/config:/app/config:rw \
    -v $(pwd)/models:/app/models:rw \
    -v $(pwd)/videos:/app/videos:rw \
    -v $(pwd)/outputs:/app/outputs:rw \
    -v $(pwd)/logs:/app/logs:rw \
    --device /dev/video0:/dev/video0 \
    --network host \
    --ipc host \
    yolo-realtime-detection:cpu "$@"

# Cleanup X11 permissions
echo ""
echo "🧹 Cleaning up..."
xhost -local:docker > /dev/null 2>&1 || true

echo -e "${GREEN}✅ Done!${NC}"
