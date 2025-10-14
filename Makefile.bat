# Makefile for YOLO Real-time Object Detection Docker

.PHONY: help build-gpu build-cpu run-gpu run-cpu stop clean logs setup check

# Default target
help:
	@echo "🚀 YOLO Real-time Object Detection - Docker Commands"
	@echo "=================================================="
	@echo ""
	@echo "Setup:"
	@echo "  make setup        - Create directories and set permissions"
	@echo "  make check        - Check prerequisites (Docker, NVIDIA, etc.)"
	@echo ""
	@echo "Build:"
	@echo "  make build-gpu    - Build GPU-enabled Docker image"
	@echo "  make build-cpu    - Build CPU-only Docker image"
	@echo "  make build-all    - Build both GPU and CPU images"
	@echo ""
	@echo "Run:"
	@echo "  make run-gpu      - Run with GPU support (default)"
	@echo "  make run-cpu      - Run with CPU only"
	@echo ""
	@echo "Manage:"
	@echo "  make stop         - Stop running containers"
	@echo "  make logs         - View container logs"
	@echo "  make shell-gpu    - Open shell in GPU container"
	@echo "  make shell-cpu    - Open shell in CPU container"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        - Remove containers"
	@echo "  make clean-all    - Remove containers and images"
	@echo "  make clean-data   - Remove generated data (models, videos, outputs)"
	@echo ""

# Setup directories and permissions
setup:
	@echo "📁 Creating directories..."
	@mkdir -p config models videos outputs logs
	@echo "✅ Directories created"
	@echo "🔑 Setting X11 permissions..."
	@xhost +local:docker 2>/dev/null || echo "⚠️  xhost not found"
	@echo "✅ Setup complete!"

# Check prerequisites
check:
	@echo "🔍 Checking prerequisites..."
	@command -v docker >/dev/null 2>&1 || { echo "❌ Docker not installed"; exit 1; }
	@echo "✅ Docker: $(shell docker --version)"
	@command -v docker-compose >/dev/null 2>&1 && echo "✅ Docker Compose: $(shell docker-compose --version)" || echo "⚠️  Docker Compose not found"
	@command -v nvidia-smi >/dev/null 2>&1 && echo "✅ NVIDIA GPU: $(shell nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)" || echo "⚠️  NVIDIA GPU not detected"
	@docker info | grep -q nvidia && echo "✅ NVIDIA Docker Runtime: Available" || echo "⚠️  NVIDIA Docker Runtime not configured"
	@echo "✅ All checks complete!"

# Build GPU image
build-gpu:
	@echo "🔨 Building GPU-enabled Docker image..."
	docker build -t yolo-realtime-detection:gpu -f Dockerfile .
	@echo "✅ GPU image built successfully!"

# Build CPU image
build-cpu:
	@echo "🔨 Building CPU-only Docker image..."
	docker build -t yolo-realtime-detection:cpu -f Dockerfile.cpu .
	@echo "✅ CPU image built successfully!"

# Build both images
build-all: build-gpu build-cpu
	@echo "✅ All images built successfully!"

# Run GPU version
run-gpu: setup
	@echo "🎬 Starting YOLO detector with GPU support..."
	@chmod +x docker-run-gpu.sh
	@./docker-run-gpu.sh

# Run CPU version
run-cpu: setup
	@echo "🎬 Starting YOLO detector with CPU..."
	@chmod +x docker-run-cpu.sh
	@./docker-run-cpu.sh

# Stop containers
stop:
	@echo "🛑 Stopping containers..."
	@docker stop yolo-detector-gpu 2>/dev/null || true
	@docker stop yolo-detector-cpu 2>/dev/null || true
	@echo "✅ Containers stopped"

# View logs
logs:
	@echo "📋 Container logs (GPU):"
	@docker logs yolo-detector-gpu 2>/dev/null || echo "No GPU container running"
	@echo ""
	@echo "📋 Application logs:"
	@cat logs/app.log 2>/dev/null || echo "No logs yet"

# Open shell in GPU container
shell-gpu:
	@docker exec -it yolo-detector-gpu bash || echo "GPU container not running. Start it with 'make run-gpu'"

# Open shell in CPU container
shell-cpu:
	@docker exec -it yolo-detector-cpu bash || echo "CPU container not running. Start it with 'make run-cpu'"

# Clean containers
clean:
	@echo "🧹 Cleaning containers..."
	@docker rm -f yolo-detector-gpu 2>/dev/null || true
	@docker rm -f yolo-detector-cpu 2>/dev/null || true
	@echo "✅ Containers removed"

# Clean containers and images
clean-all: clean
	@echo "🧹 Cleaning images..."
	@docker rmi yolo-realtime-detection:gpu 2>/dev/null || true
	@docker rmi yolo-realtime-detection:cpu 2>/dev/null || true
	@echo "✅ Images removed"

# Clean generated data
clean-data:
	@echo "🧹 Cleaning generated data..."
	@read -p "This will delete models, videos, outputs, and logs. Continue? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf models/* videos/* outputs/* logs/*; \
		echo "✅ Data cleaned"; \
	else \
		echo "❌ Cancelled"; \
	fi

# Quick test
test-gpu: build-gpu
	@echo "🧪 Testing GPU version..."
	docker run --rm --gpus all yolo-realtime-detection:gpu python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

test-cpu: build-cpu
	@echo "🧪 Testing CPU version..."
	docker run --rm yolo-realtime-detection:cpu python3 -c "import torch; print('PyTorch:', torch.__version__)"

# Rebuild without cache
rebuild-gpu:
	@echo "🔨 Rebuilding GPU image (no cache)..."
	docker build --no-cache -t yolo-realtime-detection:gpu -f Dockerfile .

rebuild-cpu:
	@echo "🔨 Rebuilding CPU image (no cache)..."
	docker build --no-cache -t yolo-realtime-detection:cpu -f Dockerfile.cpu .

# Show disk usage
disk-usage:
	@echo "💾 Docker disk usage:"
	@docker system df
	@echo ""
	@echo "📊 Project disk usage:"
	@du -sh models videos outputs logs 2>/dev/null || echo "No data directories yet"

# Prune Docker system
prune:
	@echo "🧹 Pruning Docker system..."
	docker system prune -f
	@echo "✅ Docker system pruned"