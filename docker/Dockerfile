# YOLO Real-time Object Detection - GPU Dockerfile
# Base image with CUDA support for NVIDIA GPUs
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3-dev \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # X11 for display
    libx11-6 \
    libxcb1 \
    libxau6 \
    # Video processing
    ffmpeg \
    # Utilities
    wget \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Copy requirements file
COPY requirements.txt .

# Install PyTorch with CUDA 12.1 support
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Upgrade yt-dlp to latest version for YouTube downloads
RUN pip3 install --no-cache-dir --upgrade yt-dlp

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/models /app/videos /app/outputs /app/logs /app/config

# Set permissions
RUN chmod -R 755 /app

# Expose any ports if needed (optional for future web interface)
# EXPOSE 8080

# Set environment variables for optimal performance
ENV OMP_NUM_THREADS=8
ENV OPENCV_VIDEOIO_PRIORITY_MSMF=0
ENV QT_QPA_PLATFORM=offscreen
ENV MPLBACKEND=Agg

# Healthcheck to verify application is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; print('CUDA:', torch.cuda.is_available())" || exit 1

# Default command - can be overridden
CMD ["python3", "main.py"]