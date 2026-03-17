# Dockerfile for Jetson edge device
# Base: NVIDIA L4T with PyTorch pre-installed for Jetson
#
# Build:  docker build -t edge-inspector .
# Run:    docker run --runtime nvidia --device /dev/video0 -p 8000:8000 edge-inspector

ARG BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.1-py3
FROM ${BASE_IMAGE}

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install ultralytics (uses pre-installed PyTorch from base image)
RUN pip install --no-cache-dir ultralytics

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY weights/ ./weights/

# Create data directories
RUN mkdir -p data/uploads

# Expose port
EXPOSE 8000

# Environment defaults (override with docker run -e or .env file)
ENV CAMERA_SOURCE=0
ENV DET_MODEL_PATH=/app/weights/best.engine
ENV HOST=0.0.0.0
ENV PORT=8000

WORKDIR /app/backend
CMD ["python", "main.py"]
