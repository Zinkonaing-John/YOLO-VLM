#!/bin/bash
# Quick start script for edge device — runs without Docker
#
# Usage:
#   ./start.sh                          # USB camera, defaults
#   ./start.sh --camera rtsp://...      # RTSP stream
#   ./start.sh --csi                    # CSI camera (Jetson)
#
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Create data directories
mkdir -p data/uploads

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --camera)   export CAMERA_SOURCE="$2"; shift 2 ;;
        --csi)      export USE_CSI=true; shift ;;
        --det)      export DET_MODEL_PATH="$2"; shift 2 ;;
        --cls)      export CLS_MODEL_PATH="$2"; shift 2 ;;
        --port)     export PORT="$2"; shift 2 ;;
        --mqtt)     export MQTT_BROKER="$2"; shift 2 ;;
        *)          echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Check for required packages
python3 -c "import fastapi" 2>/dev/null || {
    echo "[INFO] Installing Python dependencies..."
    pip3 install -r backend/requirements.txt
}

python3 -c "import ultralytics" 2>/dev/null || {
    echo "[WARN] ultralytics not installed. Install it with:"
    echo "  pip3 install ultralytics"
    echo ""
    echo "For Jetson, follow: https://docs.ultralytics.com/guides/nvidia-jetson/"
    exit 1
}

# Check for model weights
DET_MODEL="${DET_MODEL_PATH:-weights/best.engine}"
if [ ! -f "$DET_MODEL" ]; then
    # Try .pt fallback
    PT_FALLBACK="${DET_MODEL%.engine}.pt"
    if [ -f "$PT_FALLBACK" ]; then
        echo "[WARN] TensorRT engine not found. Using PyTorch model: $PT_FALLBACK"
        echo "[INFO] Export to TensorRT for best performance:"
        echo "  yolo export model=$PT_FALLBACK format=engine half=True"
    else
        echo "[ERROR] No detection model found at $DET_MODEL"
        echo "Place your model weights in the weights/ directory."
        exit 1
    fi
fi

echo "============================================"
echo "  Edge Defect Inspector"
echo "============================================"
echo "  Camera:    ${CAMERA_SOURCE:-0}"
echo "  Det model: ${DET_MODEL_PATH:-weights/best.engine}"
echo "  Cls model: ${CLS_MODEL_PATH:-<none — detection-only>}"
echo "  Port:      ${PORT:-8000}"
echo "  MQTT:      ${MQTT_BROKER:-<disabled>}"
echo "============================================"
echo ""
echo "  Dashboard: http://localhost:${PORT:-8000}"
echo ""

cd backend
exec python3 main.py
