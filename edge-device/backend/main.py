"""Edge Device API — FastAPI backend for Jetson defect inspection.

Lightweight backend that:
  - Captures camera frames and runs two-stage YOLO inference
  - Stores results in local SQLite
  - Serves a vanilla HTML/JS frontend (no Node.js needed)
  - Streams live results via WebSocket
  - Optionally syncs NG results to a cloud backend
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from camera import Camera
from config import get_settings
from database import close_db, get_db
from inference import cleanup_old_images, get_inspections, get_statistics, run_inspection
from models import YOLOClassifier, YOLODetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()

# ── Global state ─────────────────────────────────────────────────────────────

detector = YOLODetector()
classifier = YOLOClassifier()
camera: Camera | None = None
ws_clients: list[WebSocket] = []
_inference_task: asyncio.Task | None = None


# ── WebSocket broadcast ──────────────────────────────────────────────────────

async def broadcast(data: dict):
    """Send JSON to all connected WebSocket clients."""
    payload = json.dumps(data, default=str)
    stale = []
    for ws in ws_clients:
        try:
            await ws.send_text(payload)
        except Exception:
            stale.append(ws)
    for ws in stale:
        ws_clients.remove(ws)


# ── Continuous inference loop ────────────────────────────────────────────────

async def inference_loop():
    """Background task: capture frames, run pipeline, broadcast results."""
    global camera

    logger.info("Starting continuous inference loop")
    frame_count = 0
    cleanup_interval = 1000  # Run cleanup every N frames

    while True:
        if camera is None or not camera.is_opened:
            await asyncio.sleep(0.1)
            continue

        frame = camera.read()
        if frame is None:
            await asyncio.sleep(0.01)
            continue

        frame_count += 1

        # Save frame temporarily
        frame_id = uuid.uuid4().hex[:12]
        img_name = f"{frame_id}.jpg"
        img_path = str(Path(settings.UPLOAD_DIR) / img_name)
        cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

        # Run two-stage pipeline
        result = await run_inspection(frame, img_path, detector, classifier if classifier.is_loaded else None)
        result["image_url"] = f"/uploads/{img_name}"

        # Broadcast to WebSocket clients
        await broadcast(result)

        # Periodic storage cleanup
        if frame_count % cleanup_interval == 0:
            await cleanup_old_images()

        # Yield control — don't block the event loop
        await asyncio.sleep(0.001)


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global camera, _inference_task

    logger.info("Starting %s v%s", settings.APP_TITLE, settings.APP_VERSION)

    # Initialize database
    await get_db()

    # Load models
    detector.load(settings.DET_MODEL_PATH)
    if settings.CLS_MODEL_PATH:
        classifier.load(settings.CLS_MODEL_PATH)
    else:
        logger.info("No classification model configured — running detection-only (fail-safe NG)")

    # Start camera
    try:
        camera = Camera(
            source=settings.CAMERA_SOURCE,
            width=settings.CAMERA_WIDTH,
            height=settings.CAMERA_HEIGHT,
            fps=settings.CAMERA_FPS,
            use_csi=settings.USE_CSI,
            csi_sensor_id=settings.CSI_SENSOR_ID,
        )
        camera.start()
    except Exception:
        logger.warning("Camera not available — upload-only mode")
        camera = None

    # Start background inference loop
    if camera is not None:
        _inference_task = asyncio.create_task(inference_loop())

    logger.info("Edge device ready")
    yield

    # Shutdown
    if _inference_task:
        _inference_task.cancel()
        try:
            await _inference_task
        except asyncio.CancelledError:
            pass

    if camera:
        camera.stop()
    await close_db()
    logger.info("Shutdown complete")


# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded images
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

# Serve frontend static files
frontend_dir = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the edge dashboard."""
    index_path = frontend_dir / "index.html"
    return HTMLResponse(content=index_path.read_text())


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "detector_loaded": detector.is_loaded,
        "classifier_loaded": classifier.is_loaded,
        "camera_active": camera is not None and camera.is_opened,
        "version": settings.APP_VERSION,
    }


@app.post("/api/inspect")
async def inspect_upload(file: UploadFile = File(...)):
    """Upload an image for inspection (manual mode)."""
    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        return {"error": "Invalid image"}

    img_name = f"{uuid.uuid4().hex[:12]}_{file.filename}"
    img_path = str(Path(settings.UPLOAD_DIR) / img_name)
    cv2.imwrite(img_path, image)

    result = await run_inspection(image, img_path, detector, classifier if classifier.is_loaded else None)
    result["image_url"] = f"/uploads/{img_name}"

    await broadcast(result)
    return result


@app.get("/api/inspections")
async def list_inspections(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    verdict: str | None = Query(None),
):
    return await get_inspections(limit=limit, offset=offset, verdict=verdict)


@app.get("/api/statistics")
async def statistics():
    return await get_statistics()


@app.post("/api/camera/start")
async def camera_start():
    """Start the camera and inference loop."""
    global camera, _inference_task

    if camera and camera.is_opened:
        return {"status": "already_running"}

    try:
        camera = Camera(
            source=settings.CAMERA_SOURCE,
            width=settings.CAMERA_WIDTH,
            height=settings.CAMERA_HEIGHT,
            fps=settings.CAMERA_FPS,
            use_csi=settings.USE_CSI,
        )
        camera.start()
        if _inference_task is None or _inference_task.done():
            _inference_task = asyncio.create_task(inference_loop())
        return {"status": "started"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/api/camera/stop")
async def camera_stop():
    """Stop the camera and inference loop."""
    global camera, _inference_task

    if _inference_task and not _inference_task.done():
        _inference_task.cancel()
        try:
            await _inference_task
        except asyncio.CancelledError:
            pass
        _inference_task = None

    if camera:
        camera.stop()
        camera = None

    return {"status": "stopped"}


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """Real-time inspection results stream."""
    await websocket.accept()
    ws_clients.append(websocket)
    logger.info("WebSocket client connected (%d active)", len(ws_clients))
    try:
        while True:
            data = await websocket.receive_text()
            if data.strip().lower() == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in ws_clients:
            ws_clients.remove(websocket)
        logger.info("WebSocket client disconnected (%d active)", len(ws_clients))


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level="info",
    )
