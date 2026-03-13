"""Industrial AI Vision API – FastAPI application entry point."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import get_settings
from app.core.database import dispose_db, init_db
from app.models.ai_models import CLIPClassifier, YOLODetector
from app.routers import auth, inspection, statistics

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()

# ── Global application state ─────────────────────────────────────────────────

app_state: dict[str, Any] = {}


# ── WebSocket connection manager ─────────────────────────────────────────────


class ConnectionManager:
    """Manages active WebSocket connections for real-time inspection feed."""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.append(websocket)
        logger.info(
            "WebSocket client connected (%d active)", len(self._connections)
        )

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            if websocket in self._connections:
                self._connections.remove(websocket)
        logger.info(
            "WebSocket client disconnected (%d active)", len(self._connections)
        )

    async def broadcast(self, data: dict[str, Any]) -> None:
        """Send a JSON message to every connected client."""
        payload = json.dumps(data, default=str)
        async with self._lock:
            stale: list[WebSocket] = []
            for ws in self._connections:
                try:
                    await ws.send_text(payload)
                except Exception:
                    stale.append(ws)
            for ws in stale:
                self._connections.remove(ws)


ws_manager = ConnectionManager()


# ── Lifespan ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle hook."""
    logger.info("Starting %s v%s", settings.APP_TITLE, settings.APP_VERSION)

    # Initialise database tables
    await init_db()
    logger.info("Database tables ensured")

    # Load YOLO model
    detector = YOLODetector()
    detector.load_model(settings.YOLO_WEIGHTS_PATH)
    app_state["detector"] = detector

    # Load CLIP classifier
    clip_classifier = CLIPClassifier()
    clip_classifier.load_model(
        model_name=settings.CLIP_MODEL_NAME,
        ok_labels=settings.CLIP_LABELS_OK,
        ng_labels=settings.CLIP_LABELS_NG,
    )
    app_state["clip_classifier"] = clip_classifier
    logger.info("CLIP classifier initialized (loaded=%s)", clip_classifier.is_loaded)

    # Load SimpleNet anomaly detector
    # weights/ is at project root, one level above backend/
    simplenet_path = str(Path(__file__).resolve().parents[2] / "weights" / "simplenet.pth")
    try:
        from app.models.simplenet import SimpleNet
        simplenet = SimpleNet(backbone="resnet18", input_size=256)
        if Path(simplenet_path).exists():
            simplenet.load(simplenet_path)
            app_state["simplenet"] = simplenet
            logger.info("SimpleNet loaded from %s", simplenet_path)
        else:
            app_state["simplenet"] = None
            logger.warning("SimpleNet weights not found at %s", simplenet_path)
    except Exception:
        app_state["simplenet"] = None
        logger.exception("Failed to load SimpleNet")

    logger.info("Application startup complete")
    yield

    # Shutdown
    logger.info("Shutting down …")
    await dispose_db()
    logger.info("Database connections closed")


# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description=(
        "REST + WebSocket API for real-time industrial quality inspection "
        "powered by YOLO object detection and CLIP defect classification."
    ),
    lifespan=lifespan,
)

# CORS – allow all origins (tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────────────────

app.include_router(inspection.router)
app.include_router(statistics.router)
app.include_router(auth.router)

# ── Static files (uploaded images) ──────────────────────────────────────────
uploads_dir = Path(settings.UPLOAD_DIR)
uploads_dir.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")


# ── Health check ─────────────────────────────────────────────────────────────


@app.get("/health", tags=["system"])
async def health_check() -> dict[str, Any]:
    """Basic liveness / readiness probe."""
    detector: YOLODetector = app_state.get("detector")  # type: ignore[assignment]
    clip_cls: CLIPClassifier = app_state.get("clip_classifier")  # type: ignore[assignment]
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": settings.APP_VERSION,
        "yolo_loaded": detector.is_loaded if detector else False,
        "clip_loaded": clip_cls.is_loaded if clip_cls else False,
    }


# ── WebSocket endpoint ──────────────────────────────────────────────────────


@app.websocket("/ws/inspection")
async def websocket_inspection(websocket: WebSocket) -> None:
    """Real-time inspection feed.

    Clients connect here to receive live JSON messages whenever an
    inspection completes.  The server also accepts ``ping`` messages
    and replies with ``pong`` to keep the connection alive.
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data.strip().lower() == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception:
        await ws_manager.disconnect(websocket)
