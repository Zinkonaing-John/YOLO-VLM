"""Edge device configuration — lightweight settings for Jetson deployment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Base directory is edge-device/
BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class EdgeSettings:
    """All edge-device settings, loaded from environment variables."""

    # ── App ──────────────────────────────────────────────────────────────
    APP_TITLE: str = "Edge Defect Inspector"
    APP_VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── Database (SQLite — no external DB needed) ────────────────────────
    DATABASE_PATH: str = str(BASE_DIR / "data" / "inspections.db")

    # ── Camera ───────────────────────────────────────────────────────────
    CAMERA_SOURCE: str = "0"
    CAMERA_WIDTH: int = 1280
    CAMERA_HEIGHT: int = 720
    CAMERA_FPS: int = 30
    USE_CSI: bool = False
    CSI_SENSOR_ID: int = 0

    # ── Stage 1: YOLO Detection ──────────────────────────────────────────
    DET_MODEL_PATH: str = str(BASE_DIR / "weights" / "best.engine")
    DET_CONFIDENCE: float = 0.5
    DET_IOU: float = 0.45
    DET_IMG_SIZE: int = 640

    # ── Stage 2: YOLO Classification ─────────────────────────────────────
    CLS_MODEL_PATH: str = ""  # Empty = detection-only mode
    CLS_CONFIDENCE: float = 0.5

    # ── Upstream sync (optional — send NG results to cloud backend) ──────
    UPSTREAM_API_URL: str = ""
    UPSTREAM_UPLOAD_NG_ONLY: bool = True

    # ── MQTT (optional) ──────────────────────────────────────────────────
    MQTT_BROKER: str = ""
    MQTT_PORT: int = 1883
    MQTT_CLIENT_ID: str = "jetson-edge-001"

    # ── Storage ──────────────────────────────────────────────────────────
    UPLOAD_DIR: str = str(BASE_DIR / "data" / "uploads")
    MAX_STORED_IMAGES: int = 5000  # Auto-cleanup oldest when exceeded

    # ── Inference ────────────────────────────────────────────────────────
    MIN_ROI_SIZE: int = 32  # Minimum crop size in pixels

    def __post_init__(self):
        """Override defaults from environment variables."""
        for fld in self.__dataclass_fields__:
            env_val = os.environ.get(fld)
            if env_val is not None:
                current = getattr(self, fld)
                if isinstance(current, bool):
                    setattr(self, fld, env_val.lower() in ("1", "true", "yes"))
                elif isinstance(current, int):
                    setattr(self, fld, int(env_val))
                elif isinstance(current, float):
                    setattr(self, fld, float(env_val))
                else:
                    setattr(self, fld, env_val)

        # Ensure directories exist
        Path(self.DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)
        Path(self.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)


_settings: EdgeSettings | None = None


def get_settings() -> EdgeSettings:
    global _settings
    if _settings is None:
        _settings = EdgeSettings()
    return _settings
