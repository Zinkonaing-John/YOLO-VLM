"""Application configuration using pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── App ──────────────────────────────────────────────────────────────
    APP_TITLE: str = "Industrial AI Vision API"
    APP_VERSION: str = "3.0.0"
    DEBUG: bool = False

    # ── Database ─────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost/aiinspect"

    # ── YOLO Defect Detection ────────────────────────────────────────────
    YOLO_DEFECT_WEIGHTS_PATH: str = "weights/best.pt"
    YOLO_DEFECT_CONFIDENCE: float = 0.15

    # ── YOLO Segmentation ────────────────────────────────────────────────
    YOLO_SEG_WEIGHTS_PATH: str = "yolov8n-seg.pt"
    YOLO_SEG_CONFIDENCE: float = 0.25

    # ── CLIP Classification ──────────────────────────────────────────────
    CLIP_MODEL_NAME: str = "ViT-B/32"
    CLIP_DEFECT_THRESHOLD: float = 0.5
    CLIP_LABELS_OK: list[str] = [
        "a photo of a smooth clean metal surface",
        "a photo of a flawless steel product",
        "a photo of a normal metal surface without defects",
    ]
    CLIP_LABELS_NG: list[str] = [
        "a photo of a scratched metal surface",
        "a photo of a cracked metal surface",
        "a photo of a metal surface with crazing defects",
        "a photo of a metal surface with inclusion defects",
        "a photo of a pitted metal surface",
        "a photo of a metal surface with rolled-in scale",
    ]

    # ── CNN / ResNet Classification ──────────────────────────────────────
    CNN_RESNET_WEIGHTS_PATH: str = "weights/resnet_classifier.pth"
    CNN_RESNET_ARCH: str = "resnet18"
    CNN_RESNET_NUM_CLASSES: int = 2
    CNN_RESNET_CLASS_NAMES: list[str] = ["OK", "NG"]
    CNN_RESNET_THRESHOLD: float = 0.5

    # ── Auth ─────────────────────────────────────────────────────────────
    API_KEY: Optional[str] = None
    API_KEY_HEADER: str = "X-API-Key"

    # ── Storage ──────────────────────────────────────────────────────────
    UPLOAD_DIR: str = "uploads/"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()
