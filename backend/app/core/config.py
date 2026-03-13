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
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False

    # ── Database ─────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost/aiinspect"

    # ── Redis ────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── AI / Model paths ─────────────────────────────────────────────────
    YOLO_WEIGHTS_PATH: str = "yolov8m.pt"
    YOLO_CONFIDENCE: float = 0.20

    # ── CLIP defect classification ───────────────────────────────────────
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

    # ── Auth ─────────────────────────────────────────────────────────────
    API_KEY: Optional[str] = None  # None = auth disabled
    API_KEY_HEADER: str = "X-API-Key"

    # ── Storage ──────────────────────────────────────────────────────────
    UPLOAD_DIR: str = "uploads/"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()
