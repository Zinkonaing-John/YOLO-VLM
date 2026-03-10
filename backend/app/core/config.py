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
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # ── Database ─────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost/aiinspect"

    # ── Redis ────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── AI / Model paths ─────────────────────────────────────────────────
    YOLO_WEIGHTS_PATH: str = "weights/best.pt"
    YOLO_CONFIDENCE: float = 0.25

    # ── VLM (Ollama / LLaVA) ─────────────────────────────────────────────
    VLM_API_URL: str = "http://localhost:11434/api/generate"
    VLM_MODEL: str = "llava:13b"
    VLM_ENABLED: bool = True

    # ── Auth ─────────────────────────────────────────────────────────────
    API_KEY: Optional[str] = None  # None = auth disabled
    API_KEY_HEADER: str = "X-API-Key"

    # ── Storage ──────────────────────────────────────────────────────────
    UPLOAD_DIR: str = "uploads/"

    # ── Color inspection ─────────────────────────────────────────────────
    DELTA_E_THRESHOLD: float = 5.0


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()
