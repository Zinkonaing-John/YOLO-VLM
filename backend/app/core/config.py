"""Central application configuration via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── App ──────────────────────────────────────────────────────────────
    APP_TITLE: str = "Industrial Defect Detection API"
    APP_VERSION: str = "4.0.0"
    DEBUG: bool = False

    # ── Database ─────────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./aiinspect.db"

    # ── Storage ──────────────────────────────────────────────────────────
    UPLOAD_DIR: str = "uploads/"

    # ── YOLO ─────────────────────────────────────────────────────────────
    YOLO_DEFECT_WEIGHTS_PATH: str = "weights/best.pt"
    YOLO_DEFECT_CONFIDENCE: float = 0.25

    # ── PatchCore ────────────────────────────────────────────────────────
    PATCHCORE_WEIGHTS_PATH: str = "weights/patchcore_memory.pt"
    PATCHCORE_THRESHOLD: float = 0.5

    # ── EfficientAD ──────────────────────────────────────────────────────
    EFFICIENTAD_WEIGHTS_PATH: str = "models/efficientad/model.ckpt"
    EFFICIENTAD_THRESHOLD: float = 0.5

    # ── CLIP ─────────────────────────────────────────────────────────────
    CLIP_MODEL_NAME: str = "ViT-B/32"
    CLIP_DEFECT_THRESHOLD: float = 0.5
    CLIP_LABELS_OK: list[str] = [
        "a photo of a smooth clean metal surface",
        "a photo of a flawless steel product",
        "a photo of a normal metal surface without defects",
        "a photo of an undamaged uniform metal surface",
        "a photo of a high-quality defect-free steel sheet",
        "a photo of a pristine metal part with no visible flaws",
    ]
    CLIP_LABELS_NG: list[str] = [
        "a photo of a scratched metal surface",
        "a photo of a cracked metal surface",
        "a photo of a metal surface with crazing defects",
        "a photo of a metal surface with inclusion defects",
        "a photo of a pitted metal surface",
        "a photo of a metal surface with rolled-in scale",
    ]

    # ── ResNet CNN ────────────────────────────────────────────────────────
    CNN_RESNET_WEIGHTS_PATH: str = "weights/resnet_classifier.pth"
    CNN_RESNET_ARCH: str = "resnet18"
    CNN_RESNET_NUM_CLASSES: int = 2
    CNN_RESNET_CLASS_NAMES: list[str] = ["OK", "NG"]
    CNN_RESNET_THRESHOLD: float = 0.5

    # ── VLM (local Ollama) ───────────────────────────────────────────────
    VLM_ENABLED: bool = True
    VLM_MODEL: str = "qwen2.5vl:7b"
    VLM_OLLAMA_URL: str = "http://localhost:11434/v1"

    # ── Auth ─────────────────────────────────────────────────────────────
    API_KEY: Optional[str] = None
    API_KEY_HEADER: str = "X-API-Key"


@lru_cache
def get_settings() -> Settings:
    return Settings()
