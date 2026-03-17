"""Typed, lazy-loading model registry.

Replaces the plain app_state dict with a typed container.
Models load on first access and stay cached for the process lifetime.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class ModelRegistry:
    """Central registry for all AI models.

    Models are loaded eagerly via ``load_all()`` at startup, but each
    loader is fault-tolerant — a missing weights file disables that model
    without crashing the application.
    """

    _yolo_defect: Any = field(default=None, repr=False)
    _yolo_seg: Any = field(default=None, repr=False)
    _clip: Any = field(default=None, repr=False)
    _resnet: Any = field(default=None, repr=False)

    # ── Public accessors ─────────────────────────────────────────────

    @property
    def yolo_defect(self):
        """YOLODetector for industrial defect detection."""
        return self._yolo_defect

    @property
    def yolo_seg(self):
        """YOLOSegmentor for instance segmentation."""
        return self._yolo_seg

    @property
    def clip(self):
        """CLIPClassifier for zero-shot OK/NG classification."""
        return self._clip

    @property
    def resnet(self):
        """ResNetClassifier for CNN-based OK/NG classification."""
        return self._resnet

    # ── Loader ───────────────────────────────────────────────────────

    def load_all(self) -> None:
        """Load all models.  Each failure is logged but does not halt startup."""
        self._load_yolo_defect()
        self._load_yolo_seg()
        self._load_clip()
        self._load_resnet()
        logger.info("Model registry ready: %s", self.summary())

    def summary(self) -> dict[str, bool]:
        """Return a dict of model-name -> is_loaded."""
        return {
            "yolo_defect": self._yolo_defect is not None and self._yolo_defect.is_loaded,
            "yolo_seg": self._yolo_seg is not None and self._yolo_seg.is_loaded,
            "clip": self._clip is not None and self._clip.is_loaded,
            "resnet": self._resnet is not None and self._resnet.is_loaded,
        }

    # ── Private loaders ──────────────────────────────────────────────

    def _load_yolo_defect(self) -> None:
        try:
            from app.models.ai_models import YOLODetector

            det = YOLODetector()
            det.load_model(settings.YOLO_DEFECT_WEIGHTS_PATH)
            self._yolo_defect = det
        except Exception:
            logger.exception("Failed to load YOLO defect detector")

    def _load_yolo_seg(self) -> None:
        try:
            from app.models.ai_models import YOLOSegmentor

            seg = YOLOSegmentor()
            seg.load_model(settings.YOLO_SEG_WEIGHTS_PATH)
            self._yolo_seg = seg
        except Exception:
            logger.exception("Failed to load YOLO segmentor")

    def _load_clip(self) -> None:
        try:
            from app.models.ai_models import CLIPClassifier

            cls = CLIPClassifier()
            cls.load_model(
                model_name=settings.CLIP_MODEL_NAME,
                ok_labels=settings.CLIP_LABELS_OK,
                ng_labels=settings.CLIP_LABELS_NG,
            )
            self._clip = cls
        except Exception:
            logger.exception("Failed to load CLIP classifier")

    def _load_resnet(self) -> None:
        try:
            from app.models.cnn_models import ResNetClassifier

            cls = ResNetClassifier()
            weights = str(
                Path(__file__).resolve().parents[2] / settings.CNN_RESNET_WEIGHTS_PATH
            )
            cls.load_model(
                weights_path=weights,
                model_arch=settings.CNN_RESNET_ARCH,
                num_classes=settings.CNN_RESNET_NUM_CLASSES,
                class_names=settings.CNN_RESNET_CLASS_NAMES,
            )
            self._resnet = cls
        except Exception:
            logger.exception("Failed to load ResNet classifier")


# Singleton
_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """Return the global model registry (create on first call)."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
