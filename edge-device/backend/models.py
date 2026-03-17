"""YOLO model wrappers for edge deployment — detection + classification."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    logger.warning("ultralytics not installed — models will not load")


@dataclass
class Detection:
    """Single detection result from YOLO-det."""
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    # Normalized 0-1 coordinates
    nx1: float
    ny1: float
    nx2: float
    ny2: float
    # Stage 2 classification (filled after classify_roi)
    roi_verdict: str | None = None
    roi_confidence: float | None = None

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": {
                "x1": round(self.x1, 1), "y1": round(self.y1, 1),
                "x2": round(self.x2, 1), "y2": round(self.y2, 1),
            },
            "bbox_norm": {
                "x1": round(self.nx1, 4), "y1": round(self.ny1, 4),
                "x2": round(self.nx2, 4), "y2": round(self.ny2, 4),
            },
            "roi_verdict": self.roi_verdict,
            "roi_confidence": round(self.roi_confidence, 4) if self.roi_confidence is not None else None,
        }


class YOLODetector:
    """Stage 1 — YOLO object/defect detection with TensorRT support."""

    def __init__(self) -> None:
        self._model: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self, model_path: str) -> bool:
        if YOLO is None:
            logger.error("ultralytics not installed")
            return False

        resolved = self._resolve_path(model_path)
        if resolved is None:
            logger.error("Detection model not found: %s", model_path)
            return False

        try:
            self._model = YOLO(str(resolved))
            logger.info("Detection model loaded: %s", resolved)
            return True
        except Exception:
            logger.exception("Failed to load detection model: %s", resolved)
            return False

    def detect(
        self,
        image: np.ndarray,
        conf: float = 0.5,
        iou: float = 0.45,
        img_size: int = 640,
    ) -> list[Detection]:
        if self._model is None:
            return []

        h, w = image.shape[:2]
        results = self._model.predict(
            source=image, conf=conf, iou=iou, imgsz=img_size, verbose=False,
        )

        detections: list[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for i in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                detections.append(Detection(
                    class_id=int(r.boxes.cls[i]),
                    class_name=r.names.get(int(r.boxes.cls[i]), "unknown"),
                    confidence=float(r.boxes.conf[i]),
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    nx1=x1 / w, ny1=y1 / h, nx2=x2 / w, ny2=y2 / h,
                ))
        return detections

    @staticmethod
    def _resolve_path(model_path: str) -> Path | None:
        path = Path(model_path)
        if path.exists():
            return path
        # .engine → .pt fallback
        fallback = path.with_suffix(".pt")
        if fallback.exists():
            return fallback
        return None


class YOLOClassifier:
    """Stage 2 — YOLO-cls ROI classification (OK/NG) with TensorRT support."""

    _OK_KEYWORDS = ("ok", "good", "normal", "pass")

    def __init__(self) -> None:
        self._model: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self, model_path: str) -> bool:
        if YOLO is None:
            logger.error("ultralytics not installed")
            return False

        path = Path(model_path)
        if not path.exists():
            fallback = path.with_suffix(".pt")
            if not fallback.exists():
                logger.warning("Classification model not found: %s", model_path)
                return False
            path = fallback

        try:
            self._model = YOLO(str(path))
            logger.info("Classification model loaded: %s", path)
            return True
        except Exception:
            logger.exception("Failed to load classification model: %s", path)
            return False

    def classify(
        self,
        roi: np.ndarray,
        confidence_threshold: float = 0.5,
    ) -> tuple[str, float]:
        """Classify a cropped ROI as OK or NG.

        Returns (verdict, confidence).  Fail-safe: returns NG on any error.
        """
        if self._model is None:
            return "NG", 0.0

        try:
            results = self._model.predict(source=roi, verbose=False)
        except Exception:
            logger.exception("Classification inference failed")
            return "NG", 0.0

        if not results or results[0].probs is None:
            return "NG", 0.0

        probs = results[0].probs
        top1_idx = int(probs.top1)
        top1_conf = float(probs.top1conf)
        class_name = results[0].names.get(top1_idx, "").lower()

        is_ok = any(kw in class_name for kw in self._OK_KEYWORDS)

        if is_ok and top1_conf >= confidence_threshold:
            return "OK", top1_conf
        else:
            return "NG", top1_conf if not is_ok else 1.0 - top1_conf
