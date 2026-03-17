"""Unified inspection pipeline with strategy selection.

Supports three pipeline modes:
  - yolo_clip:  YOLO detection + CLIP per-ROI classification
  - cnn:        ResNet full-image binary classification
  - ensemble:   Both pipelines vote; any NG → final NG (fail-safe)
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.db_models import Defect, Inspection

logger = logging.getLogger(__name__)
settings = get_settings()

PipelineMode = Literal["yolo_clip", "cnn", "ensemble"]


@dataclass
class InspectionResult:
    """Data transfer object returned after an inspection completes."""
    inspection_id: uuid.UUID
    verdict: str
    total_defects: int
    processing_ms: float
    pipeline: str
    detections: list[dict[str, Any]] = field(default_factory=list)
    image_path: str = ""
    cnn_confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "inspection_id": str(self.inspection_id),
            "verdict": self.verdict,
            "total_defects": self.total_defects,
            "processing_ms": round(self.processing_ms, 2),
            "pipeline": self.pipeline,
            "detections": self.detections,
            "image_path": self.image_path,
        }
        if self.cnn_confidence is not None:
            d["cnn_confidence"] = round(self.cnn_confidence, 4)
        return d


# ── Pipeline stages ──────────────────────────────────────────────────────────


def _run_yolo_clip(
    image: np.ndarray,
    detector,
    clip_classifier,
) -> tuple[str, list[dict[str, Any]]]:
    """YOLO detection + per-ROI CLIP classification.

    Returns (verdict, detection_dicts).
    """
    h, w = image.shape[:2]

    detections = detector.detect(
        image,
        conf=settings.YOLO_DEFECT_CONFIDENCE,
        detection_type="defect",
    )

    detection_dicts: list[dict[str, Any]] = []
    for det in detections:
        d = det.to_dict()

        # Per-ROI CLIP enrichment
        px1 = max(0, int(det.bbox_x1 * w))
        py1 = max(0, int(det.bbox_y1 * h))
        px2 = min(w, int(det.bbox_x2 * w))
        py2 = min(h, int(det.bbox_y2 * h))
        roi = image[py1:py2, px1:px2]

        if roi.size > 0 and clip_classifier is not None and clip_classifier.is_loaded:
            roi_clip = clip_classifier.classify(roi, threshold=settings.CLIP_DEFECT_THRESHOLD)
            d["clip_label"] = roi_clip.label
            d["clip_score"] = round(roi_clip.score, 4)
            d["is_defect"] = roi_clip.is_defect
        else:
            d["clip_label"] = None
            d["clip_score"] = None
            d["is_defect"] = True  # Fail-safe: YOLO detected it, assume defect

        detection_dicts.append(d)

    defect_count = sum(1 for d in detection_dicts if d.get("is_defect"))
    verdict = "NG" if defect_count > 0 else "OK"
    return verdict, detection_dicts


def _run_cnn(
    image: np.ndarray,
    resnet_classifier,
) -> tuple[str, float, list[dict[str, Any]]]:
    """ResNet full-image classification.

    Returns (verdict, confidence, detection_dicts).
    """
    result = resnet_classifier.classify(image, threshold=settings.CNN_RESNET_THRESHOLD)
    verdict = result.label
    detection_dicts = [{
        "defect_class": f"cnn_{result.label.lower()}",
        "confidence": result.confidence,
        "bbox_x1": 0.0,
        "bbox_y1": 0.0,
        "bbox_x2": 1.0,
        "bbox_y2": 1.0,
        "detection_type": "defect" if result.is_defect else "object",
        "clip_label": f"ResNet: {result.label} ({result.confidence:.1%})",
        "clip_score": result.confidence,
        "is_defect": result.is_defect,
    }]
    return verdict, result.confidence, detection_dicts


# ── Main entry point ─────────────────────────────────────────────────────────


async def run_inspection(
    image: np.ndarray,
    image_path: str,
    db: AsyncSession,
    *,
    detector=None,
    clip_classifier=None,
    resnet_classifier=None,
    pipeline: PipelineMode = "yolo_clip",
    tenant_id: uuid.UUID | None = None,
) -> InspectionResult:
    """Execute an inspection using the specified pipeline.

    Args:
        pipeline: One of "yolo_clip", "cnn", or "ensemble".
            - yolo_clip: YOLO detection + CLIP per-ROI classification.
            - cnn: ResNet full-image binary classification.
            - ensemble: Both pipelines; any NG → final NG (fail-safe).
    """
    start = time.perf_counter()
    inspection_id = uuid.uuid4()

    verdict = "OK"
    all_detections: list[dict[str, Any]] = []
    cnn_confidence: float | None = None

    # ── YOLO + CLIP stage ─────────────────────────────────────────────
    if pipeline in ("yolo_clip", "ensemble"):
        if detector is not None and detector.is_loaded:
            yolo_verdict, yolo_dets = _run_yolo_clip(image, detector, clip_classifier)
            all_detections.extend(yolo_dets)
            if yolo_verdict == "NG":
                verdict = "NG"
        else:
            logger.warning("YOLO detector not available — skipping detection stage")

    # ── CNN stage ─────────────────────────────────────────────────────
    if pipeline in ("cnn", "ensemble"):
        if resnet_classifier is not None and resnet_classifier.is_loaded:
            cnn_verdict, cnn_conf, cnn_dets = _run_cnn(image, resnet_classifier)
            cnn_confidence = cnn_conf
            if pipeline == "cnn":
                # CNN-only: use CNN detections and verdict directly
                all_detections = cnn_dets
                verdict = cnn_verdict
            else:
                # Ensemble: CNN NG overrides OK from YOLO
                all_detections.extend(cnn_dets)
                if cnn_verdict == "NG":
                    verdict = "NG"
        else:
            logger.warning("ResNet classifier not available — skipping CNN stage")

    defect_count = sum(1 for d in all_detections if d.get("is_defect"))
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    # ── Persist to DB ─────────────────────────────────────────────────
    inspection = Inspection(
        id=inspection_id,
        tenant_id=tenant_id,
        image_path=image_path,
        verdict=verdict,
        total_defects=defect_count,
        processing_ms=elapsed_ms,
        pipeline=pipeline,
    )
    db.add(inspection)

    for d in all_detections:
        defect = Defect(
            id=uuid.uuid4(),
            inspection_id=inspection_id,
            defect_class=d.get("defect_class", "unknown"),
            confidence=d.get("confidence", 0.0),
            bbox_x1=d.get("bbox_x1", 0.0),
            bbox_y1=d.get("bbox_y1", 0.0),
            bbox_x2=d.get("bbox_x2", 1.0),
            bbox_y2=d.get("bbox_y2", 1.0),
            detection_type=d.get("detection_type", "defect"),
            clip_label=d.get("clip_label", ""),
            clip_score=d.get("clip_score"),
            is_defect=d.get("is_defect", False),
        )
        db.add(defect)

    await db.flush()

    logger.info(
        "Inspection %s [%s] completed in %.1fms — verdict=%s defects=%d",
        inspection_id, pipeline, elapsed_ms, verdict, defect_count,
    )

    return InspectionResult(
        inspection_id=inspection_id,
        verdict=verdict,
        total_defects=defect_count,
        processing_ms=elapsed_ms,
        pipeline=pipeline,
        detections=all_detections,
        image_path=image_path,
        cnn_confidence=cnn_confidence,
    )


# ── Utility ──────────────────────────────────────────────────────────────────


def save_upload(file_bytes: bytes, filename: str) -> str:
    """Save uploaded bytes to disk and return the path."""
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    dest = upload_dir / unique_name
    dest.write_bytes(file_bytes)
    return str(dest)


def load_image(path: str) -> np.ndarray:
    """Load an image from disk as a BGR numpy array."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image at {path}")
    return img
