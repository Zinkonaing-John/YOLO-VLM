"""Core inspection orchestration service."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.ai_models import CLIPClassifier, DetectionResult, YOLODetector
from app.models.db_models import Defect, Inspection

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class InspectionResult:
    """Data transfer object returned after an inspection completes."""
    inspection_id: uuid.UUID
    verdict: str
    total_defects: int
    processing_ms: float
    detections: list[dict[str, Any]] = field(default_factory=list)
    image_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "inspection_id": str(self.inspection_id),
            "verdict": self.verdict,
            "total_defects": self.total_defects,
            "processing_ms": round(self.processing_ms, 2),
            "detections": self.detections,
            "image_path": self.image_path,
        }


async def run_inspection(
    image: np.ndarray,
    image_path: str,
    detector: YOLODetector,
    clip_classifier: CLIPClassifier,
    db: AsyncSession,
    tenant_id: uuid.UUID | None = None,
) -> InspectionResult:
    """Execute the inspection pipeline.

    Pipeline stages:
      1. YOLO object detection
      2. Crop ROI for each detection
      3. CLIP defect classification on each ROI
      4. Determine OK / NG verdict
      5. Save to DB
    """
    start = time.perf_counter()
    inspection_id = uuid.uuid4()

    h, w = image.shape[:2]

    # ── 1. YOLO object detection (primary defect detector) ────────────
    detections: list[DetectionResult] = detector.detect(
        image,
        conf=settings.YOLO_CONFIDENCE,
    )

    # ── 2. Enrich YOLO detections with per-ROI CLIP scores ───────────
    detection_dicts: list[dict[str, Any]] = []

    for det in detections:
        d = det.to_dict()

        # Convert normalized coords back to pixel coords for ROI crop
        px1 = max(0, int(det.bbox_x1 * w))
        py1 = max(0, int(det.bbox_y1 * h))
        px2 = min(w, int(det.bbox_x2 * w))
        py2 = min(h, int(det.bbox_y2 * h))
        roi = image[py1:py2, px1:px2]

        if roi.size > 0 and clip_classifier.is_loaded:
            roi_clip = clip_classifier.classify(
                roi,
                threshold=settings.CLIP_DEFECT_THRESHOLD,
            )
            d["clip_label"] = roi_clip.label
            d["clip_score"] = round(roi_clip.score, 4)
            d["is_defect"] = roi_clip.is_defect
        else:
            d["clip_label"] = None
            d["clip_score"] = None
            # YOLO detection = defect by default (it was trained on defect classes)
            d["is_defect"] = True

        detection_dicts.append(d)

    # ── 3. OK / NG Verdict (driven by YOLO detections) ────────────────
    defect_count = len(detection_dicts)
    verdict = "NG" if defect_count > 0 else "OK"

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    # ── 4. Save to DB ────────────────────────────────────────────────────
    inspection = Inspection(
        id=inspection_id,
        tenant_id=tenant_id,
        image_path=image_path,
        verdict=verdict,
        total_defects=defect_count,
        processing_ms=elapsed_ms,
    )
    db.add(inspection)

    for d in detection_dicts:
        defect = Defect(
            id=uuid.uuid4(),
            inspection_id=inspection_id,
            defect_class=d["defect_class"],
            confidence=d["confidence"],
            bbox_x1=d["bbox_x1"],
            bbox_y1=d["bbox_y1"],
            bbox_x2=d["bbox_x2"],
            bbox_y2=d["bbox_y2"],
            clip_label=d.get("clip_label", ""),
            clip_score=d.get("clip_score"),
            is_defect=d.get("is_defect", True),
        )
        db.add(defect)

    await db.flush()

    logger.info(
        "Inspection %s completed in %.1fms – verdict=%s defects=%d",
        inspection_id,
        elapsed_ms,
        verdict,
        defect_count,
    )

    return InspectionResult(
        inspection_id=inspection_id,
        verdict=verdict,
        total_defects=defect_count,
        processing_ms=elapsed_ms,
        detections=detection_dicts,
        image_path=image_path,
    )


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
