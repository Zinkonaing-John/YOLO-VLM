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
from app.models.ai_models import DetectionResult, YOLODetector
from app.models.db_models import Defect, Inspection
from app.services.color_service import ColorInspector
from app.services.vlm_service import VLMService

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
    color_result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "inspection_id": str(self.inspection_id),
            "verdict": self.verdict,
            "total_defects": self.total_defects,
            "processing_ms": round(self.processing_ms, 2),
            "detections": self.detections,
            "image_path": self.image_path,
            "color_result": self.color_result,
        }


async def run_inspection(
    image: np.ndarray,
    image_path: str,
    detector: YOLODetector,
    vlm_service: VLMService,
    color_inspector: ColorInspector,
    db: AsyncSession,
    tenant_id: uuid.UUID | None = None,
    reference_image: np.ndarray | None = None,
    enable_vlm: bool = True,
) -> InspectionResult:
    """Execute the full inspection pipeline.

    Pipeline stages:
      1. YOLO defect detection
      2. Color inspection (if reference image provided)
      3. VLM defect explanation (if enabled)
      4. Persist results to database

    Args:
        image: BGR numpy array of the image under inspection.
        image_path: Filesystem path where the image has been saved.
        detector: Loaded YOLODetector instance.
        vlm_service: VLMService for generating explanations.
        color_inspector: ColorInspector for Delta-E checks.
        db: Async SQLAlchemy session.
        tenant_id: Optional tenant UUID for multi-tenancy.
        reference_image: Optional golden-reference image for color comparison.
        enable_vlm: Whether to call the VLM for each defect.

    Returns:
        ``InspectionResult`` with full details.
    """
    start = time.perf_counter()
    inspection_id = uuid.uuid4()

    # ── 1. YOLO Detection ────────────────────────────────────────────────
    detections: list[DetectionResult] = detector.detect(
        image,
        conf=settings.YOLO_CONFIDENCE,
    )
    total_defects = len(detections)
    verdict = "FAIL" if total_defects > 0 else "PASS"

    # ── 2. Color Inspection ──────────────────────────────────────────────
    color_result_dict: dict[str, Any] | None = None
    if reference_image is not None:
        try:
            color_result = color_inspector.check_color(image, reference_image)
            color_result_dict = color_result.to_dict()
            if color_result.verdict == "FAIL":
                verdict = "FAIL"
        except Exception:
            logger.exception("Color inspection failed")

    # ── 3. VLM Explanation (per defect) ──────────────────────────────────
    detection_dicts: list[dict[str, Any]] = []
    for det in detections:
        d = det.to_dict()
        if enable_vlm and settings.VLM_ENABLED:
            try:
                explanation = await vlm_service.explain_defect(
                    image_path=image_path,
                    defect_type=det.defect_class,
                )
                d["vlm_description"] = explanation
            except Exception:
                logger.exception("VLM explanation failed for %s", det.defect_class)
                d["vlm_description"] = ""
        else:
            d["vlm_description"] = ""
        detection_dicts.append(d)

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    # ── 4. Persist to Database ───────────────────────────────────────────
    inspection = Inspection(
        id=inspection_id,
        tenant_id=tenant_id,
        image_path=image_path,
        verdict=verdict,
        total_defects=total_defects,
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
            delta_e=color_result_dict["delta_e"] if color_result_dict else None,
            vlm_description=d.get("vlm_description", ""),
        )
        db.add(defect)

    await db.flush()

    logger.info(
        "Inspection %s completed in %.1fms – verdict=%s defects=%d",
        inspection_id,
        elapsed_ms,
        verdict,
        total_defects,
    )

    return InspectionResult(
        inspection_id=inspection_id,
        verdict=verdict,
        total_defects=total_defects,
        processing_ms=elapsed_ms,
        detections=detection_dicts,
        image_path=image_path,
        color_result=color_result_dict,
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
