"""Inspection router – upload images and retrieve results."""

from __future__ import annotations

import io
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import delete, select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db
from app.models.db_models import Defect, Inspection
from app.routers.auth import verify_api_key
from app.services.inspection_service import (
    InspectionResult,
    load_image,
    run_inspection,
    save_upload,
)

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(tags=["inspection"])


# ── Pydantic response schemas ───────────────────────────────────────────────


class DefectSchema(BaseModel):
    id: str
    defect_class: str
    confidence: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    clip_label: Optional[str] = None
    clip_score: Optional[float] = None
    is_defect: bool = False


class InspectionSchema(BaseModel):
    id: str
    tenant_id: Optional[str] = None
    timestamp: datetime
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    verdict: str
    total_defects: int
    processing_ms: Optional[float] = None
    defects: list[DefectSchema] = []


class InspectionListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[InspectionSchema]


class InspectResponse(BaseModel):
    inspection_id: str
    verdict: str
    total_defects: int
    processing_ms: float
    detections: list[dict[str, Any]]
    image_path: str
    image_url: Optional[str] = None


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_image_url(image_path: str | None) -> str | None:
    """Convert a filesystem path like 'uploads/abc_img.jpg' to '/uploads/abc_img.jpg'."""
    if not image_path:
        return None
    filename = Path(image_path).name
    return f"/uploads/{filename}"


def _inspection_to_schema(insp: Inspection) -> InspectionSchema:
    return InspectionSchema(
        id=str(insp.id),
        tenant_id=str(insp.tenant_id) if insp.tenant_id else None,
        timestamp=insp.timestamp or datetime.now(timezone.utc),
        image_path=insp.image_path,
        image_url=_make_image_url(insp.image_path),
        verdict=insp.verdict,
        total_defects=insp.total_defects,
        processing_ms=insp.processing_ms,
        defects=[
            DefectSchema(
                id=str(d.id),
                defect_class=d.defect_class,
                confidence=d.confidence,
                bbox_x1=d.bbox_x1,
                bbox_y1=d.bbox_y1,
                bbox_x2=d.bbox_x2,
                bbox_y2=d.bbox_y2,
                clip_label=d.clip_label,
                clip_score=d.clip_score,
                is_defect=d.is_defect,
            )
            for d in (insp.defects or [])
        ],
    )


# ── Routes ───────────────────────────────────────────────────────────────────


@router.post("/inspect", response_model=InspectResponse, status_code=status.HTTP_201_CREATED)
async def inspect_image(
    file: UploadFile = File(..., description="Image to inspect"),
    tenant_id: Optional[str] = Query(None, description="Tenant UUID"),
    db: AsyncSession = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> InspectResponse:
    """Upload an image for AI-powered quality inspection.

    Pipeline: YOLO detect → Crop ROI → CLIP classify → OK/NG → Save to DB.
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file upload")

    filename = file.filename or "image.jpg"
    image_path = save_upload(contents, filename)

    try:
        image = load_image(image_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    from app.main import app_state

    detector = app_state["detector"]
    clip_classifier = app_state["clip_classifier"]

    tid = uuid.UUID(tenant_id) if tenant_id else None

    result: InspectionResult = await run_inspection(
        image=image,
        image_path=image_path,
        detector=detector,
        clip_classifier=clip_classifier,
        db=db,
        tenant_id=tid,
    )

    # Broadcast via WebSocket
    try:
        from app.main import ws_manager
        await ws_manager.broadcast(result.to_dict())
    except Exception:
        logger.debug("WebSocket broadcast skipped (no active connections)")

    response_data = result.to_dict()
    response_data["image_url"] = _make_image_url(response_data.get("image_path"))
    return InspectResponse(**response_data)


@router.get("/inspections", response_model=InspectionListResponse)
async def list_inspections(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    verdict: Optional[str] = Query(None, regex="^(OK|NG)$"),
    db: AsyncSession = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> InspectionListResponse:
    """List recent inspections with pagination and optional verdict filter."""
    query = select(Inspection).order_by(Inspection.timestamp.desc())
    count_query = select(func.count(Inspection.id))

    if verdict:
        query = query.where(Inspection.verdict == verdict)
        count_query = count_query.where(Inspection.verdict == verdict)

    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)
    result = await db.execute(query)
    inspections = result.scalars().all()

    return InspectionListResponse(
        total=total,
        page=page,
        page_size=page_size,
        items=[_inspection_to_schema(i) for i in inspections],
    )


@router.get("/inspections/{inspection_id}", response_model=InspectionSchema)
async def get_inspection(
    inspection_id: str,
    db: AsyncSession = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> InspectionSchema:
    """Get a single inspection by ID."""
    try:
        uid = uuid.UUID(inspection_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID format")

    result = await db.execute(
        select(Inspection).where(Inspection.id == uid)
    )
    inspection = result.scalar_one_or_none()
    if inspection is None:
        raise HTTPException(status_code=404, detail="Inspection not found")

    return _inspection_to_schema(inspection)


@router.get("/inspections/{inspection_id}/heatmap")
async def get_heatmap(
    inspection_id: str,
    db: AsyncSession = Depends(get_db),
    _key: str = Depends(verify_api_key),
):
    """Generate and return an anomaly heatmap image for this inspection."""
    try:
        uid = uuid.UUID(inspection_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID format")

    result = await db.execute(select(Inspection).where(Inspection.id == uid))
    inspection = result.scalar_one_or_none()
    if inspection is None:
        raise HTTPException(status_code=404, detail="Inspection not found")

    if not inspection.image_path:
        raise HTTPException(status_code=404, detail="No image stored for this inspection")

    from app.main import app_state
    simplenet = app_state.get("simplenet")
    if simplenet is None:
        raise HTTPException(status_code=503, detail="SimpleNet model not loaded")

    image = load_image(inspection.image_path)
    anomaly = simplenet.predict(image)

    # Encode heatmap as PNG
    _, buf = cv2.imencode(".png", anomaly.heatmap)
    return StreamingResponse(
        io.BytesIO(buf.tobytes()),
        media_type="image/png",
        headers={"X-Anomaly-Score": str(round(anomaly.score, 4))},
    )


@router.get("/inspections/{inspection_id}/clip-details")
async def get_clip_details(
    inspection_id: str,
    db: AsyncSession = Depends(get_db),
    _key: str = Depends(verify_api_key),
):
    """Run CLIP classification on each detection ROI and return detailed results."""
    try:
        uid = uuid.UUID(inspection_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID format")

    result = await db.execute(select(Inspection).where(Inspection.id == uid))
    inspection = result.scalar_one_or_none()
    if inspection is None:
        raise HTTPException(status_code=404, detail="Inspection not found")

    if not inspection.image_path:
        raise HTTPException(status_code=404, detail="No image stored")

    from app.main import app_state
    clip_classifier = app_state.get("clip_classifier")
    if not clip_classifier or not clip_classifier.is_loaded:
        raise HTTPException(status_code=503, detail="CLIP model not loaded")

    image = load_image(inspection.image_path)
    h, w = image.shape[:2]

    details = []
    for defect in (inspection.defects or []):
        px1 = max(0, int(defect.bbox_x1 * w))
        py1 = max(0, int(defect.bbox_y1 * h))
        px2 = min(w, int(defect.bbox_x2 * w))
        py2 = min(h, int(defect.bbox_y2 * h))
        roi = image[py1:py2, px1:px2]

        if roi.size == 0:
            continue

        clip_result = clip_classifier.classify(roi)
        details.append({
            "defect_id": str(defect.id),
            "defect_class": defect.defect_class,
            "confidence": round(defect.confidence, 4),
            "bbox": [defect.bbox_x1, defect.bbox_y1, defect.bbox_x2, defect.bbox_y2],
            "clip_label": clip_result.label,
            "clip_score": round(clip_result.score, 4),
            "is_defect": clip_result.is_defect,
        })

    return {"inspection_id": inspection_id, "clip_details": details}


@router.delete("/inspections", status_code=status.HTTP_200_OK)
async def delete_all_inspections(
    db: AsyncSession = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> dict:
    """Delete all inspections and their defects (demo/reset use)."""
    await db.execute(delete(Defect))
    result = await db.execute(delete(Inspection))
    await db.commit()
    return {"deleted": result.rowcount}
