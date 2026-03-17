"""Inspection API — unified endpoints for all pipeline modes."""

from __future__ import annotations

import base64
import logging
import uuid
from typing import Any

import cv2
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import get_settings
from app.core.database import get_session
from app.core.model_registry import get_registry
from app.models.db_models import Defect, Inspection
from app.services.inspection_service import (
    PipelineMode,
    load_image,
    run_inspection,
    save_upload,
)

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(tags=["inspection"])


# ── POST /inspect ────────────────────────────────────────────────────────────


@router.post("/inspect")
async def inspect_image(
    file: UploadFile = File(...),
    pipeline: PipelineMode = Form("yolo_clip"),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Upload an image and run the specified inspection pipeline.

    Pipeline modes:
      - ``yolo_clip``: YOLO detection + CLIP per-ROI classification
      - ``cnn``: ResNet full-image binary OK/NG classification
      - ``ensemble``: Both pipelines vote; any NG -> final NG
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    image_path = save_upload(contents, file.filename or "upload.jpg")
    image = load_image(image_path)

    registry = get_registry()

    result = await run_inspection(
        image=image,
        image_path=image_path,
        db=db,
        detector=registry.yolo_defect,
        clip_classifier=registry.clip,
        resnet_classifier=registry.resnet,
        pipeline=pipeline,
    )

    # Broadcast via WebSocket
    from app.main import ws_manager
    await ws_manager.broadcast(result.to_dict())

    await db.commit()
    return result.to_dict()


# ── GET /inspections ─────────────────────────────────────────────────────────


@router.get("/inspections")
async def list_inspections(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    verdict: str | None = Query(None),
    pipeline: str | None = Query(None),
    defect_class: str | None = Query(None),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Paginated inspection history with filtering."""
    query = select(Inspection).options(selectinload(Inspection.defects))

    if verdict:
        query = query.where(Inspection.verdict == verdict.upper())
    if pipeline:
        query = query.where(Inspection.pipeline == pipeline)
    if defect_class:
        query = query.join(Inspection.defects).where(Defect.defect_class == defect_class)

    count_query = select(func.count(Inspection.id))
    if verdict:
        count_query = count_query.where(Inspection.verdict == verdict.upper())
    if pipeline:
        count_query = count_query.where(Inspection.pipeline == pipeline)
    total = (await db.execute(count_query)).scalar() or 0

    query = query.order_by(Inspection.timestamp.desc()).offset(offset).limit(limit)
    rows = (await db.execute(query)).unique().scalars().all()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "inspections": [_serialize_inspection(r) for r in rows],
    }


# ── GET /inspections/{id} ───────────────────────────────────────────────────


@router.get("/inspections/{inspection_id}")
async def get_inspection(
    inspection_id: uuid.UUID,
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    query = (
        select(Inspection)
        .options(selectinload(Inspection.defects))
        .where(Inspection.id == inspection_id)
    )
    row = (await db.execute(query)).unique().scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Inspection not found")
    return _serialize_inspection(row)


# ── GET /inspections/{id}/gradcam ────────────────────────────────────────────


@router.get("/inspections/{inspection_id}/gradcam")
async def get_gradcam(
    inspection_id: uuid.UUID,
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Generate GradCAM heatmap from ResNet classifier (on-demand)."""
    row = (await db.execute(
        select(Inspection).where(Inspection.id == inspection_id)
    )).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Inspection not found")

    registry = get_registry()
    if registry.resnet is None or not registry.resnet.is_loaded:
        raise HTTPException(status_code=503, detail="ResNet classifier not loaded")

    image = load_image(row.image_path)
    heatmap = registry.resnet.get_cam_heatmap(image)
    if heatmap is None:
        raise HTTPException(status_code=500, detail="GradCAM generation failed")

    _, heatmap_png = cv2.imencode(".png", heatmap)
    return {
        "inspection_id": str(inspection_id),
        "gradcam_base64": base64.b64encode(heatmap_png.tobytes()).decode(),
    }


# ── GET /inspections/{id}/segmentation ───────────────────────────────────────


@router.get("/inspections/{inspection_id}/segmentation")
async def get_segmentation(
    inspection_id: uuid.UUID,
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Run YOLOv8-seg and return mask overlay (on-demand)."""
    row = (await db.execute(
        select(Inspection).where(Inspection.id == inspection_id)
    )).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Inspection not found")

    registry = get_registry()
    if registry.yolo_seg is None or not registry.yolo_seg.is_loaded:
        raise HTTPException(status_code=503, detail="YOLO segmentor not loaded")

    image = load_image(row.image_path)
    segments = registry.yolo_seg.segment(image, conf=settings.YOLO_SEG_CONFIDENCE)
    mask_rgba = registry.yolo_seg.render_mask(image, segments)
    _, mask_png = cv2.imencode(".png", mask_rgba)

    return {
        "inspection_id": str(inspection_id),
        "segments": [s.to_dict() for s in segments],
        "mask_base64": base64.b64encode(mask_png.tobytes()).decode(),
    }


# ── DELETE /inspections ──────────────────────────────────────────────────────


@router.delete("/inspections")
async def delete_inspections(
    db: AsyncSession = Depends(get_session),
) -> dict[str, str]:
    await db.execute(delete(Defect))
    await db.execute(delete(Inspection))
    await db.commit()
    return {"status": "deleted"}


# ── GET /defect-classes ──────────────────────────────────────────────────────


@router.get("/defect-classes")
async def list_defect_classes(
    db: AsyncSession = Depends(get_session),
) -> list[str]:
    result = await db.execute(
        select(Defect.defect_class).where(Defect.is_defect.is_(True)).distinct()
    )
    return [row[0] for row in result.all() if row[0]]


# ── Serialization ────────────────────────────────────────────────────────────


def _serialize_inspection(row: Inspection) -> dict[str, Any]:
    return {
        "id": str(row.id),
        "timestamp": row.timestamp.isoformat() if row.timestamp else None,
        "image_path": row.image_path,
        "verdict": row.verdict,
        "total_defects": row.total_defects,
        "processing_ms": round(row.processing_ms, 2) if row.processing_ms else 0,
        "pipeline": row.pipeline,
        "defects": [
            {
                "id": str(d.id),
                "defect_class": d.defect_class,
                "confidence": round(d.confidence, 4) if d.confidence else 0,
                "bbox_x1": d.bbox_x1,
                "bbox_y1": d.bbox_y1,
                "bbox_x2": d.bbox_x2,
                "bbox_y2": d.bbox_y2,
                "detection_type": d.detection_type,
                "clip_label": d.clip_label,
                "clip_score": round(d.clip_score, 4) if d.clip_score else None,
                "is_defect": d.is_defect,
            }
            for d in (row.defects or [])
        ],
    }
