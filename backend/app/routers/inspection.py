"""Inspection router – upload images and retrieve results."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import delete, select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db
from app.models.db_models import Defect, Inspection
from app.routers.auth import verify_api_key
from app.services.color_service import ColorInspector
from app.services.inspection_service import (
    InspectionResult,
    load_image,
    run_inspection,
    save_upload,
)
from app.services.vlm_service import VLMService

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
    delta_e: Optional[float] = None
    vlm_description: Optional[str] = None


class InspectionSchema(BaseModel):
    id: str
    tenant_id: Optional[str] = None
    timestamp: datetime
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    verdict: str
    total_defects: int
    processing_ms: Optional[float] = None
    operator_review: bool
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
    color_result: Optional[dict[str, Any]] = None
    vlm_response: Optional[str] = None


class AskResponse(BaseModel):
    response: str
    image_url: Optional[str] = None
    processing_ms: float


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
        operator_review=insp.operator_review,
        defects=[
            DefectSchema(
                id=str(d.id),
                defect_class=d.defect_class,
                confidence=d.confidence,
                bbox_x1=d.bbox_x1,
                bbox_y1=d.bbox_y1,
                bbox_x2=d.bbox_x2,
                bbox_y2=d.bbox_y2,
                delta_e=d.delta_e,
                vlm_description=d.vlm_description,
            )
            for d in (insp.defects or [])
        ],
    )


# ── Routes ───────────────────────────────────────────────────────────────────


@router.post("/inspect", response_model=InspectResponse, status_code=status.HTTP_201_CREATED)
async def inspect_image(
    file: UploadFile = File(..., description="Image to inspect"),
    enable_vlm: bool = Query(True, description="Enable VLM defect explanations"),
    prompt: Optional[str] = Query(None, description="Custom prompt for VLM analysis"),
    tenant_id: Optional[str] = Query(None, description="Tenant UUID"),
    db: AsyncSession = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> InspectResponse:
    """Upload an image for AI-powered quality inspection.

    The pipeline runs YOLO detection, optional colour checking, and
    optional VLM explanation before persisting results.
    If a custom prompt is provided, the VLM will also answer it.
    """
    # Read and save the uploaded file
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file upload")

    filename = file.filename or "image.jpg"
    image_path = save_upload(contents, filename)

    try:
        image = load_image(image_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Resolve services from app state (set during lifespan)
    from app.main import app_state

    detector = app_state["detector"]
    vlm_service = VLMService()
    color_inspector = ColorInspector()

    tid = uuid.UUID(tenant_id) if tenant_id else None

    result: InspectionResult = await run_inspection(
        image=image,
        image_path=image_path,
        detector=detector,
        vlm_service=vlm_service,
        color_inspector=color_inspector,
        db=db,
        tenant_id=tid,
        enable_vlm=enable_vlm,
    )

    # Run custom prompt through VLM if provided
    vlm_response: str | None = None
    if prompt and prompt.strip():
        vlm_response = await vlm_service.ask(image_path, prompt.strip())

    # Broadcast via WebSocket
    try:
        from app.main import ws_manager
        await ws_manager.broadcast(result.to_dict())
    except Exception:
        logger.debug("WebSocket broadcast skipped (no active connections)")

    response_data = result.to_dict()
    response_data["image_url"] = _make_image_url(response_data.get("image_path"))
    response_data["vlm_response"] = vlm_response
    return InspectResponse(**response_data)


@router.post("/ask", response_model=AskResponse)
async def ask_about_image(
    file: UploadFile = File(..., description="Image to analyze"),
    prompt: str = Query(..., description="Your question about the image"),
    _key: str = Depends(verify_api_key),
) -> AskResponse:
    """Ask a free-form question about an uploaded image using the VLM."""
    import time

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file upload")

    filename = file.filename or "image.jpg"
    image_path = save_upload(contents, filename)

    start = time.perf_counter()
    vlm_service = VLMService()
    response_text = await vlm_service.ask(image_path, prompt.strip())
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return AskResponse(
        response=response_text,
        image_url=_make_image_url(image_path),
        processing_ms=round(elapsed_ms, 2),
    )


@router.get("/inspections", response_model=InspectionListResponse)
async def list_inspections(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    verdict: Optional[str] = Query(None, regex="^(PASS|FAIL)$"),
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


@router.delete("/inspections", status_code=status.HTTP_200_OK)
async def delete_all_inspections(
    db: AsyncSession = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> dict:
    """Delete all inspections and their defects (demo/reset use)."""
    # Defects cascade-deleted via FK ON DELETE CASCADE
    await db.execute(delete(Defect))
    result = await db.execute(delete(Inspection))
    await db.commit()
    return {"deleted": result.rowcount}
