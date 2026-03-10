"""Statistics router – aggregate pass/fail counts and daily trends."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import case, cast, func, select, Date
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.db_models import Defect, Inspection
from app.routers.auth import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/statistics", tags=["statistics"])


# ── Response schemas ─────────────────────────────────────────────────────────


class DefectTypeCount(BaseModel):
    defect_class: str
    count: int


class OverallStatistics(BaseModel):
    total_inspections: int
    pass_count: int
    fail_count: int
    pass_rate: float
    avg_processing_ms: Optional[float]
    defect_type_distribution: list[DefectTypeCount]


class DailyStatItem(BaseModel):
    date: date
    total: int
    pass_count: int
    fail_count: int
    pass_rate: float


class DailyStatisticsResponse(BaseModel):
    days: int
    items: list[DailyStatItem]


# ── Routes ───────────────────────────────────────────────────────────────────


@router.get("", response_model=OverallStatistics)
async def get_statistics(
    tenant_id: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> OverallStatistics:
    """Return overall pass/fail counts, defect-type distribution, etc."""

    base = select(Inspection)
    if tenant_id:
        base = base.where(Inspection.tenant_id == tenant_id)

    # Total / pass / fail
    agg_q = select(
        func.count(Inspection.id).label("total"),
        func.sum(case((Inspection.verdict == "PASS", 1), else_=0)).label("pass_count"),
        func.sum(case((Inspection.verdict == "FAIL", 1), else_=0)).label("fail_count"),
        func.avg(Inspection.processing_ms).label("avg_ms"),
    )
    if tenant_id:
        agg_q = agg_q.where(Inspection.tenant_id == tenant_id)

    row = (await db.execute(agg_q)).one()
    total = row.total or 0
    pass_count = row.pass_count or 0
    fail_count = row.fail_count or 0
    avg_ms = round(row.avg_ms, 2) if row.avg_ms else None
    pass_rate = (pass_count / total * 100.0) if total > 0 else 0.0

    # Defect type distribution
    defect_q = (
        select(
            Defect.defect_class,
            func.count(Defect.id).label("cnt"),
        )
        .group_by(Defect.defect_class)
        .order_by(func.count(Defect.id).desc())
    )
    if tenant_id:
        defect_q = defect_q.join(Inspection).where(Inspection.tenant_id == tenant_id)

    defect_rows = (await db.execute(defect_q)).all()
    distribution = [
        DefectTypeCount(defect_class=r.defect_class, count=r.cnt) for r in defect_rows
    ]

    return OverallStatistics(
        total_inspections=total,
        pass_count=pass_count,
        fail_count=fail_count,
        pass_rate=round(pass_rate, 2),
        avg_processing_ms=avg_ms,
        defect_type_distribution=distribution,
    )


@router.get("/daily", response_model=DailyStatisticsResponse)
async def get_daily_statistics(
    days: int = Query(30, ge=1, le=365),
    tenant_id: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> DailyStatisticsResponse:
    """Return daily pass/fail counts for charting."""

    since = datetime.now(timezone.utc) - timedelta(days=days)

    q = (
        select(
            cast(Inspection.timestamp, Date).label("day"),
            func.count(Inspection.id).label("total"),
            func.sum(case((Inspection.verdict == "PASS", 1), else_=0)).label("pass_count"),
            func.sum(case((Inspection.verdict == "FAIL", 1), else_=0)).label("fail_count"),
        )
        .where(Inspection.timestamp >= since)
        .group_by(cast(Inspection.timestamp, Date))
        .order_by(cast(Inspection.timestamp, Date))
    )

    if tenant_id:
        q = q.where(Inspection.tenant_id == tenant_id)

    rows = (await db.execute(q)).all()

    items = []
    for r in rows:
        total = r.total or 0
        pc = r.pass_count or 0
        fc = r.fail_count or 0
        items.append(
            DailyStatItem(
                date=r.day,
                total=total,
                pass_count=pc,
                fail_count=fc,
                pass_rate=round((pc / total * 100.0) if total > 0 else 0.0, 2),
            )
        )

    return DailyStatisticsResponse(days=days, items=items)
