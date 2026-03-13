"""SQLAlchemy ORM models for inspections and defects."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class Inspection(Base):
    """A single inspection run against an uploaded image."""

    __tablename__ = "inspections"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )
    tenant_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    timestamp = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )
    image_path = Column(Text, nullable=True)
    verdict = Column(
        String(4),
        CheckConstraint("verdict IN ('OK','NG')"),
        nullable=False,
    )
    total_defects = Column(Integer, default=0, nullable=False)
    processing_ms = Column(Float, nullable=True)

    # Relationships
    defects = relationship(
        "Defect",
        back_populates="inspection",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<Inspection {self.id} verdict={self.verdict}>"


class Defect(Base):
    """An individual defect detected within an inspection."""

    __tablename__ = "defects"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    inspection_id = Column(
        UUID(as_uuid=True),
        ForeignKey("inspections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    defect_class = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)
    clip_label = Column(String(100), nullable=True)
    clip_score = Column(Float, nullable=True)
    is_defect = Column(Boolean, default=False, nullable=False)

    # Relationships
    inspection = relationship("Inspection", back_populates="defects")

    def __repr__(self) -> str:
        return f"<Defect {self.defect_class} clip={self.clip_label} is_defect={self.is_defect}>"
