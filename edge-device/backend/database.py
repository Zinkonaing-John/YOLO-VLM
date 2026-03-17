"""SQLite database for edge-device inspection storage.

Uses aiosqlite for async access — no PostgreSQL dependency needed on Jetson.
"""

from __future__ import annotations

import aiosqlite
import logging

from config import get_settings

logger = logging.getLogger(__name__)

_db: aiosqlite.Connection | None = None

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS inspections (
    id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    image_path TEXT,
    verdict TEXT NOT NULL,
    defect_count INTEGER DEFAULT 0,
    total_detections INTEGER DEFAULT 0,
    pipeline_ms REAL DEFAULT 0,
    uploaded_to_cloud INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS detections (
    id TEXT PRIMARY KEY,
    inspection_id TEXT NOT NULL,
    class_name TEXT,
    det_confidence REAL,
    roi_verdict TEXT,
    roi_confidence REAL,
    bbox_x1 REAL,
    bbox_y1 REAL,
    bbox_x2 REAL,
    bbox_y2 REAL,
    FOREIGN KEY (inspection_id) REFERENCES inspections(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_inspections_timestamp ON inspections(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_inspections_verdict ON inspections(verdict);
CREATE INDEX IF NOT EXISTS idx_detections_inspection ON detections(inspection_id);
"""


async def get_db() -> aiosqlite.Connection:
    """Return the shared database connection, creating it if needed."""
    global _db
    if _db is None:
        settings = get_settings()
        _db = await aiosqlite.connect(settings.DATABASE_PATH)
        _db.row_factory = aiosqlite.Row
        await _db.execute("PRAGMA journal_mode=WAL")
        await _db.execute("PRAGMA foreign_keys=ON")
        await _db.executescript(CREATE_TABLES_SQL)
        await _db.commit()
        logger.info("SQLite database initialized at %s", settings.DATABASE_PATH)
    return _db


async def close_db():
    """Close the database connection."""
    global _db
    if _db is not None:
        await _db.close()
        _db = None
        logger.info("SQLite database closed")
