"""Two-stage inspection pipeline for edge devices.

Pipeline:
  Camera frame → YOLO-det (find defects) → crop ROIs → YOLO-cls (OK/NG) → verdict
"""

from __future__ import annotations

import base64
import logging
import time
import uuid
from typing import Any

import cv2
import numpy as np
import aiosqlite

from config import get_settings
from database import get_db
from models import Detection, YOLOClassifier, YOLODetector

logger = logging.getLogger(__name__)
settings = get_settings()


async def run_inspection(
    image: np.ndarray,
    image_path: str,
    detector: YOLODetector,
    classifier: YOLOClassifier | None,
) -> dict[str, Any]:
    """Execute the two-stage edge inspection pipeline.

    Returns a JSON-serializable result dict.
    """
    t_start = time.perf_counter()
    inspection_id = uuid.uuid4().hex

    h, w = image.shape[:2]

    # ── Stage 1: Detection ────────────────────────────────────────────
    detections = detector.detect(
        image,
        conf=settings.DET_CONFIDENCE,
        iou=settings.DET_IOU,
        img_size=settings.DET_IMG_SIZE,
    )

    # ── Stage 2: Per-ROI Classification ───────────────────────────────
    for det in detections:
        if classifier is not None and classifier.is_loaded:
            x1 = max(0, int(det.x1))
            y1 = max(0, int(det.y1))
            x2 = min(w, int(det.x2))
            y2 = min(h, int(det.y2))
            roi = image[y1:y2, x1:x2]

            if roi.shape[0] < settings.MIN_ROI_SIZE or roi.shape[1] < settings.MIN_ROI_SIZE:
                det.roi_verdict = "NG"
                det.roi_confidence = 0.0
            else:
                verdict, conf = classifier.classify(roi, settings.CLS_CONFIDENCE)
                det.roi_verdict = verdict
                det.roi_confidence = conf
        else:
            # No classifier — every detection is NG (fail-safe)
            det.roi_verdict = "NG"
            det.roi_confidence = det.confidence

    # ── Verdict (fail-safe toward NG) ─────────────────────────────────
    if not detections:
        verdict = "OK"
    elif any(d.roi_verdict == "NG" for d in detections):
        verdict = "NG"
    else:
        verdict = "OK"

    defect_count = sum(1 for d in detections if d.roi_verdict == "NG")
    pipeline_ms = (time.perf_counter() - t_start) * 1000.0

    # ── Save to local SQLite ──────────────────────────────────────────
    db = await get_db()
    await db.execute(
        """INSERT INTO inspections (id, timestamp, image_path, verdict, defect_count,
           total_detections, pipeline_ms) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (inspection_id, time.time(), image_path, verdict, defect_count,
         len(detections), pipeline_ms),
    )
    for det in detections:
        await db.execute(
            """INSERT INTO detections (id, inspection_id, class_name, det_confidence,
               roi_verdict, roi_confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (uuid.uuid4().hex, inspection_id, det.class_name, det.confidence,
             det.roi_verdict, det.roi_confidence,
             det.nx1, det.ny1, det.nx2, det.ny2),
        )
    await db.commit()

    result = {
        "inspection_id": inspection_id,
        "timestamp": time.time(),
        "verdict": verdict,
        "defect_count": defect_count,
        "total_detections": len(detections),
        "pipeline_ms": round(pipeline_ms, 2),
        "image_path": image_path,
        "detections": [d.to_dict() for d in detections],
    }

    logger.info(
        "Inspection %s: %s | %d defects | %.1fms",
        inspection_id[:8], verdict, defect_count, pipeline_ms,
    )
    return result


async def get_inspections(
    limit: int = 50,
    offset: int = 0,
    verdict: str | None = None,
) -> list[dict]:
    """Fetch inspection history from local SQLite."""
    db = await get_db()
    if verdict:
        cursor = await db.execute(
            "SELECT * FROM inspections WHERE verdict = ? ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (verdict, limit, offset),
        )
    else:
        cursor = await db.execute(
            "SELECT * FROM inspections ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
    rows = await cursor.fetchall()

    results = []
    for row in rows:
        inspection = dict(row)
        det_cursor = await db.execute(
            "SELECT * FROM detections WHERE inspection_id = ?", (row["id"],)
        )
        inspection["detections"] = [dict(d) for d in await det_cursor.fetchall()]
        results.append(inspection)
    return results


async def get_statistics() -> dict:
    """Compute inspection statistics from local SQLite."""
    db = await get_db()

    total = await (await db.execute("SELECT COUNT(*) FROM inspections")).fetchone()
    ok = await (await db.execute("SELECT COUNT(*) FROM inspections WHERE verdict='OK'")).fetchone()
    ng = await (await db.execute("SELECT COUNT(*) FROM inspections WHERE verdict='NG'")).fetchone()
    avg_ms = await (await db.execute("SELECT AVG(pipeline_ms) FROM inspections")).fetchone()

    # Defect class distribution
    classes = await (await db.execute(
        """SELECT class_name, COUNT(*) as count FROM detections
           WHERE roi_verdict='NG' GROUP BY class_name ORDER BY count DESC"""
    )).fetchall()

    # Hourly trend (last 24h)
    hourly = await (await db.execute(
        """SELECT
             CAST((timestamp / 3600) AS INTEGER) * 3600 AS hour_ts,
             verdict,
             COUNT(*) AS count
           FROM inspections
           WHERE timestamp > (strftime('%s', 'now') - 86400)
           GROUP BY hour_ts, verdict
           ORDER BY hour_ts"""
    )).fetchall()

    total_count = total[0] if total else 0
    ok_count = ok[0] if ok else 0
    ng_count = ng[0] if ng else 0

    return {
        "total": total_count,
        "ok": ok_count,
        "ng": ng_count,
        "ok_rate": round(ok_count / total_count * 100, 1) if total_count > 0 else 0,
        "ng_rate": round(ng_count / total_count * 100, 1) if total_count > 0 else 0,
        "avg_pipeline_ms": round(avg_ms[0], 1) if avg_ms and avg_ms[0] else 0,
        "defect_classes": [{"class_name": r[0], "count": r[1]} for r in classes],
        "hourly_trend": [{"hour_ts": r[0], "verdict": r[1], "count": r[2]} for r in hourly],
    }


async def cleanup_old_images():
    """Remove oldest images when storage exceeds MAX_STORED_IMAGES."""
    from pathlib import Path

    db = await get_db()
    count = await (await db.execute("SELECT COUNT(*) FROM inspections")).fetchone()
    if count[0] <= settings.MAX_STORED_IMAGES:
        return

    excess = count[0] - settings.MAX_STORED_IMAGES
    old_rows = await (await db.execute(
        "SELECT id, image_path FROM inspections ORDER BY timestamp ASC LIMIT ?",
        (excess,)
    )).fetchall()

    for row in old_rows:
        img_path = Path(row["image_path"])
        if img_path.exists():
            img_path.unlink()
        await db.execute("DELETE FROM inspections WHERE id = ?", (row["id"],))

    await db.commit()
    logger.info("Cleaned up %d old inspections", len(old_rows))
