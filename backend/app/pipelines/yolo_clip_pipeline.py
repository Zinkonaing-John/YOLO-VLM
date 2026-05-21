"""YOLO detection + CLIP per-ROI classification pipeline."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from app.core.config import get_settings
from app.pipelines.base_pipeline import BasePipeline, PipelineResult

logger = logging.getLogger(__name__)
settings = get_settings()


class YoloClipPipeline(BasePipeline):
    """Detect bounding boxes with YOLO, then classify each ROI with CLIP."""

    name = "yolo_clip"
    description = "YOLO defect detection + CLIP zero-shot ROI classification"

    def __init__(self) -> None:
        self._detector: Any = None
        self._clip: Any = None

    def load(self) -> None:
        from app.models_runtime.yolo_detector import YOLODetector
        from app.models_runtime.clip_classifier import CLIPClassifier

        self._detector = YOLODetector()
        self._detector.load_model(settings.YOLO_DEFECT_WEIGHTS_PATH)

        self._clip = CLIPClassifier()
        self._clip.load_model(
            model_name=settings.CLIP_MODEL_NAME,
            ok_labels=settings.CLIP_LABELS_OK,
            ng_labels=settings.CLIP_LABELS_NG,
        )
        logger.info("YoloClipPipeline loaded (yolo=%s clip=%s)",
                    self._detector.is_loaded, self._clip.is_loaded)

    @property
    def is_loaded(self) -> bool:
        return self._detector is not None and self._detector.is_loaded

    def run(self, image: np.ndarray, context: dict[str, Any]) -> PipelineResult:
        t0 = time.perf_counter()
        threshold = context.get("threshold", settings.CLIP_DEFECT_THRESHOLD)
        h, w = image.shape[:2]

        raw_dets = self._detector.detect(image, conf=settings.YOLO_DEFECT_CONFIDENCE, detection_type="defect") \
            if self._detector and self._detector.is_loaded else []

        detections: list[dict] = []
        defect_labels: list[str] = []

        # Confidence gate: high-confidence YOLO detections are trusted directly;
        # CLIP is only used to suppress borderline detections.
        HIGH_CONF = 0.5

        for det in raw_dets:
            entry = det.to_dict()
            if self._clip and self._clip.is_loaded:
                px1, py1 = max(0, int(det.bbox_x1 * w)), max(0, int(det.bbox_y1 * h))
                px2, py2 = min(w, int(det.bbox_x2 * w)), min(h, int(det.bbox_y2 * h))
                roi = image[py1:py2, px1:px2]
                if roi.size > 0:
                    clip_res = self._clip.classify(roi, threshold=threshold)
                    # High-confidence YOLO: trust the specialised detector.
                    # Low-confidence YOLO: let CLIP decide.
                    is_defect = clip_res.is_defect if det.confidence < HIGH_CONF else True
                    entry.update({
                        "defect_label": det.defect_class,
                        "clip_label": clip_res.label,
                        "clip_score": round(clip_res.score, 4),
                        "is_defect": is_defect,
                        "x1": det.bbox_x1, "y1": det.bbox_y1,
                        "x2": det.bbox_x2, "y2": det.bbox_y2,
                    })
                    if is_defect:
                        defect_labels.append(det.defect_class)
                    detections.append(entry)
                    continue

            entry.update({
                "defect_label": det.defect_class,
                "is_defect": True,
                "x1": det.bbox_x1, "y1": det.bbox_y1,
                "x2": det.bbox_x2, "y2": det.bbox_y2,
            })
            defect_labels.append(det.defect_class)
            detections.append(entry)

        defect_count = sum(1 for d in detections if d.get("is_defect"))
        verdict = "NG" if defect_count > 0 else "OK"
        elapsed = (time.perf_counter() - t0) * 1000

        return PipelineResult(
            verdict=verdict,
            overall_score=float(defect_count) / max(len(detections), 1) if detections else 0.0,
            threshold=threshold,
            detections=detections,
            anomaly_scores=[],
            labels=list(set(defect_labels)),
            processing_time_ms=elapsed,
        )
