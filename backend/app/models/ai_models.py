"""AI model wrappers for YOLO object detection and anomaly detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Single detection bounding-box result."""
    defect_class: str
    confidence: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "defect_class": self.defect_class,
            "confidence": round(self.confidence, 4),
            "bbox_x1": round(self.bbox_x1, 2),
            "bbox_y1": round(self.bbox_y1, 2),
            "bbox_x2": round(self.bbox_x2, 2),
            "bbox_y2": round(self.bbox_y2, 2),
        }


class YOLODetector:
    """Wrapper around Ultralytics YOLO for industrial defect detection."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._weights_path: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load_model(self, weights_path: str) -> None:
        """Load YOLO weights from disk.

        Args:
            weights_path: Path to a .pt weights file.
        """
        path = Path(weights_path)
        if not path.exists():
            logger.warning(
                "YOLO weights not found at %s – detector will not run until "
                "valid weights are provided.",
                weights_path,
            )
            return

        try:
            from ultralytics import YOLO  # lazy import to speed up startup

            self._model = YOLO(str(path))
            self._weights_path = str(path)
            logger.info("YOLO model loaded from %s", weights_path)
        except Exception:
            logger.exception("Failed to load YOLO model from %s", weights_path)

    def detect(
        self,
        image: np.ndarray,
        conf: float = 0.5,
        iou: float = 0.45,
    ) -> list[DetectionResult]:
        """Run inference on a BGR numpy image.

        Args:
            image: HxWxC numpy array (BGR, as read by OpenCV).
            conf: Minimum confidence threshold.
            iou: NMS IoU threshold.

        Returns:
            List of ``DetectionResult`` objects.
        """
        if self._model is None:
            logger.error("YOLO model is not loaded – returning empty results.")
            return []

        results = self._model.predict(
            source=image,
            conf=conf,
            iou=iou,
            verbose=False,
        )

        detections: list[DetectionResult] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = result.names.get(cls_id, f"class_{cls_id}")
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    DetectionResult(
                        defect_class=cls_name,
                        confidence=confidence,
                        bbox_x1=x1,
                        bbox_y1=y1,
                        bbox_x2=x2,
                        bbox_y2=y2,
                    )
                )

        logger.info(
            "YOLO detected %d defect(s) at conf>=%.2f",
            len(detections),
            conf,
        )
        return detections


@dataclass
class AnomalyScore:
    """Result from the anomaly detector."""
    score: float
    is_anomalous: bool
    heatmap: Optional[np.ndarray] = field(default=None, repr=False)


class AnomalyDetector:
    """Stub for Anomalib-based unsupervised anomaly detection.

    Replace the internals with a real Anomalib model (e.g. PaDiM,
    PatchCore) when ready.
    """

    def __init__(self) -> None:
        self._model: Any | None = None
        self._threshold: float = 0.5

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load_model(self, model_path: str, threshold: float = 0.5) -> None:
        """Load an Anomalib exported model.

        Args:
            model_path: Path to the exported anomalib model (.onnx / .pt).
            threshold: Anomaly score threshold.
        """
        path = Path(model_path)
        if not path.exists():
            logger.warning(
                "Anomalib model not found at %s – anomaly detection disabled.",
                model_path,
            )
            return

        try:
            # Placeholder: in production, load the actual anomalib inference engine
            # from anomalib.deploy import OpenVINOInferencer
            # self._model = OpenVINOInferencer(path=model_path)
            self._threshold = threshold
            logger.info("Anomalib model loaded from %s", model_path)
        except Exception:
            logger.exception("Failed to load anomalib model from %s", model_path)

    def predict(self, image: np.ndarray) -> AnomalyScore:
        """Run anomaly prediction on an image.

        Args:
            image: HxWxC numpy array (BGR).

        Returns:
            ``AnomalyScore`` with score, boolean flag, and optional heatmap.
        """
        if self._model is None:
            logger.warning(
                "Anomaly model not loaded – returning default non-anomalous score."
            )
            return AnomalyScore(score=0.0, is_anomalous=False)

        # Placeholder inference – replace with real anomalib call
        # prediction = self._model.predict(image)
        # score = prediction.pred_score
        score = 0.0
        return AnomalyScore(
            score=score,
            is_anomalous=score >= self._threshold,
            heatmap=None,
        )
