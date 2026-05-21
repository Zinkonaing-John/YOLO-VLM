"""ResNet CNN binary classification pipeline."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from app.core.config import get_settings
from app.pipelines.base_pipeline import BasePipeline, PipelineResult

logger = logging.getLogger(__name__)
settings = get_settings()


class CNNPipeline(BasePipeline):
    """ResNet-18 full-image binary OK/NG classification with GradCAM."""

    name = "cnn"
    description = "ResNet-18 full-image binary classification"

    def __init__(self) -> None:
        self._classifier: Any = None

    def load(self) -> None:
        from app.models_runtime.resnet_classifier import ResNetClassifier
        from pathlib import Path

        self._classifier = ResNetClassifier()
        weights = str(Path(__file__).resolve().parents[2] / settings.CNN_RESNET_WEIGHTS_PATH)
        self._classifier.load_model(
            weights_path=weights,
            model_arch=settings.CNN_RESNET_ARCH,
            num_classes=settings.CNN_RESNET_NUM_CLASSES,
            class_names=settings.CNN_RESNET_CLASS_NAMES,
        )
        logger.info("CNNPipeline loaded (loaded=%s)", self._classifier.is_loaded)

    @property
    def is_loaded(self) -> bool:
        return self._classifier is not None and self._classifier.is_loaded

    def run(self, image: np.ndarray, context: dict[str, Any]) -> PipelineResult:
        t0 = time.perf_counter()
        threshold = context.get("threshold", settings.CNN_RESNET_THRESHOLD)

        if not self.is_loaded:
            return PipelineResult(
                verdict="OK", overall_score=0.0, threshold=threshold,
                detections=[], anomaly_scores=[],
                processing_time_ms=(time.perf_counter() - t0) * 1000,
                metadata={"warning": "CNN weights not trained — pipeline disabled"},
            )

        result = self._classifier.classify(image, threshold=threshold)
        elapsed = (time.perf_counter() - t0) * 1000

        # overall_score must be the NG probability (0=clean, 1=defect) so the
        # decision engine threshold comparison works correctly. result.confidence
        # is the confidence of the *predicted* class (high for both OK and NG),
        # which would cause the decision engine to flip OK predictions to NG.
        ng_score = result.class_probabilities.get("NG", result.confidence if result.is_defect else 0.0)

        return PipelineResult(
            verdict=result.label,
            overall_score=ng_score,
            threshold=threshold,
            detections=[{
                "defect_label": f"cnn_{result.label.lower()}",
                "confidence": result.confidence,
                "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0,
                "is_defect": result.is_defect,
                "metadata": {"class_probabilities": result.class_probabilities},
            }],
            anomaly_scores=[{
                "model_name": "resnet18",
                "score": ng_score,
                "threshold": threshold,
                "passed": not result.is_defect,
            }],
            labels=["defect"] if result.is_defect else [],
            processing_time_ms=elapsed,
        )
