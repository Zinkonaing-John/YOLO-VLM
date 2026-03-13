"""AI model wrappers for YOLO object detection and CLIP defect classification."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AnomalyScore:
    """Result from SimpleNet anomaly detection."""
    score: float
    is_anomalous: bool
    heatmap: np.ndarray  # BGR colormap image, same size as input


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
        path = Path(weights_path)
        # Allow Ultralytics standard model names (e.g. "yolov8n.pt") to auto-download
        is_standard_name = path.name == weights_path and weights_path.startswith("yolov8")
        if not path.exists() and not is_standard_name:
            logger.warning(
                "YOLO weights not found at %s – detector will not run until "
                "valid weights are provided.",
                weights_path,
            )
            return

        try:
            from ultralytics import YOLO

            self._model = YOLO(weights_path)
            self._weights_path = weights_path
            logger.info("YOLO model loaded from %s", weights_path)
        except Exception:
            logger.exception("Failed to load YOLO model from %s", weights_path)

    def detect(
        self,
        image: np.ndarray,
        conf: float = 0.5,
        iou: float = 0.45,
        augment: bool = True,
    ) -> list[DetectionResult]:
        if self._model is None:
            logger.error("YOLO model is not loaded – returning empty results.")
            return []

        h, w = image.shape[:2]

        results = self._model.predict(
            source=image,
            conf=conf,
            iou=iou,
            augment=augment,
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

                # Normalize pixel coordinates to 0-1 range for frontend rendering
                detections.append(
                    DetectionResult(
                        defect_class=cls_name,
                        confidence=confidence,
                        bbox_x1=x1 / w,
                        bbox_y1=y1 / h,
                        bbox_x2=x2 / w,
                        bbox_y2=y2 / h,
                    )
                )

        logger.info(
            "YOLO detected %d object(s) at conf>=%.2f",
            len(detections),
            conf,
        )
        return detections


@dataclass
class CLIPResult:
    """Result from CLIP zero-shot classification on a cropped ROI."""
    label: str
    score: float
    is_defect: bool


class CLIPClassifier:
    """Zero-shot defect classifier using OpenAI CLIP.

    Classifies cropped ROIs as OK or NG by comparing CLIP similarity
    scores between OK-labels and NG-labels.
    """

    def __init__(self) -> None:
        self._model: Any | None = None
        self._preprocess: Any | None = None
        self._device: str = "cpu"
        self._ok_labels: list[str] = []
        self._ng_labels: list[str] = []

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load_model(
        self,
        model_name: str = "ViT-B/32",
        ok_labels: list[str] | None = None,
        ng_labels: list[str] | None = None,
    ) -> None:
        try:
            import clip
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model, self._preprocess = clip.load(model_name, device=self._device)
            self._ok_labels = ok_labels or ["a normal surface", "a good quality part"]
            self._ng_labels = ng_labels or ["a scratched surface", "a cracked surface", "a surface with bubbles"]
            logger.info("CLIP model %s loaded on %s", model_name, self._device)
        except Exception:
            logger.exception("Failed to load CLIP model %s", model_name)
            self._model = None

    def classify(self, image: np.ndarray, threshold: float = 0.5) -> CLIPResult:
        """Classify a BGR numpy ROI as OK or NG.

        Args:
            image: HxWxC numpy array (BGR).
            threshold: If the max NG score exceeds this, classify as defect.

        Returns:
            ``CLIPResult`` with the top label, score, and defect flag.
        """
        if self._model is None:
            logger.warning("CLIP model not loaded – returning OK by default.")
            return CLIPResult(label="unknown", score=0.0, is_defect=False)

        try:
            import clip
            import torch
            from PIL import Image

            # BGR → RGB → PIL
            rgb = image[:, :, ::-1]
            pil_image = Image.fromarray(rgb)
            image_input = self._preprocess(pil_image).unsqueeze(0).to(self._device)

            all_labels = self._ok_labels + self._ng_labels
            text_inputs = clip.tokenize(all_labels).to(self._device)

            with torch.no_grad():
                image_features = self._model.encode_image(image_input)
                text_features = self._model.encode_text(text_inputs)

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                similarities = (image_features @ text_features.T).squeeze(0)
                probs = similarities.softmax(dim=-1).cpu().numpy()

            n_ok = len(self._ok_labels)
            # Aggregate scores: sum of OK probs vs sum of NG probs
            ok_total = float(probs[:n_ok].sum())
            ng_total = float(probs[n_ok:].sum())
            ng_best_idx = int(probs[n_ok:].argmax())
            ok_best_idx = int(probs[:n_ok].argmax())

            # Defect if NG aggregate exceeds threshold (default 0.5 = majority)
            if ng_total >= threshold:
                return CLIPResult(
                    label=self._ng_labels[ng_best_idx],
                    score=ng_total,
                    is_defect=True,
                )
            else:
                return CLIPResult(
                    label=self._ok_labels[ok_best_idx],
                    score=ok_total,
                    is_defect=False,
                )

        except Exception:
            logger.exception("CLIP inference failed")
            return CLIPResult(label="error", score=0.0, is_defect=False)
