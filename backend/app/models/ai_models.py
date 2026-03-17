"""AI model wrappers for YOLO object detection and CLIP defect classification."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    detection_type: str = "object"  # "object" or "defect"

    def to_dict(self) -> dict[str, Any]:
        return {
            "defect_class": self.defect_class,
            "confidence": round(self.confidence, 4),
            "bbox_x1": round(self.bbox_x1, 2),
            "bbox_y1": round(self.bbox_y1, 2),
            "bbox_x2": round(self.bbox_x2, 2),
            "bbox_y2": round(self.bbox_y2, 2),
            "detection_type": self.detection_type,
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
        detection_type: str = "object",
    ) -> list[DetectionResult]:
        """Run YOLO inference on an image.

        Args:
            detection_type: Tag each result as ``"object"`` or ``"defect"``.
        """
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
                        detection_type=detection_type,
                    )
                )

        logger.info(
            "YOLO detected %d %s(s) at conf>=%.2f",
            len(detections),
            detection_type,
            conf,
        )
        return detections


@dataclass
class SegmentationResult:
    """Single instance segmentation result with polygon mask."""
    defect_class: str
    confidence: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    polygon: list[list[float]]  # list of [x, y] normalized 0-1 points

    def to_dict(self) -> dict[str, Any]:
        return {
            "defect_class": self.defect_class,
            "confidence": round(self.confidence, 4),
            "bbox_x1": round(self.bbox_x1, 2),
            "bbox_y1": round(self.bbox_y1, 2),
            "bbox_x2": round(self.bbox_x2, 2),
            "bbox_y2": round(self.bbox_y2, 2),
            "polygon": [[round(x, 4), round(y, 4)] for x, y in self.polygon],
        }


class YOLOSegmentor:
    """Wrapper around Ultralytics YOLOv8-seg for instance segmentation."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._weights_path: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load_model(self, weights_path: str) -> None:
        path = Path(weights_path)
        is_standard_name = path.name == weights_path and "seg" in weights_path
        if not path.exists() and not is_standard_name:
            logger.warning(
                "YOLO-seg weights not found at %s – segmentor will not run.",
                weights_path,
            )
            return

        try:
            from ultralytics import YOLO

            self._model = YOLO(weights_path)
            self._weights_path = weights_path
            logger.info("YOLO-seg model loaded from %s", weights_path)
        except Exception:
            logger.exception("Failed to load YOLO-seg model from %s", weights_path)

    def segment(
        self,
        image: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> list[SegmentationResult]:
        """Run YOLOv8-seg inference on an image.

        Returns a list of ``SegmentationResult`` with polygon masks.
        """
        if self._model is None:
            logger.error("YOLO-seg model is not loaded – returning empty results.")
            return []

        h, w = image.shape[:2]

        results = self._model.predict(
            source=image,
            conf=conf,
            iou=iou,
            verbose=False,
        )

        segmentations: list[SegmentationResult] = []
        for result in results:
            boxes = result.boxes
            masks = result.masks
            if boxes is None or masks is None:
                continue

            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0].item())
                cls_name = result.names.get(cls_id, f"class_{cls_id}")
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Get polygon points (already normalized 0-1 by xyn)
                polygon_points = masks.xyn[i].tolist()

                segmentations.append(
                    SegmentationResult(
                        defect_class=cls_name,
                        confidence=confidence,
                        bbox_x1=x1 / w,
                        bbox_y1=y1 / h,
                        bbox_x2=x2 / w,
                        bbox_y2=y2 / h,
                        polygon=polygon_points,
                    )
                )

        logger.info("YOLO-seg detected %d instance(s) at conf>=%.2f", len(segmentations), conf)
        return segmentations

    def render_mask(
        self,
        image: np.ndarray,
        segmentations: list[SegmentationResult],
    ) -> np.ndarray:
        """Render a combined RGBA mask image with color-coded segments.

        Returns an RGBA numpy array (same size as input) with semi-transparent masks.
        """
        import cv2

        h, w = image.shape[:2]
        mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)

        # Single red color for all segments (BGR)
        color = (0, 0, 255)

        for i, seg in enumerate(segmentations):
            pts = np.array(
                [[int(x * w), int(y * h)] for x, y in seg.polygon],
                dtype=np.int32,
            )
            if len(pts) < 3:
                continue

            # Draw filled polygon with color
            cv2.fillPoly(mask_rgba, [pts], (*color, 120))
            # Draw polygon outline
            cv2.polylines(mask_rgba, [pts], isClosed=True, color=(*color, 220), thickness=2)

        return mask_rgba


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

    def classify_labels(
        self, image: np.ndarray, labels: list[str],
    ) -> tuple[str, float]:
        """Zero-shot classification against arbitrary text labels.

        Args:
            image: HxWxC numpy array (BGR).
            labels: List of candidate text labels.

        Returns:
            (best_label, confidence) tuple.
        """
        if self._model is None or not labels:
            return ("unknown", 0.0)

        try:
            import clip
            import torch
            from PIL import Image

            rgb = image[:, :, ::-1]
            pil_image = Image.fromarray(rgb)
            image_input = self._preprocess(pil_image).unsqueeze(0).to(self._device)
            text_inputs = clip.tokenize(labels).to(self._device)

            with torch.no_grad():
                img_feat = self._model.encode_image(image_input)
                txt_feat = self._model.encode_text(text_inputs)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
                probs = (img_feat @ txt_feat.T).squeeze(0).softmax(dim=-1).cpu().numpy()

            best_idx = int(probs.argmax())
            return (labels[best_idx], float(probs[best_idx]))
        except Exception:
            logger.exception("CLIP label classification failed")
            return ("unknown", 0.0)


# Note: detect_objects_by_contour removed — unreliable in production.
# Use YOLO for both object and defect detection instead.
