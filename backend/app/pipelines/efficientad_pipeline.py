"""EfficientAD anomaly detection pipeline — primary production model.

EfficientAD is a fast, memory-efficient anomaly detector well-suited for edge deployment.
Weights are loaded from ``models/efficientad/model.ckpt``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.config import get_settings
from app.pipelines.base_pipeline import BasePipeline, PipelineResult

logger = logging.getLogger(__name__)
settings = get_settings()

# Input resolution expected by the trained EfficientAD model
_INPUT_SIZE = 256


class EfficientADPipeline(BasePipeline):
    """EfficientAD: student-teacher distillation anomaly detection."""

    name = "efficientad"
    description = "EfficientAD student-teacher anomaly detection (primary production model)"

    def __init__(self) -> None:
        self._model: Any = None
        self._device: Any = None
        self._transform: Any = None
        self._loaded = False
        # Online calibration: bypass post_processor's domain-specific min-max
        # normalization which saturates to 1.0 for out-of-distribution inputs.
        self._raw_min: float = float("inf")
        self._raw_max: float = float("-inf")

    def load(self) -> None:
        """Load EfficientAD checkpoint from models/efficientad/model.ckpt."""
        ckpt_path = Path(settings.EFFICIENTAD_WEIGHTS_PATH)
        if not ckpt_path.exists():
            logger.warning("EfficientAD checkpoint not found at %s", ckpt_path)
            return

        try:
            import torch
            from anomalib.models import EfficientAd  # type: ignore
            from anomalib.models.image.efficient_ad.torch_model import EfficientAdModelSize  # type: ignore
            from torchvision.transforms import v2 as T  # type: ignore

            # Resolve best available device
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            logger.info("Loading EfficientAD from %s on %s", ckpt_path, device)

            # PyTorch 2.6+ requires explicit safe globals for custom classes
            torch.serialization.add_safe_globals([EfficientAdModelSize])
            model = EfficientAd.load_from_checkpoint(
                str(ckpt_path),
                map_location=device,
                weights_only=False,
            )
            model.eval()
            model.to(device)

            self._model = model
            self._device = device
            # NOTE: Do NOT add ImageNet Normalize here.
            # EfficientAD's torch_model calls imagenet_norm_batch() internally.
            # Adding Normalize here would double-normalize and produce inverted scores.
            self._transform = T.Compose([
                T.Resize((_INPUT_SIZE, _INPUT_SIZE)),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
            ])
            self._loaded = True
            logger.info("EfficientAD loaded successfully (device=%s)", device)

        except ImportError as exc:
            logger.warning("anomalib or torchvision not installed — EfficientAD unavailable: %s", exc)
        except Exception:
            logger.exception("Failed to load EfficientAD model")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def run(self, image: np.ndarray, context: dict[str, Any]) -> PipelineResult:
        """Run EfficientAD inference on a BGR numpy image."""
        t0 = time.perf_counter()
        threshold = float(context.get("threshold", settings.EFFICIENTAD_THRESHOLD))

        if not self._loaded:
            logger.warning("EfficientAD not loaded — returning default OK")
            return PipelineResult(
                verdict="OK",
                overall_score=0.0,
                threshold=threshold,
                detections=[],
                anomaly_scores=[{"model_name": "efficientad", "score": 0.0,
                                  "threshold": threshold, "passed": True}],
                processing_time_ms=(time.perf_counter() - t0) * 1000,
                metadata={"warning": "Model not loaded"},
            )

        import torch
        from PIL import Image as PILImage  # type: ignore

        # BGR → RGB PIL image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)

        tensor = self._transform(pil_img).unsqueeze(0).to(self._device)

        with torch.no_grad():
            # Bypass post_processor: use raw torch_model outputs.
            # post_processor's normalization stats are domain-specific and
            # saturate to pred_score=1.0 for out-of-distribution images.
            # Raw anomaly_map.max() gives stable discrimination across domains.
            raw_out = self._model.model(tensor)

        # Extract raw anomaly map before post_processor normalization
        raw_am: np.ndarray | None = None
        raw_score: float = 0.0
        if hasattr(raw_out, 'anomaly_map') and raw_out.anomaly_map is not None:
            raw_am = raw_out.anomaly_map.squeeze().cpu().float().numpy()
            raw_score = float(raw_am.max())

        # Online min-max normalization (adapts to deployment domain)
        if raw_score < self._raw_min or self._raw_min == float("inf"):
            self._raw_min = raw_score
        if raw_score * 1.1 > self._raw_max or self._raw_max == float("-inf"):
            self._raw_max = raw_score * 1.1
        if self._raw_max > self._raw_min:
            score = min(1.0, (raw_score - self._raw_min) / (self._raw_max - self._raw_min))
        else:
            score = 0.0

        # anomaly_map for visualisation
        heatmap_np: np.ndarray | None = None
        if raw_am is not None:
            am_min, am_max = raw_am.min(), raw_am.max()
            if am_max > am_min:
                am_norm = ((raw_am - am_min) / (am_max - am_min) * 255).astype(np.uint8)
            else:
                am_norm = np.zeros_like(raw_am, dtype=np.uint8)
            heatmap_np = cv2.applyColorMap(am_norm, cv2.COLORMAP_JET)  # BGR

        passed = score < threshold
        verdict = "OK" if passed else "NG"

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("EfficientAD raw=%.4f norm=%.4f threshold=%.4f verdict=%s (%.1f ms)",
                    raw_score, score, threshold, verdict, elapsed)

        return PipelineResult(
            verdict=verdict,
            overall_score=score,
            threshold=threshold,
            detections=[],
            anomaly_scores=[{
                "model_name": "efficientad",
                "score": round(score, 4),
                "threshold": threshold,
                "passed": passed,
            }],
            heatmap=heatmap_np,
            processing_time_ms=elapsed,
            metadata={"device": str(self._device)},
        )
