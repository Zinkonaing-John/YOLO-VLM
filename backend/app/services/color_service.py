"""Color inspection service using CIE LAB color space and Delta-E."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class ColorResult:
    """Result of a color comparison."""
    delta_e: float
    verdict: str  # "PASS" or "FAIL"
    sample_lab_mean: tuple[float, float, float]
    reference_lab_mean: tuple[float, float, float]

    def to_dict(self) -> dict:
        return {
            "delta_e": round(self.delta_e, 4),
            "verdict": self.verdict,
            "sample_lab_mean": [round(v, 2) for v in self.sample_lab_mean],
            "reference_lab_mean": [round(v, 2) for v in self.reference_lab_mean],
        }


class ColorInspector:
    """Performs color-difference inspection between a sample and a reference."""

    def __init__(self, threshold: float | None = None) -> None:
        self.threshold = threshold if threshold is not None else settings.DELTA_E_THRESHOLD

    @staticmethod
    def _bgr_to_lab(image: np.ndarray) -> np.ndarray:
        """Convert a BGR image (OpenCV default) to CIE-LAB via RGB."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
        return rgb2lab(rgb)

    @staticmethod
    def _mean_lab(lab_image: np.ndarray) -> tuple[float, float, float]:
        """Compute the mean L*, a*, b* values over the entire image."""
        mean = lab_image.reshape(-1, 3).mean(axis=0)
        return (float(mean[0]), float(mean[1]), float(mean[2]))

    def check_color(
        self,
        sample_image: np.ndarray,
        reference_image: np.ndarray,
        roi: tuple[int, int, int, int] | None = None,
    ) -> ColorResult:
        """Compare the colour of *sample_image* against *reference_image*.

        Both images should be BGR numpy arrays (as returned by ``cv2.imread``).

        Args:
            sample_image: The image under inspection.
            reference_image: The golden-reference image.
            roi: Optional (x1, y1, x2, y2) region-of-interest to crop both
                 images before comparison.

        Returns:
            ``ColorResult`` with the Delta-E value and PASS/FAIL verdict.
        """
        if roi is not None:
            x1, y1, x2, y2 = roi
            sample_image = sample_image[y1:y2, x1:x2]
            reference_image = reference_image[y1:y2, x1:x2]

        # Resize reference to sample dimensions if they differ
        if sample_image.shape[:2] != reference_image.shape[:2]:
            reference_image = cv2.resize(
                reference_image,
                (sample_image.shape[1], sample_image.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

        sample_lab = self._bgr_to_lab(sample_image)
        reference_lab = self._bgr_to_lab(reference_image)

        # Per-pixel Delta-E using CIEDE2000, then average
        de_map = deltaE_ciede2000(reference_lab, sample_lab)
        delta_e = float(np.mean(de_map))

        verdict = "PASS" if delta_e < self.threshold else "FAIL"

        sample_mean = self._mean_lab(sample_lab)
        ref_mean = self._mean_lab(reference_lab)

        logger.info(
            "Color check: DeltaE=%.4f  threshold=%.1f  verdict=%s",
            delta_e,
            self.threshold,
            verdict,
        )
        return ColorResult(
            delta_e=delta_e,
            verdict=verdict,
            sample_lab_mean=sample_mean,
            reference_lab_mean=ref_mean,
        )

    def check_color_from_files(
        self,
        sample_path: str,
        reference_path: str,
        roi: tuple[int, int, int, int] | None = None,
    ) -> ColorResult:
        """Convenience wrapper that reads images from file paths."""
        sample = cv2.imread(sample_path)
        reference = cv2.imread(reference_path)
        if sample is None:
            raise FileNotFoundError(f"Cannot read sample image: {sample_path}")
        if reference is None:
            raise FileNotFoundError(f"Cannot read reference image: {reference_path}")
        return self.check_color(sample, reference, roi=roi)
