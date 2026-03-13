"""SimpleNet anomaly detection model for industrial surface inspection.

Reference: SimpleNet: A Simple Network for Image Anomaly Detection and Localization
https://arxiv.org/abs/2303.15140

Architecture:
  1. FeatureExtractor (frozen ResNet-18 backbone) — extracts multi-scale features
  2. AnomalyFeatureGenerator — projects normal features + noise to create pseudo-anomalies
  3. Discriminator — classifies features as normal vs anomalous
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

logger = logging.getLogger(__name__)

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_device() -> torch.device:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FeatureExtractor(nn.Module):
    """Frozen ResNet-18 backbone that extracts intermediate feature maps.

    Hooks into layers 1-3 of ResNet to capture multi-scale features.
    All parameters are frozen — no gradients flow through the backbone.
    """

    def __init__(self, backbone: str = "resnet18") -> None:
        super().__init__()

        if backbone == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.feature_dims = [64, 128, 256]
        elif backbone == "wide_resnet50_2":
            base = models.wide_resnet50_2(
                weights=models.Wide_ResNet50_2_Weights.DEFAULT
            )
            self.feature_dims = [256, 512, 1024]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Build sequential stages
        self.stage0 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.stage1 = base.layer1
        self.stage2 = base.layer2
        self.stage3 = base.layer3

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract feature maps from layers 1-3.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            List of 3 feature map tensors at different scales.
        """
        x = self.stage0(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        return [f1, f2, f3]

    @property
    def total_feature_dim(self) -> int:
        """Total channel dimension when features are concatenated."""
        return sum(self.feature_dims)


class FeatureProjector(nn.Module):
    """Projects concatenated backbone features to a fixed embedding dimension."""

    def __init__(self, in_dim: int, out_dim: int = 128) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class AnomalyFeatureGenerator(nn.Module):
    """Generates pseudo-anomaly features by adding learned noise to normal features.

    During training, takes normal feature embeddings and produces synthetic
    anomalous features that serve as negative examples for the discriminator.
    """

    def __init__(self, feature_dim: int = 128, noise_std: float = 0.015) -> None:
        super().__init__()
        self.noise_std = noise_std
        self.generator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate pseudo-anomaly features from normal features.

        Args:
            x: Normal feature embeddings (B, D).

        Returns:
            Pseudo-anomaly features (B, D).
        """
        noise = torch.randn_like(x) * self.noise_std
        return self.generator(x + noise)


class Discriminator(nn.Module):
    """Binary classifier: normal (0) vs anomalous (1).

    Small MLP that outputs anomaly probability for each feature vector.
    """

    def __init__(self, feature_dim: int = 128, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify features as normal or anomalous.

        Args:
            x: Feature embeddings (B, D).

        Returns:
            Anomaly scores in [0, 1] of shape (B, 1).
        """
        return self.net(x)


class SimpleNet(nn.Module):
    """SimpleNet anomaly detection model.

    Combines a frozen feature extractor with a trainable feature projector,
    anomaly feature generator, and discriminator for surface defect detection.

    Usage:
        model = SimpleNet(backbone="resnet18", input_size=256)
        model.load("weights/simplenet.pth")
        result = model.predict(bgr_image)  # returns AnomalyScore
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        feature_dim: int = 128,
        input_size: int = 256,
        noise_std: float = 0.015,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.feature_dim = feature_dim

        # Components
        self.feature_extractor = FeatureExtractor(backbone=backbone)
        self.projector = FeatureProjector(
            in_dim=self.feature_extractor.total_feature_dim,
            out_dim=feature_dim,
        )
        self.generator = AnomalyFeatureGenerator(
            feature_dim=feature_dim,
            noise_std=noise_std,
        )
        self.discriminator = Discriminator(
            feature_dim=feature_dim,
            hidden_dim=feature_dim,
        )

        # Preprocessing transform
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (input_size, input_size),
                    antialias=True,
                ),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        self._device = get_device()
        self._threshold = 0.5

    @property
    def device(self) -> torch.device:
        return self._device

    def _extract_and_project(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Extract backbone features, resize to common spatial size, concatenate, and project.

        Returns:
            Tuple of (projected_features, spatial_h, spatial_w).
            projected_features shape: (B * H * W, feature_dim)
        """
        features = self.feature_extractor(x)

        # Resize all feature maps to the largest spatial size (from layer 1)
        target_h, target_w = features[0].shape[2], features[0].shape[3]
        resized = []
        for f in features:
            if f.shape[2] != target_h or f.shape[3] != target_w:
                f = F.interpolate(
                    f, size=(target_h, target_w), mode="bilinear", align_corners=False
                )
            resized.append(f)

        # Concatenate along channel dimension: (B, C_total, H, W)
        concat = torch.cat(resized, dim=1)
        B, C, H, W = concat.shape

        # Reshape to (B*H*W, C) for projection
        flat = concat.permute(0, 2, 3, 1).reshape(-1, C)
        projected = self.projector(flat)

        return projected, H, W

    def forward(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Forward pass for inference.

        Args:
            x: Input tensor (B, 3, H, W), normalized.

        Returns:
            Dict with:
              - anomaly_score: per-image anomaly score (B,)
              - anomaly_map: spatial anomaly map (B, 1, input_size, input_size)
        """
        B = x.shape[0]

        projected, H, W = self._extract_and_project(x)

        # Discriminator scores for each spatial position
        scores = self.discriminator(projected)  # (B*H*W, 1)
        score_map = scores.reshape(B, 1, H, W)

        # Upsample to input resolution for heatmap
        anomaly_map = F.interpolate(
            score_map,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )

        # Per-image anomaly score: max of spatial scores
        anomaly_score = score_map.reshape(B, -1).max(dim=1).values

        return {
            "anomaly_score": anomaly_score,
            "anomaly_map": anomaly_map,
        }

    def training_step_disc(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """Discriminator training step.

        Args:
            x: Batch of normal images (B, 3, H, W).

        Returns:
            Discriminator loss.
        """
        with torch.no_grad():
            projected, H, W = self._extract_and_project(x)
            anomaly_features = self.generator(projected)

        normal_labels = torch.zeros(projected.shape[0], 1, device=x.device)
        anomaly_labels = torch.ones(anomaly_features.shape[0], 1, device=x.device)

        normal_scores = self.discriminator(projected)
        anomaly_scores = self.discriminator(anomaly_features)

        disc_loss = F.binary_cross_entropy(
            normal_scores, normal_labels
        ) + F.binary_cross_entropy(anomaly_scores, anomaly_labels)

        return disc_loss

    def training_step_gen(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """Generator training step.

        Args:
            x: Batch of normal images (B, 3, H, W).

        Returns:
            Generator loss.
        """
        projected, H, W = self._extract_and_project(x)
        anomaly_features = self.generator(projected)

        normal_labels = torch.zeros(anomaly_features.shape[0], 1, device=x.device)

        # Generator wants discriminator to classify anomalies as normal
        gen_anomaly_scores = self.discriminator(anomaly_features)
        gen_loss = F.binary_cross_entropy(gen_anomaly_scores, normal_labels)

        return gen_loss

    def predict(self, image: np.ndarray, threshold: Optional[float] = None) -> object:
        """Run anomaly detection on a BGR numpy image.

        Args:
            image: HxWxC numpy array (BGR, as from OpenCV).
            threshold: Override anomaly threshold (uses instance default if None).

        Returns:
            AnomalyScore dataclass with score, is_anomalous, and heatmap.
        """
        from app.models.ai_models import AnomalyScore

        thresh = threshold if threshold is not None else self._threshold

        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess
        tensor = self.transform(rgb).unsqueeze(0).to(self._device)

        # Inference
        self.eval()
        with torch.no_grad():
            output = self.forward(tensor)

        score = float(output["anomaly_score"].cpu().item())
        heatmap_tensor = output["anomaly_map"][0, 0].cpu().numpy()

        # Normalize heatmap to 0-255 for visualization
        heatmap_vis = (heatmap_tensor * 255).clip(0, 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)

        # Resize heatmap to match input image dimensions
        h, w = image.shape[:2]
        heatmap_colored = cv2.resize(heatmap_colored, (w, h))

        return AnomalyScore(
            score=score,
            is_anomalous=score >= thresh,
            heatmap=heatmap_colored,
        )

    def load(self, model_path: str, threshold: float = 0.5) -> None:
        """Load trained weights from disk.

        Args:
            model_path: Path to .pth weights file.
            threshold: Anomaly score threshold.
        """
        self._threshold = threshold
        checkpoint = torch.load(model_path, map_location=self._device, weights_only=False)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.load_state_dict(checkpoint["state_dict"])
        else:
            self.load_state_dict(checkpoint)

        self.to(self._device)
        self.eval()
        logger.info(
            "SimpleNet loaded from %s (threshold=%.3f, device=%s)",
            model_path,
            threshold,
            self._device,
        )

    def save(self, model_path: str) -> None:
        """Save model weights to disk.

        Args:
            model_path: Destination path for .pth file.
        """
        torch.save(
            {
                "state_dict": self.state_dict(),
                "input_size": self.input_size,
                "feature_dim": self.feature_dim,
                "threshold": self._threshold,
            },
            model_path,
        )
        logger.info("SimpleNet saved to %s", model_path)
