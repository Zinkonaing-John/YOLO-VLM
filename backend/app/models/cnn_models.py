"""CNN & ResNet model wrappers for image classification-based defect detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ResNetResult:
    """Result from ResNet CNN classification."""
    label: str        # "OK" or "NG"
    confidence: float  # 0-1 confidence of the predicted class
    is_defect: bool
    class_probabilities: dict[str, float]  # {class_name: probability}


class ResNetClassifier:
    """CNN image classifier using a ResNet backbone for OK/NG classification.

    Pipeline: Camera → Image → CNN (ResNet) → Detect → OK/NG

    Supports both pretrained ImageNet ResNet (fine-tuned) and custom-trained
    weights for binary industrial defect classification.
    """

    def __init__(self) -> None:
        self._model: Any | None = None
        self._device: str = "cpu"
        self._class_names: list[str] = ["OK", "NG"]
        self._transform: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load_model(
        self,
        weights_path: str | None = None,
        model_arch: str = "resnet18",
        num_classes: int = 2,
        class_names: list[str] | None = None,
    ) -> None:
        """Load a ResNet model for binary classification.

        Args:
            weights_path: Path to custom-trained weights (.pth). If None,
                          loads pretrained ImageNet weights with modified head.
            model_arch: Architecture — 'resnet18', 'resnet34', 'resnet50'.
            num_classes: Number of output classes (default 2 for OK/NG).
            class_names: Human-readable class names matching output indices.
        """
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as T

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._class_names = class_names or ["OK", "NG"]

            # Build the model architecture
            arch_map = {
                "resnet18": models.resnet18,
                "resnet34": models.resnet34,
                "resnet50": models.resnet50,
            }
            builder = arch_map.get(model_arch)
            if builder is None:
                logger.error("Unsupported ResNet architecture: %s", model_arch)
                return

            if weights_path and Path(weights_path).exists():
                # Load custom-trained weights
                model = builder(weights=None)
                in_features = model.fc.in_features
                model.fc = torch.nn.Linear(in_features, num_classes)
                state_dict = torch.load(weights_path, map_location=self._device, weights_only=True)
                model.load_state_dict(state_dict)
                logger.info("ResNet custom weights loaded from %s", weights_path)
            else:
                # Use pretrained ImageNet weights with modified classifier head
                model = builder(weights="IMAGENET1K_V1")
                in_features = model.fc.in_features
                model.fc = torch.nn.Linear(in_features, num_classes)
                if weights_path:
                    logger.warning(
                        "ResNet weights not found at %s – using pretrained ImageNet backbone "
                        "(classifier head is untrained, results may be unreliable until fine-tuned).",
                        weights_path,
                    )
                else:
                    logger.info("ResNet loaded with pretrained ImageNet backbone (no custom weights)")

            model.to(self._device)
            model.eval()
            self._model = model

            # Standard ImageNet preprocessing
            self._transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            logger.info(
                "ResNet classifier ready (%s, %d classes, device=%s)",
                model_arch, num_classes, self._device,
            )
        except Exception:
            logger.exception("Failed to load ResNet model")
            self._model = None

    def classify(self, image: np.ndarray, threshold: float = 0.5) -> ResNetResult:
        """Classify a BGR numpy image as OK or NG.

        Args:
            image: HxWxC numpy array (BGR).
            threshold: If NG probability exceeds this, classify as defect.

        Returns:
            ``ResNetResult`` with label, confidence, and class probabilities.
        """
        if self._model is None:
            logger.warning("ResNet model not loaded – returning OK by default.")
            return ResNetResult(
                label="OK", confidence=0.0, is_defect=False,
                class_probabilities={"OK": 0.5, "NG": 0.5},
            )

        try:
            import torch
            from PIL import Image

            # BGR → RGB → PIL
            rgb = image[:, :, ::-1]
            pil_image = Image.fromarray(rgb)
            input_tensor = self._transform(pil_image).unsqueeze(0).to(self._device)

            with torch.no_grad():
                logits = self._model(input_tensor)
                probs = torch.nn.functional.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            class_probs = {
                name: round(float(probs[i]), 4)
                for i, name in enumerate(self._class_names)
            }

            # Determine verdict
            ng_idx = self._class_names.index("NG") if "NG" in self._class_names else 1
            ng_prob = float(probs[ng_idx])
            is_defect = ng_prob >= threshold

            if is_defect:
                label = "NG"
                confidence = ng_prob
            else:
                ok_idx = 0 if ng_idx == 1 else 1
                label = "OK"
                confidence = float(probs[ok_idx])

            return ResNetResult(
                label=label,
                confidence=round(confidence, 4),
                is_defect=is_defect,
                class_probabilities=class_probs,
            )

        except Exception:
            logger.exception("ResNet inference failed")
            return ResNetResult(
                label="OK", confidence=0.0, is_defect=False,
                class_probabilities={"OK": 0.5, "NG": 0.5},
            )

    def get_cam_heatmap(self, image: np.ndarray) -> np.ndarray | None:
        """Generate a class activation map (GradCAM-like) heatmap for visualization.

        Returns a BGR heatmap image of same size as input, or None on failure.
        """
        if self._model is None:
            return None

        try:
            import cv2
            import torch
            from PIL import Image

            h, w = image.shape[:2]

            # BGR → RGB → PIL
            rgb = image[:, :, ::-1]
            pil_image = Image.fromarray(rgb)
            input_tensor = self._transform(pil_image).unsqueeze(0).to(self._device)
            input_tensor.requires_grad_(True)

            # Hook into the last conv layer (layer4 for ResNet)
            activations = []
            gradients = []

            def forward_hook(module, inp, out):
                activations.append(out)

            def backward_hook(module, grad_in, grad_out):
                gradients.append(grad_out[0])

            target_layer = self._model.layer4[-1]
            fh = target_layer.register_forward_hook(forward_hook)
            bh = target_layer.register_full_backward_hook(backward_hook)

            # Forward pass
            output = self._model(input_tensor)
            pred_class = output.argmax(dim=1).item()

            # Backward pass for predicted class
            self._model.zero_grad()
            output[0, pred_class].backward()

            # Compute GradCAM
            grads = gradients[0].squeeze(0)   # [C, H, W]
            acts = activations[0].squeeze(0)  # [C, H, W]
            weights = grads.mean(dim=(1, 2))  # [C]

            cam = torch.zeros(acts.shape[1:], device=self._device)
            for i, weight in enumerate(weights):
                cam += weight * acts[i]

            cam = torch.relu(cam)
            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()

            cam_np = cam.cpu().numpy()
            cam_resized = cv2.resize(cam_np, (w, h))
            heatmap = cv2.applyColorMap(
                (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
            )

            # Cleanup hooks
            fh.remove()
            bh.remove()

            return heatmap

        except Exception:
            logger.exception("GradCAM heatmap generation failed")
            return None
