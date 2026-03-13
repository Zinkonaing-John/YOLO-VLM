"""SimpleNet training script for anomaly detection.

Usage:
    python train_simplenet.py --data-dir dataset/metal --epochs 100 --output weights/simplenet.pth

Dataset structure expected:
    data_dir/
        train/
            good/       # Normal (defect-free) images only
        test/
            good/       # Normal test images
            defect/     # Defective test images (any subfolders are also scanned)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.models.simplenet import SimpleNet, IMAGENET_MEAN, IMAGENET_STD, get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
)
logger = logging.getLogger(__name__)


class AnomalyDataset(Dataset):
    """Dataset for anomaly detection training/evaluation.

    For training: loads only normal (good) images.
    For testing: loads both good and defect images with labels.
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        input_size: int = 256,
        augment: bool = True,
    ) -> None:
        self.root = Path(root_dir)
        self.split = split
        self.input_size = input_size

        # Collect image paths and labels
        self.samples: list[tuple[Path, int]] = []  # (path, label) where 0=good, 1=defect

        if split == "train":
            good_dir = self.root / "train" / "good"
            if good_dir.exists():
                for f in sorted(good_dir.rglob("*")):
                    if f.suffix.lower() in self.EXTENSIONS:
                        self.samples.append((f, 0))
        else:
            # Test: good + defect
            test_good = self.root / "test" / "good"
            if test_good.exists():
                for f in sorted(test_good.rglob("*")):
                    if f.suffix.lower() in self.EXTENSIONS:
                        self.samples.append((f, 0))

            test_defect = self.root / "test" / "defect"
            if test_defect.exists():
                for f in sorted(test_defect.rglob("*")):
                    if f.suffix.lower() in self.EXTENSIONS:
                        self.samples.append((f, 1))

        if not self.samples:
            raise ValueError(f"No images found in {root_dir}/{split}/")

        # Transforms
        if augment and split == "train":
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

        logger.info(
            "Loaded %d images for %s from %s",
            len(self.samples), split, root_dir,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        import cv2

        path, label = self.samples[idx]
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img)
        return tensor, label


def compute_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute Area Under ROC Curve."""
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return 0.0


def train(
    data_dir: str,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-4,
    backbone: str = "resnet18",
    input_size: int = 256,
    feature_dim: int = 128,
    output_path: str = "weights/simplenet.pth",
    device_str: str | None = None,
) -> None:
    """Train SimpleNet on a dataset of normal images."""

    device = torch.device(device_str) if device_str else get_device()
    logger.info("Training on device: %s", device)

    # Dataset
    train_dataset = AnomalyDataset(data_dir, split="train", input_size=input_size)
    use_pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=use_pin_memory,
        drop_last=True,
    )

    # Model
    model = SimpleNet(
        backbone=backbone,
        feature_dim=feature_dim,
        input_size=input_size,
    )
    model.to(device)

    # Optimizers — separate for generator and discriminator
    gen_params = list(model.projector.parameters()) + list(model.generator.parameters())
    disc_params = list(model.discriminator.parameters())

    gen_optimizer = torch.optim.Adam(gen_params, lr=lr, weight_decay=1e-5)
    disc_optimizer = torch.optim.Adam(disc_params, lr=lr, weight_decay=1e-5)

    gen_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(gen_optimizer, T_max=epochs)
    disc_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(disc_optimizer, T_max=epochs)

    best_auroc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        model.feature_extractor.eval()  # Keep backbone frozen

        epoch_disc_loss = 0.0
        epoch_gen_loss = 0.0
        n_batches = 0

        for images, _ in train_loader:
            images = images.to(device)

            # Update discriminator
            disc_optimizer.zero_grad()
            disc_loss = model.training_step_disc(images)
            disc_loss.backward()
            disc_optimizer.step()

            # Update generator (and projector)
            gen_optimizer.zero_grad()
            gen_loss = model.training_step_gen(images)
            gen_loss.backward()
            gen_optimizer.step()

            epoch_disc_loss += disc_loss.item()
            epoch_gen_loss += gen_loss.item()
            n_batches += 1

        gen_scheduler.step()
        disc_scheduler.step()

        avg_disc = epoch_disc_loss / max(n_batches, 1)
        avg_gen = epoch_gen_loss / max(n_batches, 1)

        # Evaluate every 10 epochs
        if epoch % 10 == 0 or epoch == epochs:
            auroc = evaluate(model, data_dir, input_size, device)
            logger.info(
                "Epoch %d/%d — disc_loss=%.4f gen_loss=%.4f AUROC=%.4f",
                epoch, epochs, avg_disc, avg_gen, auroc,
            )

            if auroc > best_auroc:
                best_auroc = auroc
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                model.save(output_path)
                logger.info("New best AUROC=%.4f — saved to %s", auroc, output_path)
        else:
            if epoch % 5 == 0:
                logger.info(
                    "Epoch %d/%d — disc_loss=%.4f gen_loss=%.4f",
                    epoch, epochs, avg_disc, avg_gen,
                )

    logger.info("Training complete. Best AUROC=%.4f", best_auroc)


def evaluate(
    model: SimpleNet,
    data_dir: str,
    input_size: int,
    device: torch.device,
) -> float:
    """Evaluate model on test set and return AUROC."""
    try:
        test_dataset = AnomalyDataset(data_dir, split="test", input_size=input_size, augment=False)
    except ValueError:
        logger.warning("No test data found — skipping evaluation")
        return 0.0

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    model.eval()
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            output = model(images)
            scores = output["anomaly_score"].cpu().numpy()
            all_scores.extend(scores.tolist())
            all_labels.extend(labels.numpy().tolist())

    if len(set(all_labels)) < 2:
        logger.warning("Test set has only one class — cannot compute AUROC")
        return 0.0

    return compute_auroc(np.array(all_labels), np.array(all_scores))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SimpleNet for anomaly detection")
    parser.add_argument("--data-dir", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "wide_resnet50_2"])
    parser.add_argument("--input-size", type=int, default=256, help="Input image size")
    parser.add_argument("--feature-dim", type=int, default=128, help="Feature embedding dimension")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--output", type=str, default="weights/simplenet.pth", help="Output weights path")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        backbone=args.backbone,
        input_size=args.input_size,
        feature_dim=args.feature_dim,
        output_path=args.output,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
