"""Train ResNet-18 binary OK/NG classifier on NEU-DET steel defect dataset.

OK class: synthetic clean steel images (uniform gray + noise texture)
NG class: NEU-DET defect images (crazing, inclusion, patches, pitted, scale, scratches)

Downloads NEU-DET from Kaggle if not already cached.
Saves trained weights to weights/resnet_classifier.pth.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
KAGGLE_CACHE = (
    Path.home()
    / ".cache/kagglehub/datasets/kaustubhdikshit"
    / "neu-surface-defect-database/versions/1/NEU-DET"
)
OUTPUT_WEIGHTS = Path("weights/resnet_classifier.pth")
EPOCHS = 20
BATCH = 32
LR = 1e-4
DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
NUM_OK_SYNTH = 600  # synthetic clean images
IMG_SIZE = 224


# ── Synthetic clean image generator ──────────────────────────────────────────

def make_clean_images(n: int, rng: np.random.Generator) -> list[Image.Image]:
    images = []
    for _ in range(n):
        brightness = int(rng.integers(75, 135))
        arr = np.full((200, 200, 3), brightness, dtype=np.uint8)
        noise = rng.integers(-18, 18, (200, 200, 3), dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        step = int(rng.integers(3, 7))
        for row in range(0, 200, step):
            delta = int(rng.integers(3, 8))
            arr[row : row + 1] = np.clip(
                arr[row : row + 1].astype(np.int16) + delta, 0, 255
            ).astype(np.uint8)
        images.append(Image.fromarray(arr))
    return images


# ── Dataset ───────────────────────────────────────────────────────────────────

class SteelDataset(Dataset):
    def __init__(
        self,
        ok_images: list[Image.Image],
        ng_paths: list[Path],
        transform,
    ) -> None:
        self.samples: list[tuple] = (
            [(img, 0) for img in ok_images] + [(p, 1) for p in ng_paths]
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        src, label = self.samples[idx]
        img = src.convert("RGB") if isinstance(src, Image.Image) else Image.open(src).convert("RGB")
        return self.transform(img), label


# ── Data helpers ─────────────────────────────────────────────────────────────

def get_ng_paths() -> list[Path]:
    if not KAGGLE_CACHE.exists():
        logger.info("Downloading NEU-DET from Kaggle...")
        try:
            import kagglehub
            kagglehub.dataset_download("kaustubhdikshit/neu-surface-defect-database")
        except Exception as e:
            raise RuntimeError(
                f"Kaggle download failed: {e}. Install with: pip install kagglehub"
            ) from e

    paths: list[Path] = []
    for split in ["train", "validation"]:
        img_dir = KAGGLE_CACHE / split / "images"
        if img_dir.exists():
            for cls_dir in sorted(img_dir.iterdir()):
                if cls_dir.is_dir():
                    paths.extend(sorted(cls_dir.glob("*.jpg")))
    if not paths:
        raise RuntimeError(f"No images found under {KAGGLE_CACHE}")
    logger.info("Found %d NEU-DET defect images", len(paths))
    return paths


# ── Training ──────────────────────────────────────────────────────────────────

def train() -> None:
    logger.info("Device: %s", DEVICE)

    rng = np.random.default_rng(42)
    ng_paths = get_ng_paths()
    ok_images = make_clean_images(NUM_OK_SYNTH, rng)
    logger.info("Dataset: %d OK (synthetic) + %d NG (NEU-DET)", len(ok_images), len(ng_paths))

    transform_train = T.Compose([
        T.Resize(256),
        T.RandomCrop(IMG_SIZE),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_val = T.Compose([
        T.Resize(256),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 80/20 split
    ok_idx = np.random.default_rng(0).permutation(len(ok_images)).tolist()
    ng_idx = np.random.default_rng(0).permutation(len(ng_paths)).tolist()
    ok_split = int(len(ok_idx) * 0.8)
    ng_split = int(len(ng_idx) * 0.8)

    train_ds = SteelDataset(
        [ok_images[i] for i in ok_idx[:ok_split]],
        [ng_paths[i]  for i in ng_idx[:ng_split]],
        transform_train,
    )
    val_ds = SteelDataset(
        [ok_images[i] for i in ok_idx[ok_split:]],
        [ng_paths[i]  for i in ng_idx[ng_split:]],
        transform_val,
    )
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=2)
    logger.info("Train: %d samples | Val: %d samples", len(train_ds), len(val_ds))

    # ResNet-18 with 2-class head
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(DEVICE)

    # Weight OK class higher to compensate for fewer OK training samples
    n_ok = ok_split
    n_ng = ng_split
    class_weights = torch.tensor([n_ng / n_ok, 1.0], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(imgs)
            train_correct += (out.argmax(1) == labels).sum().item()
        scheduler.step()

        # Validate
        model.eval()
        val_correct = 0
        ok_tp = ok_total = ng_tp = ng_total = 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                preds = model(imgs).argmax(1)
                val_correct += (preds == labels).sum().item()
                ok_tp    += ((preds == 0) & (labels == 0)).sum().item()
                ok_total += (labels == 0).sum().item()
                ng_tp    += ((preds == 1) & (labels == 1)).sum().item()
                ng_total += (labels == 1).sum().item()

        val_acc = val_correct / len(val_ds)
        ok_acc  = ok_tp / max(ok_total, 1)
        ng_acc  = ng_tp / max(ng_total, 1)
        trn_acc = train_correct / len(train_ds)
        logger.info(
            "Epoch %2d/%d  train=%.3f  val=%.3f  OK_acc=%.3f  NG_acc=%.3f",
            epoch, EPOCHS, trn_acc, val_acc, ok_acc, ng_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            OUTPUT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), OUTPUT_WEIGHTS)
            logger.info("  ↳ Best weights saved (val_acc=%.3f)", best_val_acc)

    logger.info("Done. Best val accuracy: %.3f → %s", best_val_acc, OUTPUT_WEIGHTS)


if __name__ == "__main__":
    train()
