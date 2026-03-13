"""Generate a synthetic anomaly detection dataset from real metal surface images.

Creates training data by:
- Extracting clean patches from relatively clean areas → train/good/
- Extracting patches with visible defects (scratches, stains) → test/defect/
- Extracting cleaner patches → test/good/

Also generates synthetic "good" images with controlled noise/texture.

Usage:
    python scripts/generate_training_data.py
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "anomaly"
PATCH_SIZE = 256


def extract_patches(image: np.ndarray, patch_size: int, stride: int) -> list[np.ndarray]:
    """Extract non-overlapping patches from an image."""
    h, w = image.shape[:2]
    patches = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y + patch_size, x:x + patch_size]
            patches.append(patch)
    return patches


def compute_defect_score(patch: np.ndarray) -> float:
    """Estimate how 'defective' a patch looks based on edge density and darkness."""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    # Edge density (scratches = high edges)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.mean(edges) / 255.0

    # Dark region ratio (stains/contamination)
    dark_ratio = np.mean(gray < 100) / 255.0

    # Variance (texture complexity)
    variance = np.std(gray.astype(float)) / 128.0

    score = 0.4 * edge_density + 0.3 * dark_ratio + 0.3 * variance
    return float(score)


def generate_clean_metal(size: int = 256, n: int = 50) -> list[np.ndarray]:
    """Generate synthetic clean metal surface patches."""
    patches = []
    for _ in range(n):
        # Base gray metal color
        base_val = random.randint(160, 210)
        img = np.full((size, size, 3), base_val, dtype=np.uint8)

        # Add slight color variation
        for c in range(3):
            offset = random.randint(-10, 10)
            img[:, :, c] = np.clip(img[:, :, c].astype(int) + offset, 0, 255).astype(np.uint8)

        # Add subtle Gaussian noise
        noise = np.random.normal(0, random.uniform(3, 8), img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Add subtle directional brushing texture
        if random.random() > 0.3:
            kernel_size = random.choice([3, 5, 7])
            angle = random.uniform(0, 180)
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[kernel_size // 2, :] = 1
            M = cv2.getRotationMatrix2D((kernel_size // 2, kernel_size // 2), angle, 1)
            kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
            kernel /= kernel.sum() + 1e-7
            img = cv2.filter2D(img, -1, kernel)

        # Slight Gaussian blur
        img = cv2.GaussianBlur(img, (3, 3), 0.5)

        patches.append(img)
    return patches


def generate_defect_patches(size: int = 256, n: int = 30) -> list[np.ndarray]:
    """Generate synthetic defective metal patches with scratches/stains."""
    patches = []
    for _ in range(n):
        # Start with clean base
        base_val = random.randint(160, 210)
        img = np.full((size, size, 3), base_val, dtype=np.uint8)
        noise = np.random.normal(0, 5, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        defect_type = random.choice(["scratch", "stain", "crack", "mixed"])

        if defect_type in ("scratch", "mixed"):
            # Draw random scratches
            n_scratches = random.randint(2, 8)
            for _ in range(n_scratches):
                pt1 = (random.randint(0, size), random.randint(0, size))
                pt2 = (random.randint(0, size), random.randint(0, size))
                color_val = random.randint(100, 220)
                thickness = random.randint(1, 3)
                cv2.line(img, pt1, pt2, (color_val, color_val, color_val), thickness)

        if defect_type in ("stain", "mixed"):
            # Draw random dark stains
            n_stains = random.randint(1, 4)
            for _ in range(n_stains):
                cx = random.randint(20, size - 20)
                cy = random.randint(20, size - 20)
                radius = random.randint(10, 50)
                darkness = random.randint(40, 120)
                mask = np.zeros((size, size), dtype=np.float32)
                cv2.circle(mask, (cx, cy), radius, 1.0, -1)
                mask = cv2.GaussianBlur(mask, (21, 21), radius / 3)
                for c in range(3):
                    channel = img[:, :, c].astype(np.float32)
                    channel = channel * (1 - mask * 0.6) + darkness * mask * 0.6
                    img[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)

        if defect_type in ("crack", "mixed"):
            # Draw crack-like pattern
            x, y = random.randint(30, size - 30), random.randint(30, size - 30)
            for _ in range(random.randint(5, 15)):
                dx = random.randint(-20, 20)
                dy = random.randint(-20, 20)
                nx, ny = np.clip(x + dx, 0, size - 1), np.clip(y + dy, 0, size - 1)
                cv2.line(img, (x, y), (nx, ny), (60, 60, 60), 1)
                x, y = nx, ny

        patches.append(img)
    return patches


def main():
    print("=" * 60)
    print("GENERATING ANOMALY DETECTION TRAINING DATASET")
    print("=" * 60)

    # Source image
    source_path = "/Users/zinko/Downloads/texture-background (1).jpg"
    source = cv2.imread(source_path)

    train_good_dir = OUTPUT_DIR / "train" / "good"
    test_good_dir = OUTPUT_DIR / "test" / "good"
    test_defect_dir = OUTPUT_DIR / "test" / "defect"

    train_good_dir.mkdir(parents=True, exist_ok=True)
    test_good_dir.mkdir(parents=True, exist_ok=True)
    test_defect_dir.mkdir(parents=True, exist_ok=True)

    if source is not None:
        print(f"\n[1] Extracting patches from real image: {source_path}")
        h, w = source.shape[:2]

        # Resize to manageable size for patch extraction
        scale = min(2048 / w, 2048 / h)
        if scale < 1:
            source = cv2.resize(source, (int(w * scale), int(h * scale)))
            print(f"    Resized to {source.shape[1]}x{source.shape[0]}")

        patches = extract_patches(source, PATCH_SIZE, stride=PATCH_SIZE // 2)
        print(f"    Extracted {len(patches)} patches")

        # Score each patch
        scored = [(p, compute_defect_score(p)) for p in patches]
        scored.sort(key=lambda x: x[1])

        # Bottom 40% = relatively clean → train/good
        n_clean = max(int(len(scored) * 0.4), 10)
        clean_patches = [p for p, _ in scored[:n_clean]]

        # Top 30% = most defective → test/defect
        n_defect = max(int(len(scored) * 0.3), 5)
        defect_patches = [p for p, _ in scored[-n_defect:]]

        # Middle band → test/good
        mid_start = n_clean
        mid_end = len(scored) - n_defect
        mid_patches = [p for p, _ in scored[mid_start:mid_end]]

        # Save real patches
        for i, p in enumerate(clean_patches):
            cv2.imwrite(str(train_good_dir / f"real_clean_{i:04d}.png"), p)
        print(f"    train/good (real): {len(clean_patches)} patches")

        for i, p in enumerate(mid_patches[:20]):
            cv2.imwrite(str(test_good_dir / f"real_clean_{i:04d}.png"), p)
        print(f"    test/good (real):  {min(len(mid_patches), 20)} patches")

        for i, p in enumerate(defect_patches):
            cv2.imwrite(str(test_defect_dir / f"real_defect_{i:04d}.png"), p)
        print(f"    test/defect (real): {len(defect_patches)} patches")
    else:
        print(f"\n[1] Source image not found at {source_path}, using synthetic data only")

    # Generate synthetic data
    print("\n[2] Generating synthetic clean metal patches")
    synthetic_clean = generate_clean_metal(PATCH_SIZE, n=150)
    for i, p in enumerate(synthetic_clean[:120]):
        cv2.imwrite(str(train_good_dir / f"synth_clean_{i:04d}.png"), p)
    for i, p in enumerate(synthetic_clean[120:]):
        cv2.imwrite(str(test_good_dir / f"synth_clean_{i:04d}.png"), p)
    print(f"    train/good (synthetic): 120 patches")
    print(f"    test/good (synthetic):  30 patches")

    print("\n[3] Generating synthetic defect patches")
    synthetic_defect = generate_defect_patches(PATCH_SIZE, n=50)
    for i, p in enumerate(synthetic_defect):
        cv2.imwrite(str(test_defect_dir / f"synth_defect_{i:04d}.png"), p)
    print(f"    test/defect (synthetic): 50 patches")

    # Apply augmentations to training set (flips, rotations)
    print("\n[4] Augmenting training set")
    train_images = list(train_good_dir.glob("*.png"))
    aug_count = 0
    for img_path in train_images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Horizontal flip
        flipped = cv2.flip(img, 1)
        cv2.imwrite(str(train_good_dir / f"aug_hflip_{img_path.stem}.png"), flipped)
        aug_count += 1

        # 90 degree rotation
        rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(str(train_good_dir / f"aug_rot90_{img_path.stem}.png"), rotated)
        aug_count += 1

    print(f"    Generated {aug_count} augmented images")

    # Summary
    train_count = len(list(train_good_dir.glob("*")))
    test_good_count = len(list(test_good_dir.glob("*")))
    test_defect_count = len(list(test_defect_dir.glob("*")))

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"  {OUTPUT_DIR}/")
    print(f"    train/good/   — {train_count} images")
    print(f"    test/good/    — {test_good_count} images")
    print(f"    test/defect/  — {test_defect_count} images")
    print(f"    Total:          {train_count + test_good_count + test_defect_count} images")
    print("=" * 60)


if __name__ == "__main__":
    main()
