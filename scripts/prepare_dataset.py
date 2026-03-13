"""Dataset preparation utilities for SimpleNet anomaly detection.

Supports:
  - MVTec AD dataset (https://www.mvtec.com/company/research/datasets/mvtec-ad)
  - NEU Surface Defect dataset
  - VisA anomaly dataset

Output structure:
    output_dir/
        train/
            good/           # Normal images for training
        test/
            good/           # Normal images for evaluation
            defect/         # Defective images for evaluation

Usage:
    # Prepare MVTec AD category
    python prepare_dataset.py mvtec --source ~/datasets/mvtec_ad --category metal_nut --output data/anomaly

    # Prepare NEU dataset
    python prepare_dataset.py neu --source ~/datasets/NEU-DET --output data/anomaly

    # Prepare from generic folder of good/defect images
    python prepare_dataset.py generic --good-dir ~/images/good --defect-dir ~/images/defect --output data/anomaly
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def collect_images(directory: Path) -> list[Path]:
    """Recursively collect all image files from a directory."""
    images = []
    for f in sorted(directory.rglob("*")):
        if f.suffix.lower() in IMAGE_EXTENSIONS and f.is_file():
            images.append(f)
    return images


def copy_images(images: list[Path], dest_dir: Path, prefix: str = "") -> int:
    """Copy images to destination directory with optional prefix."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for img in images:
        name = f"{prefix}{img.name}" if prefix else img.name
        dest = dest_dir / name
        # Handle name collisions
        if dest.exists():
            stem = dest.stem
            suffix = dest.suffix
            i = 1
            while dest.exists():
                dest = dest_dir / f"{stem}_{i}{suffix}"
                i += 1
        shutil.copy2(img, dest)
        count += 1
    return count


def prepare_mvtec(source_dir: str, output_dir: str, category: str) -> None:
    """Prepare MVTec AD dataset for SimpleNet training.

    MVTec AD structure:
        source_dir/
            <category>/
                train/
                    good/       → images
                test/
                    good/       → images
                    <defect_type>/  → images (scratch, dent, etc.)

    Args:
        source_dir: Root of MVTec AD dataset.
        output_dir: Output directory for prepared dataset.
        category: Category name (e.g., 'metal_nut', 'screw', 'tile').
    """
    source = Path(source_dir) / category
    output = Path(output_dir)

    if not source.exists():
        raise FileNotFoundError(f"Category directory not found: {source}")

    # Copy training good images
    train_good = source / "train" / "good"
    if train_good.exists():
        n = copy_images(collect_images(train_good), output / "train" / "good")
        logger.info("Copied %d training (good) images", n)
    else:
        raise FileNotFoundError(f"Training good directory not found: {train_good}")

    # Copy test good images
    test_good = source / "test" / "good"
    if test_good.exists():
        n = copy_images(collect_images(test_good), output / "test" / "good")
        logger.info("Copied %d test (good) images", n)

    # Copy test defect images (all defect type subdirs)
    test_dir = source / "test"
    total_defect = 0
    for subdir in sorted(test_dir.iterdir()):
        if subdir.is_dir() and subdir.name != "good":
            images = collect_images(subdir)
            n = copy_images(images, output / "test" / "defect", prefix=f"{subdir.name}_")
            total_defect += n
            logger.info("Copied %d test defect images from '%s'", n, subdir.name)

    logger.info("MVTec AD '%s' prepared: output at %s", category, output)
    logger.info("Total: train/good, test/good, test/defect=%d", total_defect)


def prepare_neu(source_dir: str, output_dir: str, train_ratio: float = 0.8) -> None:
    """Prepare NEU Surface Defect dataset for anomaly detection.

    NEU-DET has 6 classes: crazing, inclusion, patches, pitted_surface,
    rolled-in_scale, scratches. Each class has 300 images.

    For anomaly detection, we treat one category as 'good' and the rest as
    'defect'. By default, uses all images as defect samples and expects the
    user to provide a separate set of good samples.

    Alternatively, if the source has a 'good' folder, those are used for training.

    Args:
        source_dir: Root of NEU dataset (containing image folders).
        output_dir: Output directory for prepared dataset.
        train_ratio: Fraction of good images for training (rest for test).
    """
    source = Path(source_dir)
    output = Path(output_dir)

    # Check for explicit good/defect structure
    good_dir = source / "good"
    if good_dir.exists():
        good_images = collect_images(good_dir)
        random.seed(42)
        random.shuffle(good_images)

        split_idx = int(len(good_images) * train_ratio)
        train_good = good_images[:split_idx]
        test_good = good_images[split_idx:]

        n = copy_images(train_good, output / "train" / "good")
        logger.info("Copied %d training (good) images", n)
        n = copy_images(test_good, output / "test" / "good")
        logger.info("Copied %d test (good) images", n)

        # All other subdirectories are defect types
        total_defect = 0
        for subdir in sorted(source.iterdir()):
            if subdir.is_dir() and subdir.name != "good":
                images = collect_images(subdir)
                n = copy_images(images, output / "test" / "defect", prefix=f"{subdir.name}_")
                total_defect += n
                logger.info("Copied %d defect images from '%s'", n, subdir.name)
    else:
        # Flat structure — all images in source, split by pattern
        all_images = collect_images(source)
        if not all_images:
            raise FileNotFoundError(f"No images found in {source}")

        # Use all as defect samples
        logger.warning(
            "No 'good' folder found in %s. Treating all %d images as defect samples. "
            "Please provide normal images separately.",
            source, len(all_images),
        )
        n = copy_images(all_images, output / "test" / "defect")
        logger.info("Copied %d defect images", n)

    logger.info("NEU dataset prepared at %s", output)


def prepare_generic(
    good_dir: str,
    defect_dir: str | None,
    output_dir: str,
    train_ratio: float = 0.8,
) -> None:
    """Prepare a generic dataset from separate good/defect directories.

    Args:
        good_dir: Directory containing normal (good) images.
        defect_dir: Directory containing defect images (optional).
        output_dir: Output directory.
        train_ratio: Fraction of good images for training.
    """
    output = Path(output_dir)

    # Good images
    good_images = collect_images(Path(good_dir))
    if not good_images:
        raise FileNotFoundError(f"No images found in {good_dir}")

    random.seed(42)
    random.shuffle(good_images)

    split_idx = int(len(good_images) * train_ratio)
    train_good = good_images[:split_idx]
    test_good = good_images[split_idx:]

    n = copy_images(train_good, output / "train" / "good")
    logger.info("Copied %d training (good) images", n)
    n = copy_images(test_good, output / "test" / "good")
    logger.info("Copied %d test (good) images", n)

    # Defect images
    if defect_dir:
        defect_images = collect_images(Path(defect_dir))
        n = copy_images(defect_images, output / "test" / "defect")
        logger.info("Copied %d test defect images", n)

    logger.info("Generic dataset prepared at %s", output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare datasets for SimpleNet anomaly detection")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # MVTec AD
    mvtec_parser = subparsers.add_parser("mvtec", help="Prepare MVTec AD dataset")
    mvtec_parser.add_argument("--source", type=str, required=True, help="MVTec AD root directory")
    mvtec_parser.add_argument("--category", type=str, required=True, help="Category name")
    mvtec_parser.add_argument("--output", type=str, required=True, help="Output directory")

    # NEU
    neu_parser = subparsers.add_parser("neu", help="Prepare NEU Surface Defect dataset")
    neu_parser.add_argument("--source", type=str, required=True, help="NEU dataset root directory")
    neu_parser.add_argument("--output", type=str, required=True, help="Output directory")
    neu_parser.add_argument("--train-ratio", type=float, default=0.8, help="Training split ratio")

    # Generic
    generic_parser = subparsers.add_parser("generic", help="Prepare from good/defect directories")
    generic_parser.add_argument("--good-dir", type=str, required=True, help="Directory of normal images")
    generic_parser.add_argument("--defect-dir", type=str, default=None, help="Directory of defect images")
    generic_parser.add_argument("--output", type=str, required=True, help="Output directory")
    generic_parser.add_argument("--train-ratio", type=float, default=0.8, help="Training split ratio")

    args = parser.parse_args()

    if args.command == "mvtec":
        prepare_mvtec(args.source, args.output, args.category)
    elif args.command == "neu":
        prepare_neu(args.source, args.output, args.train_ratio)
    elif args.command == "generic":
        prepare_generic(args.good_dir, args.defect_dir, args.output, args.train_ratio)


if __name__ == "__main__":
    main()
