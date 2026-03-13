"""Download and prepare an anomaly detection dataset for SimpleNet training.

Downloads the MVTec AD dataset (metal_nut category) from Kaggle,
then restructures it into the expected format:
    data/anomaly/
        train/good/
        test/good/
        test/defect/
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent


def download_mvtec_from_kaggle() -> Path:
    """Download MVTec AD dataset using kagglehub."""
    import kagglehub

    print("Downloading MVTec Anomaly Detection dataset from Kaggle...")
    print("(This may take a few minutes on first download)\n")

    path = kagglehub.dataset_download("ipythonx/mvtec-ad")
    print(f"Downloaded to: {path}")
    return Path(path)


def find_category(base_path: Path, category: str = "metal_nut") -> Path | None:
    """Find the category directory in the downloaded dataset."""
    # Search recursively for the category folder
    for p in base_path.rglob(category):
        if p.is_dir():
            # Verify it has train/test structure
            if (p / "train").exists() or (p / "test").exists():
                return p
    return None


def prepare_dataset(source: Path, output: Path, category: str = "metal_nut") -> None:
    """Restructure MVTec AD category into SimpleNet format."""
    output.mkdir(parents=True, exist_ok=True)

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def collect_and_copy(src_dir: Path, dst_dir: Path, prefix: str = "") -> int:
        dst_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        if not src_dir.exists():
            return 0
        for f in sorted(src_dir.rglob("*")):
            if f.suffix.lower() in IMAGE_EXTENSIONS and f.is_file():
                name = f"{prefix}{f.name}" if prefix else f.name
                dest = dst_dir / name
                idx = 1
                while dest.exists():
                    dest = dst_dir / f"{dest.stem}_{idx}{dest.suffix}"
                    idx += 1
                shutil.copy2(f, dest)
                count += 1
        return count

    # Training: only good images
    train_good = source / "train" / "good"
    n = collect_and_copy(train_good, output / "train" / "good")
    print(f"  train/good: {n} images")

    # Test good
    test_good = source / "test" / "good"
    n = collect_and_copy(test_good, output / "test" / "good")
    print(f"  test/good:  {n} images")

    # Test defects (all non-good subdirs)
    test_dir = source / "test"
    total_defect = 0
    if test_dir.exists():
        for subdir in sorted(test_dir.iterdir()):
            if subdir.is_dir() and subdir.name != "good":
                n = collect_and_copy(subdir, output / "test" / "defect", prefix=f"{subdir.name}_")
                total_defect += n
                print(f"  test/defect/{subdir.name}: {n} images")
    print(f"  test/defect total: {total_defect} images")


def main():
    output_dir = project_root / "data" / "anomaly"

    # Check if already prepared
    if (output_dir / "train" / "good").exists():
        count = len(list((output_dir / "train" / "good").glob("*")))
        if count > 0:
            print(f"Dataset already exists at {output_dir} ({count} training images)")
            print("Delete it first if you want to re-download.")
            return

    # Download
    base_path = download_mvtec_from_kaggle()

    # Try multiple categories in order of relevance
    categories = ["metal_nut", "screw", "tile", "grid", "carpet"]
    source = None
    chosen = None

    for cat in categories:
        source = find_category(base_path, cat)
        if source:
            chosen = cat
            break

    if source is None:
        # List what's available
        print(f"\nCould not find expected categories. Available directories:")
        for p in sorted(base_path.rglob("*")):
            if p.is_dir() and (p / "train").exists():
                print(f"  {p}")
        print("\nTrying to use the first available category...")

        for p in sorted(base_path.rglob("*")):
            if p.is_dir() and (p / "train").exists() and (p / "test").exists():
                source = p
                chosen = p.name
                break

    if source is None:
        print("ERROR: No valid dataset structure found!")
        sys.exit(1)

    print(f"\nUsing category: {chosen}")
    print(f"Source: {source}")
    print(f"Output: {output_dir}\n")

    prepare_dataset(source, output_dir, chosen)

    # Verify
    train_count = len(list((output_dir / "train" / "good").rglob("*")))
    test_good = len(list((output_dir / "test" / "good").rglob("*"))) if (output_dir / "test" / "good").exists() else 0
    test_defect = len(list((output_dir / "test" / "defect").rglob("*"))) if (output_dir / "test" / "defect").exists() else 0

    print(f"\nDataset prepared successfully!")
    print(f"  {output_dir}/train/good/   — {train_count} images")
    print(f"  {output_dir}/test/good/    — {test_good} images")
    print(f"  {output_dir}/test/defect/  — {test_defect} images")


if __name__ == "__main__":
    main()
