"""Train YOLOv8n on the NEU Steel Surface Defect Dataset (NEU-DET).

Downloads the dataset from Kaggle, converts VOC XML annotations to YOLO
format, fine-tunes yolov8n.pt, and copies the best weights to weights/best.pt.
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from ultralytics import YOLO

# ── Configuration ──────────────────────────────────────────────────────
KAGGLE_CACHE = Path.home() / ".cache/kagglehub/datasets/kaustubhdikshit/neu-surface-defect-database/versions/1/NEU-DET"
DATASET_DIR = Path("datasets/NEU-DET")

BASE_WEIGHTS = "yolov8n.pt"
OUTPUT_WEIGHTS = Path("weights/best.pt")

EPOCHS = 50
IMGSZ = 640
BATCH = 16
DEVICE = "mps"

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}


def voc_to_yolo(xml_path: Path, img_w: int = 200, img_h: int = 200) -> list[str]:
    """Convert a VOC XML annotation to YOLO format lines."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is not None:
        img_w = int(size.findtext("width", str(img_w)))
        img_h = int(size.findtext("height", str(img_h)))

    lines = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()
        if name not in CLASS_TO_ID:
            continue
        cls_id = CLASS_TO_ID[name]
        bbox = obj.find("bndbox")
        xmin = float(bbox.findtext("xmin"))
        ymin = float(bbox.findtext("ymin"))
        xmax = float(bbox.findtext("xmax"))
        ymax = float(bbox.findtext("ymax"))

        # Convert to YOLO format: x_center, y_center, width, height (normalized)
        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
    return lines


def prepare_dataset() -> Path:
    """Convert Kaggle NEU-DET (VOC format) to YOLO format."""
    data_yaml = DATASET_DIR / "data.yaml"
    if data_yaml.exists():
        print(f"Dataset already prepared at {DATASET_DIR}, skipping.")
        return data_yaml

    if not KAGGLE_CACHE.exists():
        import kagglehub
        print("Downloading NEU-DET from Kaggle...")
        kagglehub.dataset_download("kaustubhdikshit/neu-surface-defect-database")

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    for split_src, split_dst in [("train", "train"), ("validation", "val")]:
        src_dir = KAGGLE_CACHE / split_src
        img_dst = DATASET_DIR / split_dst / "images"
        lbl_dst = DATASET_DIR / split_dst / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        # Copy images (flattening subdirectories)
        src_images = src_dir / "images"
        count = 0
        for class_dir in sorted(src_images.iterdir()):
            if not class_dir.is_dir():
                continue
            for img_file in sorted(class_dir.glob("*.jpg")):
                shutil.copy2(img_file, img_dst / img_file.name)
                count += 1

        # Convert annotations
        src_annotations = src_dir / "annotations"
        ann_count = 0
        for xml_file in sorted(src_annotations.glob("*.xml")):
            yolo_lines = voc_to_yolo(xml_file)
            label_file = lbl_dst / xml_file.with_suffix(".txt").name
            label_file.write_text("\n".join(yolo_lines) + "\n" if yolo_lines else "")
            ann_count += 1

        print(f"  {split_dst}: {count} images, {ann_count} annotations")

    # Write data.yaml
    config = {
        "path": str(DATASET_DIR.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": len(CLASSES),
        "names": CLASSES,
    }
    data_yaml.write_text(yaml.dump(config, default_flow_style=False))
    print(f"  data.yaml written to {data_yaml}")
    return data_yaml


def train(data_yaml: Path) -> Path:
    """Fine-tune YOLOv8n on NEU-DET and return path to best weights."""
    model = YOLO(BASE_WEIGHTS)
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project="runs/detect",
        name="train",
        exist_ok=True,
        pretrained=True,
        patience=20,
        save=True,
        plots=True,
    )
    best = Path("runs/detect/train/weights/best.pt")
    if not best.exists():
        raise FileNotFoundError(f"Training did not produce {best}")
    return best


def main() -> None:
    print("Step 1: Preparing NEU-DET dataset in YOLO format...")
    data_yaml = prepare_dataset()
    print(f"Dataset ready: {data_yaml}\n")

    print("Step 2: Training YOLOv8n on NEU-DET...")
    best_weights = train(data_yaml)
    print(f"Training complete. Best weights: {best_weights}\n")

    print("Step 3: Copying weights to weights/best.pt...")
    OUTPUT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, OUTPUT_WEIGHTS)
    print(f"Weights saved to {OUTPUT_WEIGHTS}\n")

    # Quick verification
    model = YOLO(str(OUTPUT_WEIGHTS))
    print(f"Model classes: {model.names}")
    print("Done!")


if __name__ == "__main__":
    main()
