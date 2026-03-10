#!/usr/bin/env python3
"""
Automated retraining pipeline for industrial defect detection.

Workflow:
1. Check for new annotated data in data/annotated/
2. Load the current best model metrics from MLflow
3. Train a new model
4. Compare new metrics against the current best
5. If improved, promote the new model to weights/
6. Log everything to MLflow

Can be run manually or scheduled via cron / CI.
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from mlflow_tracking import (
    get_best_run,
    setup_mlflow,
    train_and_log,
    compare_runs,
)


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ANNOTATED_DIR = DATA_DIR / "annotated"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
MARKER_FILE = DATA_DIR / ".last_retrain_marker"
DEFAULT_DATA_YAML = str(PROJECT_ROOT / "defect.yaml")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def has_new_data() -> bool:
    """Return True if annotated data has been modified since the last retrain."""
    if not ANNOTATED_DIR.exists():
        print("[Pipeline] No annotated data directory found.")
        return False

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = [
        f
        for f in ANNOTATED_DIR.rglob("*")
        if f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print("[Pipeline] No image files in annotated data directory.")
        return False

    # Find the newest file
    newest_mtime = max(f.stat().st_mtime for f in image_files)

    # Compare against marker
    if MARKER_FILE.exists():
        marker_mtime = MARKER_FILE.stat().st_mtime
        if newest_mtime <= marker_mtime:
            print("[Pipeline] No new data since last retrain.")
            return False

    print(f"[Pipeline] Found {len(image_files)} images with new/updated data.")
    return True


def update_marker():
    """Touch the marker file to record the current retrain timestamp."""
    MARKER_FILE.parent.mkdir(parents=True, exist_ok=True)
    MARKER_FILE.touch()


def load_current_best_metrics(metric: str = "mAP50") -> Dict[str, float]:
    """Load the current best model metrics from MLflow."""
    best = get_best_run(metric=metric)
    if best is None:
        print("[Pipeline] No previous runs found. Treating as first run.")
        return {"mAP50": 0.0, "mAP50_95": 0.0, "precision": 0.0, "recall": 0.0}
    metrics = best["metrics"]
    print(f"[Pipeline] Current best run: {best['run_id']}")
    print(f"[Pipeline] Current best metrics: {metrics}")
    return metrics


def promote_model(source: Path, destination: Path = WEIGHTS_DIR / "best.pt"):
    """Copy the trained model to the production weights directory."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Back up existing model
    if destination.exists():
        backup = destination.with_name(
            f"best_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        )
        shutil.copy2(destination, backup)
        print(f"[Pipeline] Backed up previous model to {backup}")

    shutil.copy2(source, destination)
    print(f"[Pipeline] Promoted new model to {destination}")


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------
def run_pipeline(
    data_yaml: str = DEFAULT_DATA_YAML,
    base_model: str = "yolov8n.pt",
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    learning_rate: float = 0.01,
    device: str = "0",
    metric: str = "mAP50",
    improvement_threshold: float = 0.005,
    force: bool = False,
):
    """Execute the full retrain-evaluate-promote pipeline."""
    print("=" * 60)
    print(f"[Pipeline] Retraining pipeline started at {datetime.now().isoformat()}")
    print("=" * 60)

    # Step 1: check for new data
    if not force and not has_new_data():
        print("[Pipeline] Skipping retraining — no new data available.")
        return

    # Step 2: load current best
    current_best = load_current_best_metrics(metric=metric)
    current_score = current_best.get(metric, 0.0)

    # Step 3: use the current best.pt as the starting point if available
    resume_model = base_model
    best_pt = WEIGHTS_DIR / "best.pt"
    if best_pt.exists():
        resume_model = str(best_pt)
        print(f"[Pipeline] Resuming from existing best model: {resume_model}")
    else:
        print(f"[Pipeline] Starting fresh from: {base_model}")

    # Step 4: train
    print("[Pipeline] Starting training...")
    run_id = train_and_log(
        data_yaml=data_yaml,
        model_name=resume_model,
        epochs=epochs,
        batch_size=batch_size,
        img_size=img_size,
        learning_rate=learning_rate,
        device=device,
        project=str(PROJECT_ROOT / "runs" / "retrain"),
        extra_params={
            "pipeline": "auto_retrain",
            "improvement_threshold": improvement_threshold,
        },
    )

    # Step 5: compare
    best_run = get_best_run(metric=metric)
    if best_run is None:
        print("[Pipeline] Could not retrieve run metrics. Aborting promotion.")
        return

    new_score = best_run["metrics"].get(metric, 0.0)
    improvement = new_score - current_score
    print(f"[Pipeline] Previous best {metric}: {current_score:.4f}")
    print(f"[Pipeline] New model {metric}:      {new_score:.4f}")
    print(f"[Pipeline] Improvement:             {improvement:+.4f}")

    # Step 6: promote if improved
    if improvement >= improvement_threshold:
        print("[Pipeline] Improvement meets threshold — promoting model.")
        trained_weights = PROJECT_ROOT / "runs" / "retrain" / run_id / "weights" / "best.pt"
        if trained_weights.exists():
            promote_model(trained_weights)
        else:
            print(f"[Pipeline] Trained weights not found at {trained_weights}")
    else:
        print(
            f"[Pipeline] Improvement ({improvement:+.4f}) below threshold "
            f"({improvement_threshold}). Model NOT promoted."
        )

    # Step 7: update marker
    update_marker()

    print("=" * 60)
    print(f"[Pipeline] Pipeline finished at {datetime.now().isoformat()}")
    print("=" * 60)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Automated retraining pipeline for defect detection"
    )
    parser.add_argument(
        "--data", type=str, default=DEFAULT_DATA_YAML, help="Path to data YAML"
    )
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt", help="Base model to fine-tune"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--device", type=str, default="0", help="CUDA device")
    parser.add_argument(
        "--metric", type=str, default="mAP50", help="Metric to compare"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.005,
        help="Minimum improvement to promote model",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining even without new data",
    )
    args = parser.parse_args()

    run_pipeline(
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        learning_rate=args.lr,
        device=args.device,
        metric=args.metric,
        improvement_threshold=args.threshold,
        force=args.force,
    )


if __name__ == "__main__":
    main()
