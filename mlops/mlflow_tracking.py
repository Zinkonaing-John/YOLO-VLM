#!/usr/bin/env python3
"""
MLflow experiment tracking for YOLO defect detection models.

Provides helpers to log training runs, metrics, parameters, and model
artifacts, plus a comparison utility for selecting the best run.
"""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.tracking import MlflowClient

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------
DEFAULT_EXPERIMENT = "defect-detection"
DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def setup_mlflow(
    experiment_name: str = DEFAULT_EXPERIMENT,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> str:
    """Configure MLflow tracking URI and experiment.

    Returns:
        The experiment ID.
    """
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_name)
    print(f"[MLflow] Experiment '{experiment_name}' (id={experiment_id})")
    return experiment_id


# ------------------------------------------------------------------
# Training with full tracking
# ------------------------------------------------------------------
def train_and_log(
    data_yaml: str = "defect.yaml",
    model_name: str = "yolov8n.pt",
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    learning_rate: float = 0.01,
    device: str = "0",
    project: str = "runs/train",
    experiment_name: str = DEFAULT_EXPERIMENT,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    extra_params: Optional[Dict[str, Any]] = None,
) -> str:
    """Train a YOLO model and log everything to MLflow.

    Returns:
        The MLflow run ID.
    """
    if YOLO is None:
        raise ImportError("ultralytics is required: pip install ultralytics")

    setup_mlflow(experiment_name, tracking_uri)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"[MLflow] Run started: {run_id}")

        # --- log parameters ---
        params = {
            "model": model_name,
            "data": data_yaml,
            "epochs": epochs,
            "batch_size": batch_size,
            "img_size": img_size,
            "learning_rate": learning_rate,
            "device": device,
        }
        if extra_params:
            params.update(extra_params)
        mlflow.log_params(params)

        # --- train ---
        model = YOLO(model_name)
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            lr0=learning_rate,
            device=device,
            project=project,
            name=run_id,
            exist_ok=True,
        )

        # --- log metrics ---
        metrics = _extract_metrics(results)
        mlflow.log_metrics(metrics)
        print(f"[MLflow] Metrics: {metrics}")

        # --- log model artifact ---
        train_dir = Path(project) / run_id
        best_weight = train_dir / "weights" / "best.pt"
        if best_weight.exists():
            mlflow.log_artifact(str(best_weight), artifact_path="model")
            print(f"[MLflow] Artifact logged: {best_weight}")

        # Log training curves if present
        for curve in ["results.csv", "confusion_matrix.png", "F1_curve.png", "PR_curve.png"]:
            curve_path = train_dir / curve
            if curve_path.exists():
                mlflow.log_artifact(str(curve_path), artifact_path="plots")

        mlflow.set_tag("status", "completed")

    print(f"[MLflow] Run {run_id} completed.")
    return run_id


# ------------------------------------------------------------------
# Metrics extraction
# ------------------------------------------------------------------
def _extract_metrics(results) -> Dict[str, float]:
    """Pull key metrics from YOLO training results."""
    metrics: Dict[str, float] = {}

    # Results object structure varies by ultralytics version
    if hasattr(results, "results_dict"):
        rd = results.results_dict
        mapping = {
            "metrics/mAP50(B)": "mAP50",
            "metrics/mAP50-95(B)": "mAP50_95",
            "metrics/precision(B)": "precision",
            "metrics/recall(B)": "recall",
        }
        for src, dst in mapping.items():
            if src in rd:
                metrics[dst] = round(float(rd[src]), 4)
    elif hasattr(results, "maps"):
        metrics["mAP50"] = round(float(results.maps[0]), 4) if len(results.maps) > 0 else 0.0
        metrics["mAP50_95"] = round(float(results.maps.mean()), 4) if len(results.maps) > 0 else 0.0

    # Ensure we always have these keys
    metrics.setdefault("mAP50", 0.0)
    metrics.setdefault("mAP50_95", 0.0)
    metrics.setdefault("precision", 0.0)
    metrics.setdefault("recall", 0.0)

    return metrics


# ------------------------------------------------------------------
# Compare runs
# ------------------------------------------------------------------
def compare_runs(
    experiment_name: str = DEFAULT_EXPERIMENT,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    metric: str = "mAP50",
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """Return the top-N runs sorted by a given metric (descending).

    Each entry contains: run_id, params, and metrics.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"[MLflow] Experiment '{experiment_name}' not found.")
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=top_n,
    )

    results = []
    for r in runs:
        results.append(
            {
                "run_id": r.info.run_id,
                "status": r.info.status,
                "params": dict(r.data.params),
                "metrics": dict(r.data.metrics),
            }
        )
    return results


def get_best_run(
    experiment_name: str = DEFAULT_EXPERIMENT,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    metric: str = "mAP50",
) -> Optional[Dict[str, Any]]:
    """Return the single best run by the given metric."""
    top = compare_runs(experiment_name, tracking_uri, metric, top_n=1)
    return top[0] if top else None


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MLflow tracking for YOLO training")
    sub = parser.add_subparsers(dest="command")

    # train
    train_parser = sub.add_parser("train", help="Train and log a run")
    train_parser.add_argument("--data", default="defect.yaml")
    train_parser.add_argument("--model", default="yolov8n.pt")
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch", type=int, default=16)
    train_parser.add_argument("--imgsz", type=int, default=640)
    train_parser.add_argument("--lr", type=float, default=0.01)
    train_parser.add_argument("--device", default="0")

    # compare
    compare_parser = sub.add_parser("compare", help="Compare top runs")
    compare_parser.add_argument("--metric", default="mAP50")
    compare_parser.add_argument("--top", type=int, default=5)

    args = parser.parse_args()

    if args.command == "train":
        train_and_log(
            data_yaml=args.data,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.imgsz,
            learning_rate=args.lr,
            device=args.device,
        )
    elif args.command == "compare":
        runs = compare_runs(metric=args.metric, top_n=args.top)
        for i, r in enumerate(runs, 1):
            print(f"\n--- Run #{i} ---")
            print(f"  ID:      {r['run_id']}")
            print(f"  Status:  {r['status']}")
            print(f"  Params:  {r['params']}")
            print(f"  Metrics: {r['metrics']}")
    else:
        parser.print_help()
