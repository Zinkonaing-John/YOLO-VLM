"""Test the YOLO + SimpleNet pipeline on a single image."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add backend to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "backend"))

def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/zinko/Downloads/texture-background (1).jpg"

    print("=" * 60)
    print("INDUSTRIAL DEFECT DETECTION PIPELINE TEST")
    print("=" * 60)

    # 1. Load image
    print(f"\n[1] Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Failed to load image at {image_path}")
        return
    h, w = image.shape[:2]
    print(f"    Image size: {w}x{h} ({image.shape})")

    # 2. YOLO Detection
    print("\n[2] YOLO Object Detection")
    from app.models.ai_models import YOLODetector

    detector = YOLODetector()
    yolo_weights = str(project_root / "backend" / "weights" / "best.pt")
    yolo_fallback = str(project_root / "weights" / "best.pt")

    # Try multiple weight locations, fall back to pretrained
    loaded = False
    for wp in [yolo_weights, yolo_fallback]:
        if Path(wp).exists():
            detector.load_model(wp)
            loaded = detector.is_loaded
            if loaded:
                print(f"    Loaded custom weights: {wp}")
                break

    if not loaded:
        print("    Custom weights not found, loading pretrained yolov8n...")
        try:
            from ultralytics import YOLO
            detector._model = YOLO("yolov8n.pt")
            detector._weights_path = "yolov8n.pt"
            print("    Loaded pretrained YOLOv8n")
        except Exception as e:
            print(f"    Failed to load YOLO: {e}")

    start = time.perf_counter()
    detections = detector.detect(image, conf=0.25)
    yolo_ms = (time.perf_counter() - start) * 1000

    print(f"    Detections: {len(detections)}")
    print(f"    Inference time: {yolo_ms:.1f}ms")
    for i, det in enumerate(detections):
        print(f"    [{i}] {det.defect_class} conf={det.confidence:.3f} "
              f"bbox=({det.bbox_x1:.0f},{det.bbox_y1:.0f},{det.bbox_x2:.0f},{det.bbox_y2:.0f})")

    # 3. SimpleNet Anomaly Detection
    print("\n[3] SimpleNet Anomaly Detection")
    from app.models.simplenet import SimpleNet, get_device

    device = get_device()
    print(f"    Device: {device}")

    model = SimpleNet(backbone="resnet18", input_size=256)
    model.to(device)
    model.eval()
    print("    SimpleNet initialized (untrained — using random weights for demo)")

    start = time.perf_counter()

    # Run on full image
    from app.models.ai_models import AnomalyScore
    result = model.predict(image, threshold=0.5)
    simplenet_ms = (time.perf_counter() - start) * 1000

    print(f"    Anomaly score: {result.score:.6f}")
    print(f"    Is anomalous: {result.is_anomalous}")
    print(f"    Heatmap shape: {result.heatmap.shape if result.heatmap is not None else 'None'}")
    print(f"    Inference time: {simplenet_ms:.1f}ms")

    # Run on ROIs if detections exist
    if detections:
        print("\n    Per-ROI anomaly scores:")
        for i, det in enumerate(detections):
            x1 = max(0, int(det.bbox_x1))
            y1 = max(0, int(det.bbox_y1))
            x2 = min(w, int(det.bbox_x2))
            y2 = min(h, int(det.bbox_y2))
            roi = image[y1:y2, x1:x2]
            if roi.size > 0:
                roi_result = model.predict(roi, threshold=0.5)
                print(f"    [{i}] {det.defect_class}: anomaly={roi_result.score:.6f} anomalous={roi_result.is_anomalous}")

    # 4. Decision Logic
    print("\n[4] Decision Logic")
    has_yolo_defects = len(detections) > 0
    has_anomaly = result.is_anomalous
    verdict = "DEFECT" if (has_yolo_defects or has_anomaly) else "OK"

    print(f"    YOLO defects found: {has_yolo_defects}")
    print(f"    Anomaly detected:   {has_anomaly}")
    print(f"    ┌─────────────────────────┐")
    print(f"    │  VERDICT: {verdict:14s} │")
    print(f"    └─────────────────────────┘")

    # 5. Save heatmap visualization
    print("\n[5] Saving Heatmap Visualization")
    output_dir = project_root / "uploads" / "test_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    if result.heatmap is not None:
        # Save raw heatmap
        heatmap_path = output_dir / "heatmap.png"
        cv2.imwrite(str(heatmap_path), result.heatmap)
        print(f"    Saved heatmap: {heatmap_path}")

        # Save overlay
        heatmap_resized = cv2.resize(result.heatmap, (w, h))
        overlay = cv2.addWeighted(image, 0.6, heatmap_resized, 0.4, 0)

        # Draw bounding boxes
        for det in detections:
            x1 = int(det.bbox_x1)
            y1 = int(det.bbox_y1)
            x2 = int(det.bbox_x2)
            y2 = int(det.bbox_y2)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{det.defect_class} {det.confidence:.2f}"
            cv2.putText(overlay, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Verdict banner
        color = (0, 0, 255) if verdict == "DEFECT" else (0, 255, 0)
        cv2.rectangle(overlay, (10, 10), (250, 70), (0, 0, 0), -1)
        cv2.putText(overlay, verdict, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.putText(overlay, f"anomaly: {result.score:.4f}", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        overlay_path = output_dir / "overlay.png"
        cv2.imwrite(str(overlay_path), overlay)
        print(f"    Saved overlay: {overlay_path}")

    # 6. Summary
    total_ms = yolo_ms + simplenet_ms
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Image:          {Path(image_path).name}")
    print(f"  YOLO detections:{len(detections)}")
    print(f"  Anomaly score:  {result.score:.6f}")
    print(f"  Verdict:        {verdict}")
    print(f"  YOLO time:      {yolo_ms:.1f}ms")
    print(f"  SimpleNet time: {simplenet_ms:.1f}ms")
    print(f"  Total time:     {total_ms:.1f}ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
