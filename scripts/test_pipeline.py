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

    # 2. YOLO Object Detection (pretrained COCO)
    print("\n[2] YOLO Object Detection (pretrained COCO)")
    from app.models.ai_models import YOLODetector

    obj_detector = YOLODetector()
    obj_detector.load_model("yolov8m.pt")
    print(f"    Object detector loaded: {obj_detector.is_loaded}")

    start = time.perf_counter()
    object_detections = obj_detector.detect(image, conf=0.40, detection_type="object")
    obj_ms = (time.perf_counter() - start) * 1000

    print(f"    Objects detected: {len(object_detections)}")
    print(f"    Inference time: {obj_ms:.1f}ms")
    for i, det in enumerate(object_detections):
        print(f"    [{i}] [O] {det.defect_class} conf={det.confidence:.3f} "
              f"bbox=({det.bbox_x1:.3f},{det.bbox_y1:.3f},{det.bbox_x2:.3f},{det.bbox_y2:.3f})")

    # 3. YOLO Defect Detection (custom-trained)
    print("\n[3] YOLO Defect Detection (custom-trained)")
    defect_detector = YOLODetector()
    yolo_weights = str(project_root / "backend" / "weights" / "best.pt")
    yolo_fallback = str(project_root / "weights" / "best.pt")

    loaded = False
    for wp in [yolo_weights, yolo_fallback]:
        if Path(wp).exists():
            defect_detector.load_model(wp)
            loaded = defect_detector.is_loaded
            if loaded:
                print(f"    Loaded custom weights: {wp}")
                break

    if not loaded:
        print("    Custom defect weights not found, loading pretrained yolov8n as fallback...")
        try:
            from ultralytics import YOLO
            defect_detector._model = YOLO("yolov8n.pt")
            defect_detector._weights_path = "yolov8n.pt"
            print("    Loaded pretrained YOLOv8n (fallback)")
        except Exception as e:
            print(f"    Failed to load YOLO: {e}")

    start = time.perf_counter()
    defect_detections = defect_detector.detect(image, conf=0.20, detection_type="defect")
    defect_ms = (time.perf_counter() - start) * 1000

    print(f"    Defects detected: {len(defect_detections)}")
    print(f"    Inference time: {defect_ms:.1f}ms")
    for i, det in enumerate(defect_detections):
        print(f"    [{i}] [D] {det.defect_class} conf={det.confidence:.3f} "
              f"bbox=({det.bbox_x1:.3f},{det.bbox_y1:.3f},{det.bbox_x2:.3f},{det.bbox_y2:.3f})")

    detections = object_detections + defect_detections
    yolo_ms = obj_ms + defect_ms

    # 4. SimpleNet Anomaly Detection
    print("\n[4] SimpleNet Anomaly Detection")
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

    # 5. Decision Logic
    print("\n[5] Decision Logic")
    has_yolo_defects = len(defect_detections) > 0
    has_anomaly = result.is_anomalous
    verdict = "DEFECT" if (has_yolo_defects or has_anomaly) else "OK"

    print(f"    Objects found:      {len(object_detections)}")
    print(f"    Defects found:      {len(defect_detections)}")
    print(f"    Anomaly detected:   {has_anomaly}")
    print(f"    ┌─────────────────────────┐")
    print(f"    │  VERDICT: {verdict:14s} │")
    print(f"    └─────────────────────────┘")

    # 6. Save heatmap visualization
    print("\n[6] Saving Heatmap Visualization")
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

        # Draw bounding boxes (green=object, red=defect)
        for det in detections:
            x1 = int(det.bbox_x1 * w)
            y1 = int(det.bbox_y1 * h)
            x2 = int(det.bbox_x2 * w)
            y2 = int(det.bbox_y2 * h)
            is_defect = det.detection_type == "defect"
            color = (0, 0, 255) if is_defect else (0, 255, 0)
            tag = "[D]" if is_defect else "[O]"
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            label = f"{tag} {det.defect_class} {det.confidence:.2f}"
            cv2.putText(overlay, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Verdict banner
        color = (0, 0, 255) if verdict == "DEFECT" else (0, 255, 0)
        cv2.rectangle(overlay, (10, 10), (250, 70), (0, 0, 0), -1)
        cv2.putText(overlay, verdict, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.putText(overlay, f"anomaly: {result.score:.4f}", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        overlay_path = output_dir / "overlay.png"
        cv2.imwrite(str(overlay_path), overlay)
        print(f"    Saved overlay: {overlay_path}")

    # 7. Summary
    total_ms = yolo_ms + simplenet_ms
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Image:           {Path(image_path).name}")
    print(f"  Objects detected:{len(object_detections)}")
    print(f"  Defects detected:{len(defect_detections)}")
    print(f"  Anomaly score:   {result.score:.6f}")
    print(f"  Verdict:         {verdict}")
    print(f"  Object det time: {obj_ms:.1f}ms")
    print(f"  Defect det time: {defect_ms:.1f}ms")
    print(f"  SimpleNet time:  {simplenet_ms:.1f}ms")
    print(f"  Total time:      {total_ms:.1f}ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
