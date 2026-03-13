"""Real-time inference pipeline combining YOLO + SimpleNet.

Camera → Capture → YOLO detect → Crop ROI → SimpleNet anomaly → Display

Usage:
    # Webcam with live display
    python realtime_inference.py --source 0 --display

    # RTSP camera with API posting
    python realtime_inference.py --source rtsp://192.168.1.100/stream --api-url http://localhost:8000

    # Video file
    python realtime_inference.py --source test_video.mp4 --display
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add project paths
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "backend"))
sys.path.insert(0, str(project_root / "edge"))
sys.path.insert(0, str(project_root))

from app.models.ai_models import AnomalyDetector, YOLODetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
)
logger = logging.getLogger(__name__)


def draw_results(
    frame: np.ndarray,
    detections: list,
    anomaly_scores: dict[int, float],
    overall_score: float | None,
    verdict: str,
    fps: float,
) -> np.ndarray:
    """Draw detection results on frame."""
    display = frame.copy()
    h, w = display.shape[:2]

    # Draw bounding boxes
    for idx, det in enumerate(detections):
        x1 = int(det.bbox_x1)
        y1 = int(det.bbox_y1)
        x2 = int(det.bbox_x2)
        y2 = int(det.bbox_y2)

        color = (0, 0, 255) if anomaly_scores.get(idx, 0) >= 0.5 else (0, 255, 0)
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

        label = f"{det.defect_class} {det.confidence:.2f}"
        if idx in anomaly_scores:
            label += f" anom:{anomaly_scores[idx]:.3f}"

        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(display, (x1, y1 - 20), (x1 + label_size[0], y1), color, -1)
        cv2.putText(display, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Verdict banner
    verdict_color = (0, 255, 0) if verdict == "OK" else (0, 0, 255)
    cv2.rectangle(display, (10, 10), (200, 80), (0, 0, 0), -1)
    cv2.rectangle(display, (10, 10), (200, 80), verdict_color, 2)
    cv2.putText(display, verdict, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, verdict_color, 3)

    # Anomaly score
    if overall_score is not None:
        score_text = f"Anomaly: {overall_score:.4f}"
        cv2.putText(display, score_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # FPS
    cv2.putText(display, f"FPS: {fps:.1f}", (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return display


def overlay_heatmap(frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay anomaly heatmap on frame."""
    if heatmap is None:
        return frame
    h, w = frame.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    return cv2.addWeighted(frame, 1 - alpha, heatmap_resized, alpha, 0)


def post_result_to_api(api_url: str, frame: np.ndarray, verdict: str, anomaly_score: float) -> None:
    """Post inspection result to backend API."""
    try:
        import requests
        _, buffer = cv2.imencode(".jpg", frame)
        files = {"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
        requests.post(
            f"{api_url}/inspect",
            files=files,
            params={"enable_vlm": "false"},
            timeout=5,
        )
    except Exception as e:
        logger.debug("Failed to post to API: %s", e)


def run_pipeline(
    source: str | int,
    yolo_weights: str,
    simplenet_weights: str | None,
    threshold: float,
    confidence: float,
    display: bool,
    api_url: str | None,
) -> None:
    """Main inference loop."""

    # Initialize models
    detector = YOLODetector()
    detector.load_model(yolo_weights)

    anomaly_detector = AnomalyDetector()
    if simplenet_weights:
        anomaly_detector.load_model(simplenet_weights, threshold=threshold)

    # Initialize camera
    try:
        from camera_capture import CameraCapture
        camera = CameraCapture(source=source)
        camera.start()
        use_camera_capture = True
        logger.info("Using edge CameraCapture")
    except ImportError:
        # Fallback to direct OpenCV
        src = int(source) if str(source).isdigit() else source
        camera = cv2.VideoCapture(src)
        if not camera.isOpened():
            logger.error("Failed to open camera source: %s", source)
            return
        use_camera_capture = False
        logger.info("Using direct OpenCV VideoCapture")

    logger.info("Pipeline started. Press 'q' to quit.")

    frame_count = 0
    fps_start = time.time()
    fps = 0.0

    try:
        while True:
            # Capture frame
            if use_camera_capture:
                frame = camera.read()
                if frame is None:
                    time.sleep(0.01)
                    continue
            else:
                ret, frame = camera.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    break

            frame_count += 1

            # Calculate FPS
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()

            # YOLO detection
            detections = detector.detect(frame, conf=confidence)

            # SimpleNet anomaly detection
            overall_score = None
            per_defect_scores: dict[int, float] = {}

            if anomaly_detector.is_loaded:
                if detections:
                    for idx, det in enumerate(detections):
                        x1 = max(0, int(det.bbox_x1))
                        y1 = max(0, int(det.bbox_y1))
                        x2 = min(frame.shape[1], int(det.bbox_x2))
                        y2 = min(frame.shape[0], int(det.bbox_y2))
                        roi = frame[y1:y2, x1:x2]
                        if roi.size > 0:
                            result = anomaly_detector.predict(roi)
                            per_defect_scores[idx] = result.score
                    if per_defect_scores:
                        overall_score = max(per_defect_scores.values())
                else:
                    result = anomaly_detector.predict(frame)
                    overall_score = result.score

            # Decision
            has_defects = len(detections) > 0
            has_anomaly = overall_score is not None and overall_score >= threshold
            verdict = "DEFECT" if (has_defects or has_anomaly) else "OK"

            # Display
            if display:
                display_frame = draw_results(frame, detections, per_defect_scores, overall_score, verdict, fps)

                # Overlay heatmap on full image if anomaly detected
                if has_anomaly and anomaly_detector.is_loaded:
                    full_result = anomaly_detector.predict(frame)
                    if full_result.heatmap is not None:
                        display_frame = overlay_heatmap(display_frame, full_result.heatmap, alpha=0.3)

                cv2.imshow("Industrial Defect Detection", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            # Post to API
            if api_url and verdict == "DEFECT":
                post_result_to_api(api_url, frame, verdict, overall_score or 0.0)

            # Log defects
            if verdict == "DEFECT":
                logger.info(
                    "DEFECT detected — %d YOLO detections, anomaly=%.4f",
                    len(detections),
                    overall_score or 0.0,
                )

    except KeyboardInterrupt:
        logger.info("Pipeline stopped by user")
    finally:
        if use_camera_capture:
            camera.stop()
        else:
            camera.release()
        if display:
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time YOLO + SimpleNet inference pipeline")
    parser.add_argument("--source", type=str, default="0", help="Camera source (index, RTSP URL, or video file)")
    parser.add_argument("--yolo-weights", type=str, default="weights/best.pt", help="YOLO weights path")
    parser.add_argument("--simplenet-weights", type=str, default=None, help="SimpleNet weights path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Anomaly score threshold")
    parser.add_argument("--confidence", type=float, default=0.5, help="YOLO confidence threshold")
    parser.add_argument("--display", action="store_true", help="Show live OpenCV window")
    parser.add_argument("--api-url", type=str, default=None, help="Backend API URL for posting results")
    args = parser.parse_args()

    source: str | int = int(args.source) if args.source.isdigit() else args.source

    run_pipeline(
        source=source,
        yolo_weights=args.yolo_weights,
        simplenet_weights=args.simplenet_weights,
        threshold=args.threshold,
        confidence=args.confidence,
        display=args.display,
        api_url=args.api_url,
    )


if __name__ == "__main__":
    main()
