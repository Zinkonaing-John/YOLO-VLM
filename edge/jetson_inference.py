#!/usr/bin/env python3
"""
NVIDIA Jetson Edge Inference Engine for Industrial Defect Detection.

Two-stage pipeline:
  1. YOLO-det  — localize defect regions (bounding boxes)
  2. YOLO-cls  — classify each ROI crop as OK / NG

Supports TensorRT engines (.engine) with FP16/INT8 for maximum throughput,
falling back to PyTorch (.pt) models when engines are unavailable.

Results are dispatched via MQTT and/or REST API.  Only NG frames are
uploaded to the backend server for full analysis (CLIP, SimpleNet, DB).
"""

import argparse
import base64
import json
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from camera_capture import CameraCapture
from mqtt_publisher import MQTTPublisher


# ── Verdict constants ────────────────────────────────────────────────────────

VERDICT_OK = "OK"
VERDICT_NG = "NG"

# Minimum ROI size (pixels) — ignore tiny crops that produce noisy classifications
_MIN_ROI_SIZE = 32


class JetsonInference:
    """Two-stage real-time inference runner for NVIDIA Jetson devices.

    Stage 1 (detection):  YOLO-det finds defect bounding boxes.
    Stage 2 (classification): YOLO-cls classifies each cropped ROI as OK/NG.

    Verdict logic (fail-safe toward NG):
      - Any ROI classified NG → frame verdict is NG
      - All ROIs classified OK *and* no detections → frame verdict is OK
    """

    def __init__(
        self,
        det_model_path: str = "weights/best.engine",
        cls_model_path: Optional[str] = None,
        camera_source: str = "0",
        api_url: Optional[str] = None,
        mqtt_broker: Optional[str] = None,
        mqtt_port: int = 1883,
        det_confidence: float = 0.5,
        cls_confidence: float = 0.5,
        img_size: int = 640,
        display: bool = False,
        upload_ng_only: bool = True,
    ):
        self.api_url = api_url
        self.det_confidence = det_confidence
        self.cls_confidence = cls_confidence
        self.img_size = img_size
        self.display = display
        self.upload_ng_only = upload_ng_only
        self.running = False

        # ------- Stage 1: detection model -------
        self.det_model = self._load_model(det_model_path, "detection")

        # ------- Stage 2: classification model (optional) -------
        self.cls_model = None
        if cls_model_path:
            self.cls_model = self._load_model(cls_model_path, "classification")

        # ------- camera -------
        self.camera = CameraCapture(source=camera_source)

        # ------- MQTT (optional) -------
        self.mqtt: Optional[MQTTPublisher] = None
        if mqtt_broker:
            self.mqtt = MQTTPublisher(broker=mqtt_broker, port=mqtt_port)
            self.mqtt.connect()

        # ------- metrics -------
        self._frame_times: list[float] = []
        self._fps: float = 0.0
        self._ok_count: int = 0
        self._ng_count: int = 0

        # ------- graceful shutdown -------
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model(self, model_path: str, role: str):
        """Load a TensorRT engine or fall back to a .pt model.

        Args:
            model_path: Path to .engine or .pt file.
            role: Human-readable label for log messages ("detection" / "classification").
        """
        if YOLO is None:
            print("[ERROR] ultralytics is not installed. Run: pip install ultralytics")
            sys.exit(1)

        path = Path(model_path)

        # Try the exact path first
        if path.exists():
            print(f"[INFO] Loading {role} model from {path}")
            return YOLO(str(path))

        # Fallback: .engine → .pt in the same directory
        fallback = path.with_suffix(".pt")
        if fallback.exists():
            print(f"[WARN] {role} TensorRT engine not found. Falling back to {fallback}")
            return YOLO(str(fallback))

        # Last resort: common locations
        for candidate in [
            Path("weights/best.pt"),
            Path("best.pt"),
            Path("yolov8n.pt"),
        ]:
            if candidate.exists():
                print(f"[WARN] {role}: using fallback model {candidate}")
                return YOLO(str(candidate))

        print(f"[ERROR] No {role} model found at {model_path}.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------
    def _signal_handler(self, signum, frame):
        print(f"\n[INFO] Received signal {signum}. Shutting down gracefully...")
        self.running = False

    # ------------------------------------------------------------------
    # FPS calculation
    # ------------------------------------------------------------------
    def _update_fps(self):
        now = time.monotonic()
        self._frame_times.append(now)
        self._frame_times = self._frame_times[-30:]
        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            self._fps = (len(self._frame_times) - 1) / elapsed if elapsed > 0 else 0.0

    # ------------------------------------------------------------------
    # Stage 1 — YOLO detection
    # ------------------------------------------------------------------
    def _detect(self, frame: np.ndarray) -> list[dict]:
        """Run YOLO-det and return a list of detection dicts."""
        results = self.det_model.predict(
            source=frame,
            conf=self.det_confidence,
            imgsz=self.img_size,
            verbose=False,
        )

        h, w = frame.shape[:2]
        detections = []

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                cls_name = r.names.get(cls_id, str(cls_id))
                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf, 4),
                    "bbox": {
                        "x1": round(x1, 1),
                        "y1": round(y1, 1),
                        "x2": round(x2, 1),
                        "y2": round(y2, 1),
                    },
                    "bbox_norm": {
                        "x1": round(x1 / w, 4),
                        "y1": round(y1 / h, 4),
                        "x2": round(x2 / w, 4),
                        "y2": round(y2 / h, 4),
                    },
                    # Will be enriched by Stage 2
                    "roi_verdict": None,
                    "roi_confidence": None,
                })

        return detections

    # ------------------------------------------------------------------
    # Stage 2 — YOLO-cls per-ROI classification
    # ------------------------------------------------------------------
    def _classify_roi(self, frame: np.ndarray, bbox: dict) -> tuple[str, float]:
        """Classify a single ROI crop as OK or NG.

        Returns:
            (verdict, confidence) tuple.
        """
        x1 = int(bbox["x1"])
        y1 = int(bbox["y1"])
        x2 = int(bbox["x2"])
        y2 = int(bbox["y2"])

        roi = frame[y1:y2, x1:x2]

        # Skip tiny ROIs — treat as NG (fail-safe)
        if roi.shape[0] < _MIN_ROI_SIZE or roi.shape[1] < _MIN_ROI_SIZE:
            return VERDICT_NG, 0.0

        results = self.cls_model.predict(
            source=roi,
            verbose=False,
        )

        if not results or results[0].probs is None:
            # Model returned nothing — fail-safe to NG
            return VERDICT_NG, 0.0

        probs = results[0].probs
        top1_idx = int(probs.top1)
        top1_conf = float(probs.top1conf)
        class_name = results[0].names.get(top1_idx, str(top1_idx))

        # Map classifier output to OK/NG
        # Convention: if the top class contains "ok", "good", or "normal" → OK
        label_lower = class_name.lower()
        if any(kw in label_lower for kw in ("ok", "good", "normal", "pass")):
            if top1_conf >= self.cls_confidence:
                return VERDICT_OK, top1_conf
            else:
                # Low confidence OK → fail-safe to NG
                return VERDICT_NG, 1.0 - top1_conf
        else:
            return VERDICT_NG, top1_conf

    def _classify_detections(
        self, frame: np.ndarray, detections: list[dict]
    ) -> list[dict]:
        """Enrich each detection with Stage 2 ROI classification."""
        if self.cls_model is None:
            # No classifier — every detection is treated as NG
            for det in detections:
                det["roi_verdict"] = VERDICT_NG
                det["roi_confidence"] = det["confidence"]
            return detections

        for det in detections:
            verdict, conf = self._classify_roi(frame, det["bbox"])
            det["roi_verdict"] = verdict
            det["roi_confidence"] = round(conf, 4)

        return detections

    # ------------------------------------------------------------------
    # Verdict logic
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_verdict(detections: list[dict]) -> str:
        """Determine frame-level verdict from per-ROI classifications.

        Fail-safe: any NG ROI → entire frame is NG.
        """
        if not detections:
            return VERDICT_OK

        for det in detections:
            if det.get("roi_verdict") == VERDICT_NG:
                return VERDICT_NG

        return VERDICT_OK

    # ------------------------------------------------------------------
    # Result formatting
    # ------------------------------------------------------------------
    def _format_results(
        self,
        detections: list[dict],
        verdict: str,
        frame_id: int,
        pipeline_ms: float,
    ) -> dict:
        """Build a JSON-serializable result payload."""
        return {
            "frame_id": frame_id,
            "timestamp": time.time(),
            "verdict": verdict,
            "detections": detections,
            "defect_count": sum(
                1 for d in detections if d.get("roi_verdict") == VERDICT_NG
            ),
            "total_detections": len(detections),
            "pipeline_ms": round(pipeline_ms, 2),
            "fps": round(self._fps, 1),
        }

    # ------------------------------------------------------------------
    # Result dispatching
    # ------------------------------------------------------------------
    def _send_to_api(self, payload: dict, frame: Optional[np.ndarray] = None):
        """POST results to the backend REST API.

        When upload_ng_only is True, only NG frames include the image.
        """
        if not self.api_url:
            return

        data = dict(payload)

        # Attach JPEG-encoded image for NG verdicts (or all if configured)
        if frame is not None and (
            not self.upload_ng_only or payload["verdict"] == VERDICT_NG
        ):
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            data["image_b64"] = base64.b64encode(jpeg.tobytes()).decode("ascii")

        try:
            resp = requests.post(self.api_url, json=data, timeout=5)
            if resp.status_code != 200:
                print(f"[WARN] API returned status {resp.status_code}")
        except requests.RequestException as exc:
            print(f"[WARN] API request failed: {exc}")

    def _publish_mqtt(self, payload: dict):
        """Publish results and alerts to MQTT topics."""
        if not self.mqtt:
            return

        self.mqtt.publish("inspection/results", payload)

        if payload["verdict"] == VERDICT_NG:
            alert = {
                "timestamp": payload["timestamp"],
                "frame_id": payload["frame_id"],
                "verdict": VERDICT_NG,
                "defect_count": payload["defect_count"],
                "pipeline_ms": payload["pipeline_ms"],
                "defects": [
                    {
                        "class": d["class_name"],
                        "det_confidence": d["confidence"],
                        "roi_verdict": d["roi_verdict"],
                        "roi_confidence": d["roi_confidence"],
                    }
                    for d in payload["detections"]
                    if d.get("roi_verdict") == VERDICT_NG
                ],
            }
            self.mqtt.publish("inspection/alerts", alert)

    # ------------------------------------------------------------------
    # Display overlay
    # ------------------------------------------------------------------
    def _draw_overlay(self, frame: np.ndarray, payload: dict) -> np.ndarray:
        """Draw bounding boxes, ROI verdicts, and stats on the frame."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        for det in payload["detections"]:
            bbox = det["bbox"]
            x1, y1 = int(bbox["x1"]), int(bbox["y1"])
            x2, y2 = int(bbox["x2"]), int(bbox["y2"])

            is_ng = det.get("roi_verdict") == VERDICT_NG
            color = (0, 0, 255) if is_ng else (0, 255, 0)  # Red for NG, green for OK

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{det['class_name']} {det['confidence']:.2f}"
            if det.get("roi_verdict"):
                label += f" | {det['roi_verdict']} {det.get('roi_confidence', 0):.2f}"

            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                annotated, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )

        # Frame-level verdict banner
        verdict = payload["verdict"]
        banner_color = (0, 0, 200) if verdict == VERDICT_NG else (0, 160, 0)
        cv2.rectangle(annotated, (0, 0), (w, 40), banner_color, -1)
        cv2.putText(
            annotated,
            f"{verdict} | FPS: {payload['fps']:.1f} | "
            f"Defects: {payload['defect_count']} | "
            f"Pipeline: {payload['pipeline_ms']:.1f}ms",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )

        return annotated

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        """Start the two-stage inference loop."""
        mode = "det+cls" if self.cls_model else "det-only"
        print(f"[INFO] Starting inference loop ({mode}). Press Ctrl+C to stop.")
        self.running = True
        frame_id = 0

        with self.camera:
            while self.running:
                frame = self.camera.read()
                if frame is None:
                    time.sleep(0.01)
                    continue

                frame_id += 1
                t_start = time.perf_counter()

                # Stage 1: detect defect regions
                detections = self._detect(frame)

                # Stage 2: classify each ROI
                detections = self._classify_detections(frame, detections)

                # Verdict
                verdict = self._compute_verdict(detections)
                pipeline_ms = (time.perf_counter() - t_start) * 1000.0

                self._update_fps()
                payload = self._format_results(detections, verdict, frame_id, pipeline_ms)

                # Track OK/NG counts
                if verdict == VERDICT_NG:
                    self._ng_count += 1
                else:
                    self._ok_count += 1

                # Dispatch
                self._send_to_api(payload, frame)
                self._publish_mqtt(payload)

                # Console output every 30 frames
                if frame_id % 30 == 0:
                    total = self._ok_count + self._ng_count
                    ng_rate = (self._ng_count / total * 100) if total > 0 else 0
                    print(
                        f"[INFO] Frame {frame_id} | "
                        f"FPS: {self._fps:.1f} | "
                        f"Verdict: {verdict} | "
                        f"Defects: {payload['defect_count']} | "
                        f"Pipeline: {pipeline_ms:.1f}ms | "
                        f"NG rate: {ng_rate:.1f}%"
                    )

                # Optional display
                if self.display:
                    annotated = self._draw_overlay(frame, payload)
                    cv2.imshow("Jetson Inference", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.running = False

        self._cleanup()

    def _cleanup(self):
        """Release resources and print summary."""
        print("[INFO] Cleaning up...")
        if self.display:
            cv2.destroyAllWindows()
        if self.mqtt:
            self.mqtt.disconnect()

        total = self._ok_count + self._ng_count
        if total > 0:
            print(
                f"[INFO] Session summary: {total} frames | "
                f"OK: {self._ok_count} | NG: {self._ng_count} | "
                f"NG rate: {self._ng_count / total * 100:.1f}%"
            )
        print("[INFO] Shutdown complete.")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Jetson two-stage defect detection: YOLO-det + YOLO-cls"
    )
    parser.add_argument(
        "--camera", type=str, default="0",
        help="Camera source: integer for USB/CSI, RTSP URL, or file path (default: 0)",
    )
    parser.add_argument(
        "--det-model", type=str, default="weights/best.engine",
        help="Stage 1 detection model: TensorRT .engine or .pt (default: weights/best.engine)",
    )
    parser.add_argument(
        "--cls-model", type=str, default=None,
        help="Stage 2 classification model: TensorRT .engine or .pt (optional)",
    )
    parser.add_argument(
        "--api-url", type=str, default=None,
        help="Backend API URL for result upload (e.g. http://server:8000/api/v1/edge-inspect)",
    )
    parser.add_argument(
        "--mqtt-broker", type=str, default=None,
        help="MQTT broker address (e.g. localhost)",
    )
    parser.add_argument(
        "--mqtt-port", type=int, default=1883,
        help="MQTT broker port (default: 1883)",
    )
    parser.add_argument(
        "--det-confidence", type=float, default=0.5,
        help="Stage 1 detection confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--cls-confidence", type=float, default=0.5,
        help="Stage 2 classification confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--img-size", type=int, default=640,
        help="Detection inference image size (default: 640)",
    )
    parser.add_argument(
        "--display", action="store_true",
        help="Show live annotated video window",
    )
    parser.add_argument(
        "--upload-all", action="store_true",
        help="Upload all frames to backend (default: NG only)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    engine = JetsonInference(
        det_model_path=args.det_model,
        cls_model_path=args.cls_model,
        camera_source=args.camera,
        api_url=args.api_url,
        mqtt_broker=args.mqtt_broker,
        mqtt_port=args.mqtt_port,
        det_confidence=args.det_confidence,
        cls_confidence=args.cls_confidence,
        img_size=args.img_size,
        display=args.display,
        upload_ng_only=not args.upload_all,
    )
    engine.run()


if __name__ == "__main__":
    main()
