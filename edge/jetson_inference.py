#!/usr/bin/env python3
"""
NVIDIA Jetson Edge Inference Engine for Industrial Defect Detection.

Loads a YOLO TensorRT engine (best.engine) or falls back to a .pt model,
captures frames from a CSI / USB / RTSP camera, runs real-time inference,
and forwards results to a backend API and/or an MQTT broker.
"""

import argparse
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


class JetsonInference:
    """Real-time inference runner for NVIDIA Jetson devices."""

    def __init__(
        self,
        model_path: str = "weights/best.engine",
        camera_source: str = "0",
        api_url: Optional[str] = None,
        mqtt_broker: Optional[str] = None,
        mqtt_port: int = 1883,
        confidence: float = 0.5,
        img_size: int = 640,
        display: bool = False,
    ):
        self.api_url = api_url
        self.confidence = confidence
        self.img_size = img_size
        self.display = display
        self.running = False

        # ------- model -------
        self.model = self._load_model(model_path)

        # ------- camera -------
        self.camera = CameraCapture(source=camera_source)

        # ------- MQTT (optional) -------
        self.mqtt: Optional[MQTTPublisher] = None
        if mqtt_broker:
            self.mqtt = MQTTPublisher(broker=mqtt_broker, port=mqtt_port)
            self.mqtt.connect()

        # ------- FPS tracking -------
        self._frame_times: list[float] = []
        self._fps: float = 0.0

        # ------- graceful shutdown -------
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model(self, model_path: str):
        """Load TensorRT engine if available, otherwise fall back to .pt."""
        if YOLO is None:
            print("[ERROR] ultralytics is not installed. Run: pip install ultralytics")
            sys.exit(1)

        path = Path(model_path)
        if path.exists():
            print(f"[INFO] Loading model from {path}")
            return YOLO(str(path))

        # Fallback: try .pt in the same directory
        fallback = path.with_suffix(".pt")
        if fallback.exists():
            print(f"[WARN] TensorRT engine not found. Falling back to {fallback}")
            return YOLO(str(fallback))

        # Last resort: look in common locations
        for candidate in [
            Path("weights/best.pt"),
            Path("best.pt"),
            Path("yolov8n.pt"),
        ]:
            if candidate.exists():
                print(f"[WARN] Using fallback model {candidate}")
                return YOLO(str(candidate))

        print("[ERROR] No model file found. Provide a valid --model path.")
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
        # Keep only the last 30 timestamps
        self._frame_times = self._frame_times[-30:]
        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            self._fps = (len(self._frame_times) - 1) / elapsed if elapsed > 0 else 0.0

    # ------------------------------------------------------------------
    # Result formatting
    # ------------------------------------------------------------------
    @staticmethod
    def _format_results(results, frame_id: int) -> dict:
        """Convert YOLO results to a JSON-serialisable dict."""
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
                detections.append(
                    {
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": round(conf, 4),
                        "bbox": {
                            "x1": round(x1, 1),
                            "y1": round(y1, 1),
                            "x2": round(x2, 1),
                            "y2": round(y2, 1),
                        },
                    }
                )
        return {
            "frame_id": frame_id,
            "timestamp": time.time(),
            "detections": detections,
            "count": len(detections),
        }

    # ------------------------------------------------------------------
    # Result dispatching
    # ------------------------------------------------------------------
    def _send_to_api(self, payload: dict):
        """POST results to the backend REST API."""
        if not self.api_url:
            return
        try:
            resp = requests.post(
                self.api_url,
                json=payload,
                timeout=5,
            )
            if resp.status_code != 200:
                print(f"[WARN] API returned status {resp.status_code}")
        except requests.RequestException as exc:
            print(f"[WARN] API request failed: {exc}")

    def _publish_mqtt(self, payload: dict):
        """Publish results and alerts to MQTT topics."""
        if not self.mqtt:
            return
        self.mqtt.publish("inspection/results", payload)
        # Publish alert if any defect detected
        if payload.get("count", 0) > 0:
            alert = {
                "timestamp": payload["timestamp"],
                "frame_id": payload["frame_id"],
                "defect_count": payload["count"],
                "defects": [
                    {"class": d["class_name"], "confidence": d["confidence"]}
                    for d in payload["detections"]
                ],
            }
            self.mqtt.publish("inspection/alerts", alert)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        """Start the inference loop."""
        print("[INFO] Starting inference loop. Press Ctrl+C to stop.")
        self.running = True
        frame_id = 0

        with self.camera:
            while self.running:
                frame = self.camera.read()
                if frame is None:
                    print("[WARN] Empty frame, retrying...")
                    time.sleep(0.01)
                    continue

                frame_id += 1

                # Run inference
                results = self.model.predict(
                    source=frame,
                    conf=self.confidence,
                    imgsz=self.img_size,
                    verbose=False,
                )

                self._update_fps()
                payload = self._format_results(results, frame_id)
                payload["fps"] = round(self._fps, 1)

                # Dispatch results
                self._send_to_api(payload)
                self._publish_mqtt(payload)

                # Console output every 30 frames
                if frame_id % 30 == 0:
                    print(
                        f"[INFO] Frame {frame_id} | "
                        f"FPS: {self._fps:.1f} | "
                        f"Detections: {payload['count']}"
                    )

                # Optional display
                if self.display:
                    annotated = results[0].plot() if results else frame
                    cv2.putText(
                        annotated,
                        f"FPS: {self._fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imshow("Jetson Inference", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.running = False

        self._cleanup()

    def _cleanup(self):
        """Release resources."""
        print("[INFO] Cleaning up...")
        if self.display:
            cv2.destroyAllWindows()
        if self.mqtt:
            self.mqtt.disconnect()
        print("[INFO] Shutdown complete.")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="NVIDIA Jetson edge inference for industrial defect detection"
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="0",
        help="Camera source: integer for USB/CSI, RTSP URL, or file path (default: 0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="weights/best.engine",
        help="Path to TensorRT engine or YOLO .pt model (default: weights/best.engine)",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="Backend API URL to POST results (e.g. http://localhost:8000/api/v1/inspections/)",
    )
    parser.add_argument(
        "--mqtt-broker",
        type=str,
        default=None,
        help="MQTT broker address (e.g. localhost)",
    )
    parser.add_argument(
        "--mqtt-port",
        type=int,
        default=1883,
        help="MQTT broker port (default: 1883)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Inference image size (default: 640)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show live annotated video window",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    engine = JetsonInference(
        model_path=args.model,
        camera_source=args.camera,
        api_url=args.api_url,
        mqtt_broker=args.mqtt_broker,
        mqtt_port=args.mqtt_port,
        confidence=args.confidence,
        img_size=args.img_size,
        display=args.display,
    )
    engine.run()


if __name__ == "__main__":
    main()
