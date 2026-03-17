"""Camera capture with threaded frame buffering for Jetson CSI/USB/RTSP."""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Camera:
    """Thread-safe camera capture with automatic RTSP reconnection.

    Supports:
      - NVIDIA CSI cameras via GStreamer (nvarguscamerasrc)
      - USB cameras (V4L2)
      - RTSP streams
      - Video files (looping)
    """

    _CSI_PIPELINE = (
        "nvarguscamerasrc sensor-id={sensor_id} ! "
        "video/x-raw(memory:NVMM), width={w}, height={h}, "
        "format=NV12, framerate={fps}/1 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, width={w}, height={h}, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
    )

    def __init__(
        self,
        source: Union[str, int] = "0",
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        use_csi: bool = False,
        csi_sensor_id: int = 0,
        reconnect_delay: float = 3.0,
        max_reconnects: int = 10,
    ):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.use_csi = use_csi
        self.csi_sensor_id = csi_sensor_id
        self.reconnect_delay = reconnect_delay
        self.max_reconnects = max_reconnects

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._is_rtsp = isinstance(source, str) and source.lower().startswith("rtsp")
        self._is_file = isinstance(source, str) and not source.isdigit() and not self._is_rtsp

    def start(self):
        self._open()
        self._stopped.clear()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        logger.info("Camera started: %s (%dx%d@%dfps)", self.source, self.width, self.height, self.fps)

    def stop(self):
        self._stopped.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._release()
        logger.info("Camera stopped")

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def _open(self):
        if self.use_csi:
            pipeline = self._CSI_PIPELINE.format(
                sensor_id=self.csi_sensor_id, w=self.width, h=self.height, fps=self.fps,
            )
            self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        elif self._is_rtsp:
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        else:
            src = int(self.source) if isinstance(self.source, str) and self.source.isdigit() else self.source
            self._cap = cv2.VideoCapture(src)

        if not self.is_opened:
            raise RuntimeError(f"Failed to open camera: {self.source}")

        if not self.use_csi:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    def _release(self):
        if self._cap:
            self._cap.release()
            self._cap = None

    def _reconnect(self) -> bool:
        for attempt in range(1, self.max_reconnects + 1):
            logger.warning("Reconnect attempt %d/%d...", attempt, self.max_reconnects)
            self._release()
            time.sleep(self.reconnect_delay)
            try:
                self._open()
                if self.is_opened:
                    logger.info("Reconnected to camera")
                    return True
            except RuntimeError:
                continue
        return False

    def _reader(self):
        failures = 0
        while not self._stopped.is_set():
            if not self.is_opened:
                if self._is_rtsp and self._reconnect():
                    failures = 0
                    continue
                break

            ret, frame = self._cap.read()
            if not ret:
                failures += 1
                if self._is_file:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    failures = 0
                elif self._is_rtsp and failures >= 30:
                    if not self._reconnect():
                        break
                    failures = 0
                continue

            failures = 0
            with self._lock:
                self._frame = frame
