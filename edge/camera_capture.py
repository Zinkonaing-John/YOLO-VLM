#!/usr/bin/env python3
"""
Multi-source camera capture with threaded frame buffering.

Supports CSI (GStreamer), USB, RTSP streams, and video files.
"""

import threading
import time
from typing import Optional, Union

import cv2
import numpy as np


class CameraCapture:
    """Thread-safe camera capture with automatic reconnection for RTSP streams.

    Usage::

        with CameraCapture(source="rtsp://192.168.1.10/stream", width=1280, height=720) as cam:
            while True:
                frame = cam.read()
                if frame is not None:
                    process(frame)
    """

    # GStreamer pipeline template for NVIDIA CSI cameras (e.g. Jetson Nano / Orin)
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
        buffer_size: int = 2,
        reconnect_delay: float = 3.0,
        max_reconnect_attempts: int = 10,
        use_csi: bool = False,
        csi_sensor_id: int = 0,
    ):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self.use_csi = use_csi
        self.csi_sensor_id = csi_sensor_id

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._is_rtsp = isinstance(source, str) and source.lower().startswith("rtsp")
        self._is_file = isinstance(source, str) and not source.isdigit() and not self._is_rtsp

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self):
        """Open the capture device and start the background reader thread."""
        self._open_capture()
        self._stopped.clear()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal the reader thread to stop and release resources."""
        self._stopped.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._release_capture()

    def read(self) -> Optional[np.ndarray]:
        """Return the most recent frame, or None if unavailable."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _build_source(self):
        """Return the OpenCV-compatible source identifier."""
        if self.use_csi:
            pipeline = self._CSI_PIPELINE.format(
                sensor_id=self.csi_sensor_id,
                w=self.width,
                h=self.height,
                fps=self.fps,
            )
            return pipeline

        # Integer source (USB camera index)
        if isinstance(self.source, int):
            return self.source
        if self.source.isdigit():
            return int(self.source)

        # RTSP or file path — return as-is
        return self.source

    def _open_capture(self):
        """Open a cv2.VideoCapture with the appropriate backend."""
        src = self._build_source()

        if self.use_csi:
            self._cap = cv2.VideoCapture(src, cv2.CAP_GSTREAMER)
        elif self._is_rtsp:
            self._cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        else:
            self._cap = cv2.VideoCapture(src)

        if not self.is_opened:
            raise RuntimeError(f"Failed to open camera source: {self.source}")

        # Apply settings for USB cameras / files
        if not self.use_csi:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

        print(
            f"[CameraCapture] Opened source={self.source} "
            f"({self.width}x{self.height} @ {self.fps}fps)"
        )

    def _release_capture(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _reconnect(self) -> bool:
        """Attempt to reconnect to an RTSP stream."""
        for attempt in range(1, self.max_reconnect_attempts + 1):
            print(
                f"[CameraCapture] Reconnect attempt {attempt}/{self.max_reconnect_attempts}..."
            )
            self._release_capture()
            time.sleep(self.reconnect_delay)
            try:
                self._open_capture()
                if self.is_opened:
                    print("[CameraCapture] Reconnected successfully.")
                    return True
            except RuntimeError:
                continue
        print("[CameraCapture] Exceeded max reconnect attempts.")
        return False

    def _reader_loop(self):
        """Background thread: continuously grab frames into the buffer."""
        consecutive_failures = 0
        max_consecutive_failures = 30

        while not self._stopped.is_set():
            if not self.is_opened:
                if self._is_rtsp and self._reconnect():
                    consecutive_failures = 0
                    continue
                else:
                    break

            ret, frame = self._cap.read()
            if not ret:
                consecutive_failures += 1
                if self._is_file:
                    # End of video file — loop or stop
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    consecutive_failures = 0
                    continue
                if self._is_rtsp and consecutive_failures >= max_consecutive_failures:
                    print("[CameraCapture] Too many read failures. Attempting reconnect...")
                    if not self._reconnect():
                        break
                    consecutive_failures = 0
                continue

            consecutive_failures = 0
            with self._lock:
                self._frame = frame

        print("[CameraCapture] Reader loop exited.")
