"""Background thread that reads the webcam, runs detection, and exposes frames."""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from .camera import Camera
from .detector import ObjectDetector
from .stabilizer import DetectionStabilizer
from .types import Detection
from .stats import StatsTracker


class DetectionStream:
    """Runs capture + inference on a worker thread.

    The Flask app reads the latest annotated JPEG via :meth:`latest_jpeg`, and
    :meth:`stats` for the live dashboard. The worker is lazily started on the
    first call so ``flask --debug`` reloads don't spawn two capture loops.
    """

    def __init__(
        self,
        detector: ObjectDetector,
        camera: Camera,
        stats: StatsTracker,
        stabilizer: DetectionStabilizer | None = None,
        jpeg_quality: int = 80,
    ) -> None:
        self.detector = detector
        self.camera = camera
        self.stats = stats
        self.stabilizer = stabilizer
        self.jpeg_quality = jpeg_quality

        self._lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._latest_detections: list[Detection] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.camera.release()

    def _run(self) -> None:
        import sys
        import traceback
        try:
            self.camera.open()
            while not self._stop.is_set():
                frame = self.camera.read()
                if frame is None:
                    time.sleep(0.02)
                    continue

                detections = self.detector.detect(frame)
                if self.stabilizer is not None:
                    detections = self.stabilizer.update(detections)
                annotated = self.detector.annotate(frame, detections)

                ok, buf = cv2.imencode(
                    ".jpg",
                    annotated,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                )
                if not ok:
                    continue

                with self._lock:
                    self._latest_jpeg = buf.tobytes()
                    self._latest_detections = detections

                self.stats.update(detections)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            raise
        finally:
            self.camera.release()

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def latest_detections(self) -> list[Detection]:
        with self._lock:
            return list(self._latest_detections)

    def mjpeg_frames(self, boundary: bytes = b"frame"):
        """Yield multipart MJPEG chunks for Flask's streaming response."""
        placeholder = _placeholder_jpeg()
        while not self._stop.is_set():
            jpeg = self.latest_jpeg() or placeholder
            yield (
                b"--" + boundary + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                + jpeg + b"\r\n"
            )
            time.sleep(1 / 30)


def _placeholder_jpeg() -> bytes:
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(
        img,
        "Warming up...",
        (180, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()
