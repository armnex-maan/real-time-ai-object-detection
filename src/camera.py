"""Webcam capture helper with graceful fallbacks."""

from __future__ import annotations

import time

import cv2
import numpy as np


class Camera:
    """A minimal webcam wrapper.

    Keeping this in its own module lets the detector and the Flask app stay
    ignorant of OpenCV capture quirks (backend selection, retries, …).
    """

    def __init__(
        self,
        source: int | str = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        # CAP_DSHOW is the fastest backend on Windows; fall back to default on
        # other platforms or if it misbehaves.
        if isinstance(self.source, int):
            cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self.source)
        else:
            cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video source {self.source!r}. "
                "Check that a webcam is connected and not in use."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap = cap

    def read(self) -> np.ndarray | None:
        if self.cap is None:
            self.open()
        assert self.cap is not None
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def iter_frames(camera: Camera, min_interval: float = 0.0):
    """Yield frames from ``camera`` forever, throttled to ``min_interval`` seconds."""
    last = 0.0
    while True:
        now = time.time()
        if min_interval and (now - last) < min_interval:
            time.sleep(min_interval - (now - last))
        frame = camera.read()
        if frame is None:
            time.sleep(0.05)
            continue
        last = time.time()
        yield frame
