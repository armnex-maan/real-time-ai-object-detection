"""Thread-safe statistics tracking for the detection stream."""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from dataclasses import dataclass

from .types import Detection


@dataclass
class Snapshot:
    fps: float
    frames_processed: int
    uptime_seconds: float
    current_counts: dict[str, int]
    total_counts: dict[str, int]
    top_classes: list[tuple[str, int]]
    recent_fps: list[float]


class StatsTracker:
    """Rolling statistics over the detection stream.

    The dashboard polls :meth:`snapshot`; the detection loop pushes frames in
    with :meth:`update`. Both sides are protected by a single lock, which is
    plenty for 30 fps.
    """

    def __init__(self, fps_window: int = 60) -> None:
        self._lock = threading.Lock()
        self._start = time.time()
        self._frames = 0
        self._frame_times: deque[float] = deque(maxlen=fps_window)
        self._current: Counter[str] = Counter()
        self._totals: Counter[str] = Counter()
        self._fps_history: deque[float] = deque(maxlen=120)

    def update(self, detections: list[Detection]) -> None:
        now = time.time()
        with self._lock:
            self._frames += 1
            self._frame_times.append(now)

            self._current = Counter(d.class_name for d in detections)
            for name, count in self._current.items():
                self._totals[name] += count

            fps = self._fps_locked()
            self._fps_history.append(round(fps, 2))

    def _fps_locked(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        if span <= 0:
            return 0.0
        return (len(self._frame_times) - 1) / span

    def snapshot(self) -> Snapshot:
        with self._lock:
            return Snapshot(
                fps=round(self._fps_locked(), 2),
                frames_processed=self._frames,
                uptime_seconds=round(time.time() - self._start, 1),
                current_counts=dict(self._current),
                total_counts=dict(self._totals),
                top_classes=self._totals.most_common(8),
                recent_fps=list(self._fps_history),
            )

    def reset(self) -> None:
        with self._lock:
            self._start = time.time()
            self._frames = 0
            self._frame_times.clear()
            self._current.clear()
            self._totals.clear()
            self._fps_history.clear()
