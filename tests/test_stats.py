"""Tests for the stats tracker — pure Python, no camera/model needed."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stats import StatsTracker
from src.types import Detection


def _det(name: str, class_id: int = 0) -> Detection:
    return Detection(
        class_id=class_id,
        class_name=name,
        confidence=0.9,
        x1=0, y1=0, x2=10, y2=10,
    )


def test_initial_snapshot_is_empty():
    s = StatsTracker().snapshot()
    assert s.frames_processed == 0
    assert s.fps == 0.0
    assert s.current_counts == {}
    assert s.total_counts == {}


def test_update_accumulates_totals_and_tracks_current():
    tracker = StatsTracker()
    tracker.update([_det("person"), _det("person"), _det("car", class_id=2)])
    tracker.update([_det("person"), _det("dog", class_id=16)])

    snap = tracker.snapshot()
    assert snap.frames_processed == 2
    assert snap.current_counts == {"person": 1, "dog": 1}
    assert snap.total_counts == {"person": 3, "car": 1, "dog": 1}
    assert snap.top_classes[0] == ("person", 3)


def test_fps_is_positive_after_a_few_frames():
    tracker = StatsTracker()
    for _ in range(5):
        tracker.update([_det("person")])
        time.sleep(0.01)
    assert tracker.snapshot().fps > 0.0


def test_reset_clears_everything():
    tracker = StatsTracker()
    tracker.update([_det("person")])
    tracker.reset()
    snap = tracker.snapshot()
    assert snap.frames_processed == 0
    assert snap.total_counts == {}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
