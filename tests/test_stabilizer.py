"""Tests for the temporal stabilizer — no camera, no model."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stabilizer import DetectionStabilizer
from src.types import Detection


def _det(name: str, class_id: int, x: int, conf: float = 0.9) -> Detection:
    return Detection(
        class_id=class_id,
        class_name=name,
        confidence=conf,
        x1=x, y1=10, x2=x + 50, y2=60,
    )


def test_single_frame_detection_is_suppressed():
    s = DetectionStabilizer(min_hits=3)
    out = s.update([_det("person", 0, 100)])
    assert out == []


def test_detection_confirmed_after_three_matches():
    s = DetectionStabilizer(min_hits=3)
    for _ in range(2):
        assert s.update([_det("person", 0, 100)]) == []
    out = s.update([_det("person", 0, 102)])
    assert len(out) == 1
    assert out[0].class_name == "person"


def test_confirmed_detection_survives_brief_disappearance():
    s = DetectionStabilizer(min_hits=2, max_misses=3)
    s.update([_det("person", 0, 100)])
    s.update([_det("person", 0, 101)])
    for _ in range(3):
        kept = s.update([])
        assert len(kept) == 1
    assert s.update([]) == []


def test_classes_do_not_cross_match():
    s = DetectionStabilizer(min_hits=2)
    s.update([_det("person", 0, 100)])
    out = s.update([_det("cell phone", 67, 100)])
    assert out == []


def test_confidence_uses_ema():
    s = DetectionStabilizer(min_hits=1, ema_alpha=0.5)
    s.update([_det("person", 0, 100, conf=1.0)])
    out = s.update([_det("person", 0, 102, conf=0.0)])
    assert 0.4 < out[0].confidence < 0.6


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
