"""Lightweight shared types.

Kept dependency-free so modules like :mod:`src.stats` can be imported — and
tested — without pulling in OpenCV or Ultralytics.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2
