"""Temporal stabilization for object detections.

Spurious one-frame detections (paper mistaken for a cell phone, a finger
flashing as a toothbrush) are one of the big complaints when running a
small model on a webcam. This module filters them out by requiring a
detection to show up in several consecutive frames before the dashboard
sees it, and keeps displaying it for a few frames after it disappears so
boxes don't flicker between adjacent inferences.

The matcher is a cheap IoU-based tracker: good enough for a webcam at
10–30 fps where objects move slowly between frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import Detection


def _iou(a: Detection, b: Detection) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, a.x2 - a.x1) * max(0, a.y2 - a.y1)
    area_b = max(0, b.x2 - b.x1) * max(0, b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


@dataclass
class _Track:
    detection: Detection
    hits: int = 0
    misses: int = 0
    confirmed: bool = False
    confidence_ema: float = field(default=0.0)


class DetectionStabilizer:
    """Require ``min_hits`` consecutive matches to confirm a detection.

    A confirmed detection survives up to ``max_misses`` frames of absence
    before being dropped — this removes most of the flicker without hiding
    objects that briefly leave the model's attention.
    """

    def __init__(
        self,
        min_hits: int = 3,
        max_misses: int = 5,
        iou_threshold: float = 0.3,
        ema_alpha: float = 0.6,
    ) -> None:
        self.min_hits = min_hits
        self.max_misses = max_misses
        self.iou_threshold = iou_threshold
        self.ema_alpha = ema_alpha
        self._tracks: list[_Track] = []

    def update(self, detections: list[Detection]) -> list[Detection]:
        used = [False] * len(detections)

        for track in self._tracks:
            best_i = -1
            best_iou = self.iou_threshold
            for i, det in enumerate(detections):
                if used[i] or det.class_id != track.detection.class_id:
                    continue
                score = _iou(track.detection, det)
                if score > best_iou:
                    best_iou = score
                    best_i = i
            if best_i >= 0:
                used[best_i] = True
                new = detections[best_i]
                track.hits += 1
                track.misses = 0
                track.confidence_ema = (
                    self.ema_alpha * new.confidence
                    + (1 - self.ema_alpha) * track.confidence_ema
                )
                track.detection = Detection(
                    class_id=new.class_id,
                    class_name=new.class_name,
                    confidence=track.confidence_ema,
                    x1=new.x1, y1=new.y1, x2=new.x2, y2=new.y2,
                )
                if track.hits >= self.min_hits:
                    track.confirmed = True
            else:
                track.misses += 1

        for i, det in enumerate(detections):
            if used[i]:
                continue
            self._tracks.append(
                _Track(
                    detection=det,
                    hits=1,
                    confidence_ema=det.confidence,
                )
            )

        self._tracks = [t for t in self._tracks if t.misses <= self.max_misses]

        return [t.detection for t in self._tracks if t.confirmed]

    def reset(self) -> None:
        self._tracks.clear()
