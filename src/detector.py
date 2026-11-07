"""YOLOv8 object detector wrapper.

Wraps Ultralytics YOLO so the rest of the app only talks to a tiny API:
run `detect(frame)` and get back a list of structured detections plus the
annotated frame. Weights are downloaded on first use and cached by Ultralytics.
"""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np
from ultralytics import YOLO

from .types import Detection

__all__ = ["Detection", "ObjectDetector"]


# A stable palette so the same class always gets the same color.
_PALETTE = [
    (56, 56, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
    (10, 249, 72), (23, 204, 146), (134, 219, 61), (52, 147, 26),
    (187, 212, 0), (168, 153, 44), (255, 194, 0), (147, 69, 52),
    (255, 115, 100), (236, 24, 0), (255, 56, 132), (133, 0, 82),
    (255, 56, 203), (200, 149, 255), (199, 55, 255),
]


def _color_for(class_id: int) -> tuple[int, int, int]:
    return _PALETTE[class_id % len(_PALETTE)]


class ObjectDetector:
    """Thin wrapper around a YOLO model.

    Parameters
    ----------
    model_path:
        Path or model name understood by Ultralytics (e.g. ``yolov8s.pt``).
        Defaults to ``yolov8s.pt`` — the "small" variant: still CPU-friendly
        but notably more accurate than ``yolov8n`` on real-world footage.
    conf:
        Minimum confidence required to keep a detection. Default 0.5 keeps
        false positives (paper mistaken for a cell phone, finger for a
        toothbrush) out of the feed.
    iou:
        IoU threshold for non-max suppression.
    imgsz:
        Inference resolution. Higher = more accurate on small objects but
        slower; 640 is YOLO's native size and a good default.
    augment:
        Test-time augmentation. Slower but squeezes extra accuracy out of
        small models — worth it on CPU with a static scene.
    device:
        Torch device string. ``None`` lets Ultralytics pick (cuda if available).
    classes:
        Optional allow-list of class IDs or names. ``None`` keeps all 80 COCO
        classes. Example: ``["person", "car", "dog"]`` or ``[0, 2, 16]``.
    """

    def __init__(
        self,
        model_path: str = "yolov8s.pt",
        conf: float = 0.5,
        iou: float = 0.45,
        imgsz: int = 640,
        augment: bool = False,
        device: str | None = None,
        classes: Iterable[int | str] | None = None,
    ) -> None:
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.augment = augment
        self.device = device
        self.names: dict[int, str] = self.model.names
        self.classes = self._resolve_classes(classes)

    def _resolve_classes(
        self, classes: Iterable[int | str] | None
    ) -> list[int] | None:
        if classes is None:
            return None
        name_to_id = {v.lower(): k for k, v in self.names.items()}
        ids: list[int] = []
        for c in classes:
            if isinstance(c, int):
                ids.append(c)
                continue
            s = str(c).strip().lower()
            if not s:
                continue
            if s.isdigit():
                ids.append(int(s))
                continue
            if s not in name_to_id:
                raise ValueError(
                    f"Unknown COCO class {c!r}. Valid names: "
                    f"{sorted(name_to_id)}"
                )
            ids.append(name_to_id[s])
        return ids or None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            augment=self.augment,
            device=self.device,
            classes=self.classes,
            verbose=False,
        )
        detections: list[Detection] = []
        if not results:
            return detections

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return detections

        xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, cls):
            detections.append(
                Detection(
                    class_id=int(k),
                    class_name=self.names.get(int(k), str(k)),
                    confidence=float(c),
                    x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                )
            )
        return detections

    @staticmethod
    def annotate(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Draw boxes and labels on a copy of ``frame``."""
        out = frame.copy()
        for d in detections:
            color = _color_for(d.class_id)
            cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), color, 2)

            label = f"{d.class_name} {d.confidence:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            cv2.rectangle(
                out,
                (d.x1, max(d.y1 - th - baseline - 4, 0)),
                (d.x1 + tw + 6, d.y1),
                color,
                -1,
            )
            cv2.putText(
                out,
                label,
                (d.x1 + 3, d.y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return out
