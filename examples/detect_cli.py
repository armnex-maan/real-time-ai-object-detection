"""Headless detection demo — no Flask, no browser.

Opens the webcam, runs YOLO, and shows an OpenCV window. Press Q to quit.

    python examples/detect_cli.py
    python examples/detect_cli.py --source path/to/video.mp4 --model yolov8s.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.camera import Camera
from src.detector import ObjectDetector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Camera index or video path")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    args = parser.parse_args()

    try:
        source: int | str = int(args.source)
    except ValueError:
        source = args.source

    detector = ObjectDetector(model_path=args.model, conf=args.conf)
    frames = 0
    t0 = time.time()

    with Camera(source=source) as cam:
        while True:
            frame = cam.read()
            if frame is None:
                break
            detections = detector.detect(frame)
            annotated = detector.annotate(frame, detections)

            frames += 1
            fps = frames / max(1e-6, time.time() - t0)
            cv2.putText(
                annotated,
                f"{fps:.1f} FPS  |  {len(detections)} objects",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 170),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Real-Time AI Object Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
