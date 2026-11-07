"""End-to-end demo on a static image — no webcam needed.

Downloads a classic Ultralytics sample (two people on bicycles with cars
in the background), runs detection, and writes an annotated output image
plus a text report. Use it to verify the pipeline without touching your
camera.

    python examples/demo_image.py
    python examples/demo_image.py --image path/to/your.jpg --model yolov8s.pt
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detector import ObjectDetector

SAMPLE_URL = "https://ultralytics.com/images/bus.jpg"


def _download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading sample image from {url}")
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r, dest.open("wb") as f:
                f.write(r.read())
            return dest
        except Exception as e:
            last_err = e
            print(f"  attempt {attempt + 1} failed: {e}; retrying...")
    raise RuntimeError(f"Could not download sample image: {last_err}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="Image path (default: download sample)")
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--classes", default=None, help="Comma-separated class names")
    parser.add_argument("--out", default="demo_output.jpg")
    args = parser.parse_args()

    if args.image is None:
        sample = Path(__file__).resolve().parents[1] / "docs" / "sample.jpg"
        image_path = _download(SAMPLE_URL, sample)
    else:
        image_path = Path(args.image)

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise SystemExit(f"Could not read image: {image_path}")

    classes = (
        [c.strip() for c in args.classes.split(",") if c.strip()]
        if args.classes
        else None
    )
    detector = ObjectDetector(
        model_path=args.model, conf=args.conf, classes=classes
    )

    detections = detector.detect(frame)
    annotated = detector.annotate(frame, detections)

    out_path = Path(args.out)
    cv2.imwrite(str(out_path), annotated)

    print(f"\nImage: {image_path}  ({frame.shape[1]}x{frame.shape[0]})")
    print(f"Model: {args.model}  (conf>={args.conf})")
    print(f"Detections: {len(detections)}")
    for d in detections:
        print(f"  - {d.class_name:<14} {d.confidence:.2f}  box=({d.x1},{d.y1})-({d.x2},{d.y2})")
    print(f"\nAnnotated image written to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
