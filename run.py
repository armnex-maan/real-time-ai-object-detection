"""Launch the dashboard.

Usage:
    python run.py                # defaults: webcam 0, yolov8n, port 5000
    python run.py --port 8080    # custom port
    python run.py --model yolov8s.pt --source 1

Environment variables (also accepted):
    YOLO_MODEL, YOLO_CONF, YOLO_IOU, YOLO_DEVICE
    CAMERA_SOURCE, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS
"""

from __future__ import annotations

import argparse
import os

from src.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time YOLO dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--model", default=None, help="YOLO model (e.g. yolov8s.pt)")
    parser.add_argument("--source", default=None, help="Camera index or video path")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold (default 0.5)")
    parser.add_argument("--imgsz", type=int, default=None, help="Inference resolution (default 640)")
    parser.add_argument("--augment", action="store_true", help="Enable test-time augmentation (slower, more accurate)")
    parser.add_argument("--classes", default=None, help="Comma-separated COCO class names to keep, e.g. 'person,car,dog'")
    parser.add_argument("--no-stabilize", action="store_true", help="Disable the temporal stabilizer")
    parser.add_argument("--device", default=None, help="Torch device (cpu / cuda / mps)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.model:
        os.environ["YOLO_MODEL"] = args.model
    if args.source is not None:
        os.environ["CAMERA_SOURCE"] = args.source
    if args.conf is not None:
        os.environ["YOLO_CONF"] = str(args.conf)
    if args.imgsz is not None:
        os.environ["YOLO_IMGSZ"] = str(args.imgsz)
    if args.augment:
        os.environ["YOLO_AUGMENT"] = "1"
    if args.classes:
        os.environ["YOLO_CLASSES"] = args.classes
    if args.no_stabilize:
        os.environ["STABILIZE"] = "0"
    if args.device:
        os.environ["YOLO_DEVICE"] = args.device

    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
