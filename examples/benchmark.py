"""Quick benchmark: how many FPS does your machine get on this model?"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detector import ObjectDetector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (args.height, args.width, 3), dtype=np.uint8)

    detector = ObjectDetector(model_path=args.model, device=args.device)

    for _ in range(5):
        detector.detect(frame)

    t0 = time.time()
    for _ in range(args.iters):
        detector.detect(frame)
    elapsed = time.time() - t0

    fps = args.iters / elapsed
    print(f"Model:      {args.model}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Iterations: {args.iters}")
    print(f"Elapsed:    {elapsed:.2f}s")
    print(f"Throughput: {fps:.1f} FPS  ({1000 / fps:.1f} ms / frame)")


if __name__ == "__main__":
    main()
