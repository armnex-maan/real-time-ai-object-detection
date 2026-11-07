"""Flask app wiring the detection stream to a live dashboard."""

from __future__ import annotations

import os
from dataclasses import asdict

from flask import Flask, Response, jsonify, render_template

from .camera import Camera
from .detector import ObjectDetector
from .stabilizer import DetectionStabilizer
from .stats import StatsTracker
from .stream import DetectionStream


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )

    model = os.environ.get("YOLO_MODEL", "yolov8s.pt")
    source_env = os.environ.get("CAMERA_SOURCE", "0")
    try:
        source: int | str = int(source_env)
    except ValueError:
        source = source_env

    classes_env = os.environ.get("YOLO_CLASSES", "").strip()
    classes: list[str] | None = (
        [c.strip() for c in classes_env.split(",") if c.strip()]
        if classes_env
        else None
    )

    detector = ObjectDetector(
        model_path=model,
        conf=_env_float("YOLO_CONF", 0.5),
        iou=_env_float("YOLO_IOU", 0.45),
        imgsz=_env_int("YOLO_IMGSZ", 640),
        augment=os.environ.get("YOLO_AUGMENT", "0") == "1",
        device=os.environ.get("YOLO_DEVICE") or None,
        classes=classes,
    )
    camera = Camera(
        source=source,
        width=_env_int("CAMERA_WIDTH", 1280),
        height=_env_int("CAMERA_HEIGHT", 720),
        fps=_env_int("CAMERA_FPS", 30),
    )
    stats = StatsTracker()

    stabilizer: DetectionStabilizer | None
    if os.environ.get("STABILIZE", "1") != "0":
        stabilizer = DetectionStabilizer(
            min_hits=_env_int("STABILIZE_MIN_HITS", 3),
            max_misses=_env_int("STABILIZE_MAX_MISSES", 5),
        )
    else:
        stabilizer = None

    stream = DetectionStream(detector, camera, stats, stabilizer=stabilizer)

    app.config["stream"] = stream
    app.config["stats"] = stats
    app.config["detector"] = detector

    @app.route("/")
    def index():
        stream.start()
        return render_template(
            "index.html",
            model_name=model,
            class_names=list(detector.names.values()),
        )

    @app.route("/video_feed")
    def video_feed():
        stream.start()
        return Response(
            stream.mjpeg_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/api/stats")
    def api_stats():
        return jsonify(asdict(stats.snapshot()))

    @app.route("/api/detections")
    def api_detections():
        return jsonify(
            [
                {
                    "class_name": d.class_name,
                    "class_id": d.class_id,
                    "confidence": round(d.confidence, 3),
                    "box": [d.x1, d.y1, d.x2, d.y2],
                }
                for d in stream.latest_detections()
            ]
        )

    @app.route("/api/reset", methods=["POST"])
    def api_reset():
        stats.reset()
        return jsonify({"status": "ok"})

    @app.route("/api/health")
    def api_health():
        return jsonify({"status": "ok", "model": model})

    return app


app = create_app()
