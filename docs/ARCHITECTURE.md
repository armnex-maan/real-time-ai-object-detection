# Architecture

A short tour of how the pieces fit together.

```
┌──────────────┐   frames    ┌───────────────┐   detections   ┌──────────────┐
│   Camera     │ ──────────▶ │ ObjectDetector│ ─────────────▶ │ StatsTracker │
│  (OpenCV)    │             │  (YOLOv8)     │                └──────┬───────┘
└──────────────┘             └───────┬───────┘                       │
                                     │ annotated JPEG                │
                                     ▼                               ▼
                             ┌──────────────┐                 ┌──────────────┐
                             │ DetectionStream │ ─ MJPEG ──▶ │   Flask app   │
                             │  (worker thread)│  /video_feed │   /api/stats  │
                             └──────────────┘                 └──────────────┘
                                                                     │
                                                                     ▼
                                                             ┌──────────────┐
                                                             │   Browser    │
                                                             │  (dashboard) │
                                                             └──────────────┘
```

## Threading model

- **Worker thread** (`DetectionStream._run`) owns the camera and model. It reads
  frames, runs inference, draws boxes, encodes JPEG, and updates shared state.
- **Request threads** (Flask) read that state. `latest_jpeg()` returns the most
  recent annotated frame; `/api/stats` returns a `Snapshot`.
- A single lock guards shared state. At 30 FPS the lock is held for ~microseconds
  so contention is a non-issue.

## Why MJPEG and not WebSockets?

MJPEG is a one-liner in Flask (`multipart/x-mixed-replace`) and `<img>` renders
it natively — no client-side decoding, no framework. For a portfolio project
that's a better tradeoff than a WebRTC/WebSocket setup, which adds a lot of
surface area for negligible gain at LAN latencies.

## Extending

- **Different model.** `ObjectDetector` takes any Ultralytics path. Train a
  custom YOLOv8 on your data and point `--model` at the `.pt` file.
- **Different source.** `Camera` accepts an int (webcam index) or any string
  OpenCV understands — file paths, RTSP URLs, HTTP streams.
- **More metrics.** Add fields to `Snapshot` and render them in `dashboard.js`;
  both ends are intentionally small.
